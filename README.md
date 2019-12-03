# The hpc4neuro library of Python utilities

This project brings together a collection of utilities that have been factored
out from different projects. In certain cases we need a specific functionality
that does not appear to be available in existing packages. We therefore develop
the required code in-house, and if it appears to be general enough, we add it
to this collection in the hope that it may be useful to others.

## Setup and Requirements

The `hpc4neuro` package requires `Python 3.6` or above. To install, please
use the following command:

```
python -m pip install git+https://gitlab.version.fz-juelich.de/hpc4neuro/hpc4neuro_utils.git
```

## Available modules

The following modules are available at this time:

### 1. `hpc4neuro.distribution`

**Note:** This module requires `mpi4py`. To install `mpi4py`, please 
follow installation instructions available 
[here](https://mpi4py.readthedocs.io/en/stable/install.html). 

This module exposes the following two classes:
1.  `DataDistributor`
2.  `ErrorHandler`

Any function that returns a *sized iterable* (i.e., an object that supports `iter()` 
and `len()`, e.g., `list`), can be be decorated by `DataDistributor` to seamlessly 
distribute items in the resulting object across all participating MPI ranks.
Moreover, `ErrorHandler` implements exception handling functions that ensure
graceful application termination via synchronization of all MPI ranks. 

The primary motivation for creating this module was to hide the details of distributing
training/validation data amongst MPI ranks when training deep artificial neural
networks in a data-parallel fashion using Horovod. Even though Horovod hides the
intricate details of distributed training, proper distribution of training/validation
data is only possible via MPI programming.

The `hpc4neuro.distribution` module provides a high-level interface for data distribution
with MPI, without the explicit need to write MPI code on the user's part. The
following examples show what the module does, and how it can be useful.

#### Examples

**Note:** All examples are available in the `hpc4neuro.examples.distribution` package.

Consider the following code that defines a simple function which returns a list of files 
read from a given directory. 

```
import os

def get_filenames(path):
    return os.listdir(path)

# List of the filenames in the 'hpc4neuro' directory
filenames = get_filenames('./hpc4neuro')
```

##### Distributed case 1: Using the static decorator syntax

Now consider a scenario in which we need to run this code on multiple processors across
multiple nodes in a cluster, and distribute the returned filenames across all the processes. 
The following example shows how the `hpc4neuro.distribution` module can help with that.

```
import os
from mpi4py import MPI

from hpc4neuro.distribution import DataDistributor

@DataDistributor(MPI.COMM_WORLD)
def get_filenames(path):
    return os.listdir(path)

# List of rank-local file names
filenames = get_filenames('./hpc4neuro')
```

`DataDistributor` decorates the `get_filenames` function such that calling
the function returns only a subset of filenames that are to be processed by the
local MPI rank. All the MPI communication required for distribution of filenames
is hidden from the user.

##### Distributed case 2: Dynamically decorating a function

In certain scenarios it is not possible to statically decorate a function using
the decorator syntax, e.g., when the MPI communicator object is not available
at the time of function definition. The following example demonstrates the use
of `DataDistributor` in such cases.

```
import os
from mpi4py import MPI

from hpc4neuro.distribution import DataDistributor

# Initialize the decorator
dist_decorator = DataDistributor(MPI.COMM_WORLD)

# Decorate the function that reads a list of filenames.
get_rank_local_filenames = dist_decorator(os.listdir)

# Use the decorated function to get the rank-local list of filenames
filenames = get_rank_local_filenames('./hpc4neuro')
```

#### Support for graceful application shutdown

A function to be decorated by `DataDistributor`, such as `os.listdir` in the examples
above, may raise an exception. Moreover, exceptions may be raised by `DataDistributor`
due to other errors. In both cases, if an exception is raised by one MPI rank, the
other MPI ranks may get stuck in a waiting state, unaware of the raised exception. To
handle such a scenario and ensure graceful termination of the application, a flag can
be set in the `DataDistributor` initializer to enable graceful application shutdown on
error. The following code examples illustrate how to enable this feature with both the
static and dynamic decoration syntax:

**Static:** `@DataDistributor(MPI.COMM_WORLD, shutdown_on_error=True)`

**Dynamic:** `dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=True)`

#### API documentation

API documentation for `hpc4neuro.distribution` is available [here](doc/text/index.txt).

## Notes for contributors

### Development setup

1.  Clone this repository
2.  Change to the cloned repository directory
3.  Create and activate a virtual environment
4.  If you use [`poetry`](https://github.com/sdispater/poetry), run `poetry install` to install 
all the required dependencies



### Test setup

[`pytest`](https://docs.pytest.org/en/latest/) is required for running and working with test code 
for this project.

Use the following command to run tests:

`mpirun -np <n> python -m pytest`

where `<n>` should be replaced with the number of MPI ranks to use for testing.