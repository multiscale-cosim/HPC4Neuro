# The SLNS library of Python utilities

This project brings together a collection of utilities that have been factored
out from different projects. In certain cases we need a specific functionality
that does not appear to be available in existing packages. We therefore develop
the required code in-house, and if it appears to be general enough, we add it
to this collection in the hope that it may be useful to others.

The following utilities are available at this time:

## `slns.distribution`

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

The `slns.distribution` module provides a high-level interface for data distribution
with MPI, without the explicit need to write MPI code on the user's part. The
following examples show what the module does, and how it can be useful.

### Examples

Consider the following code that defines a simple function which returns a list of files 
read from a given directory. 

```
import os

def get_filenames(path):
    return os.listdir(path)

# List of the filenames in the 'slns' directory
filenames = get_filenames('./slns')
```

#### Distributed case 1: Using the static decorator syntax

Now consider a scenario in which we need to run this code on multiple processors across
multiple nodes in a cluster, and distribute the returned filenames across all the processes. 
The following example shows how the `slns.distribution` module can help with that.

```
import os
from mpi4py import MPI

from slns.distribution import DataDistributor

@DataDistributor(MPI.COMM_WORLD)
def get_filenames(path):
    return os.listdir(path)

# List of rank-local file names
filenames = get_filenames('./slns')
```

`DataDistributor` decorates the `get_filenames` function such that calling
the function returns only a subset of filenames that are to be processed by the
local MPI rank. All the MPI communication required for distribution of filenames
is hidden from the user.

### Distributed case 2: Dynamically decorating a function

In certain scenarios it is not possible to statically decorate a function using
the decorator syntax, e.g., when the MPI communicator object is not available
at the time of function definition. The following example demonstrates the use
of `DataDistributor` in such cases.

```
import os
from mpi4py import MPI

from slns.distribution import DataDistributor

# Initialize the decorator
dist_decorator = DataDistributor(MPI.COMM_WORLD)

# Decorate the function that reads a list of filenames.
get_rank_local_filenames = dist_decorator(os.listdir)

# Use the decorated function to get the rank-local list of filenames
filenames = get_rank_local_filenames('./slns')
```

### Support for graceful application shutdown

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
