# The SLNS library of Python utilities

This project brings together a collection of utilities that have been factored
out from different projects. In certain cases we need a specific functionality
that does not appear to be available in existing packages. We therefore develop
the required code in-house, and if it appears to be general enough, we add it
to this collection.

The following utilities are available at this time:

## `slns.distribution`

This module exposes the following two classes:
1.  `DataDistributor`
2.  `ErrorHandler`



This module provides functions that simplify distribution of data using MPI. The
primary motivation for creating this module was to hide the details of distributing
training/validation data amongst MPI ranks when training deep artificial neural
networks in a data-parallel fashion using Horovod. Even though Horovod hides the
intricate details of distributed training, proper distribution of training/validation
data is only possible via MPI programming.

The `slns.distribution` module provides a high-level interface for data distribution
with MPI, without the explicit need to write MPI code on the user's part. Let's take
a look at a few examples to what the module does, and how it can be useful.

### Base case: No MPI

The following code defines a function that returns a list of files read from the
given directory. 

```
import os

def get_filenames(path):
    return os.listdir(path)

# List of the filenames in the 'slns' directory
filenames = get_filenames('./slns')

```

### Distributed case 1: Using the static decorator syntax
Now consider a scenario where we want to run this code on multiple processors across
multiple nodes in a cluster, and distribute the filenames across the processes. The
following example shows how the `slns.distribution` module helps with that.

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

The `DataDistributor` decorates the `get_filenames` function such that calling
the function returns only a subset of filenames that are to be processed by the
local rank. All the MPI communication required for distribution of the filenames
happens behind the scenes.

### Distributed case 2: Dynamically decorating a function

In certain scenarios it is not possible to statically decorate a function using
the decorator syntax, e.g., when the MPI communication object is not available
at the time of function definition. The following example shows how the
`DataDistributor` can be used in such cases.

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
