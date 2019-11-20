# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for details)

"""
    This program demonstrates how a library function can be decorated
    using the hpc4ns.distribution.DataDistributor.

    The syntax for decoration presented in this program can be useful not
    only for library functions, but also in situations where a reference
    to the MPI object is not available at the time of function definition.

    To run this sample from the repository root, use the following
    command on your workstation/laptop:

    mpirun -np 3 python -u -m hpc4ns.examples.distribution.dynamic_filenames_decorator

"""

import os
from mpi4py import MPI

from hpc4ns.distribution import DataDistributor


# Initialize the decorator, and get a reference to the object.
dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=True)

# Decorate the function that reads a list of filenames.
get_rank_local_filenames = dist_decorator(os.listdir)

# Use the decorated function to
filenames = get_rank_local_filenames('.')

print(f'Rank: {MPI.COMM_WORLD.Get_rank()} -- Filenames: {filenames}')
