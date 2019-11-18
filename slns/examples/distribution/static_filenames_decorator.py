# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for details)

"""
    This program demonstrates how a function that reads and returns
    a list of filenames can be easily decorated to automatically
    distribute the list across multiple MPI ranks.

    The slns.distribution.DataDistributor is used as
    the decorator.

    To run this sample from the repository root, use the following
    command on your workstation/laptop:

    mpirun -np 3 python -u -m slns.examples.distribution.static_filenames_decorator

"""

import os
from mpi4py import MPI

from slns.distribution import DataDistributor


# Decorate the function so that instead of returning the list
# of all filenames, it returns only the subset of filenames
# that should be processed by the local MPI rank.
@DataDistributor(MPI.COMM_WORLD, shutdown_on_error=True)
def get_filenames(path):
    """
    Reads and returns the list of filenames for all files
    available on 'path'.

    :param path: A valid path to an existing directory.

    :return: List. Each element is a filename as str.
    """

    return os.listdir(path)


# List of rank-local file names
filenames = get_filenames('.')

print(f'Rank: {MPI.COMM_WORLD.Get_rank()} -- Filenames: {filenames}')
