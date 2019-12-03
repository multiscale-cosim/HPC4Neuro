# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for details)

"""
    Illustration of a sequential program that defines and uses
    a function to read a list of filenames from the disk.

    To run this sample from the repository root, use the following
    command on your workstation/laptop:

    python -u -m hpc4neuro.examples.distribution.sequential_filenames

"""

import os


def get_filenames(path):
    """
    Reads and returns the list of filenames for all files
    available on 'path'.

    :param path: A valid path to an existing directory.

    :return: List. Each element is a filename as str.

    """

    return os.listdir(path)


# List of the filenames in the current directory
filenames = get_filenames('.')

print(f'Filenames: {filenames}')
