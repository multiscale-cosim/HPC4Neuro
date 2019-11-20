# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for details)

"""
    A collection of custom error classes.

"""


class BaseHpc4nsError(Exception):
    """ Base class for all error classes in the module. """


class MpiInitError(BaseHpc4nsError):
    """ Raised if MPI initialization fails. """


class DataDistributionError(BaseHpc4nsError):
    """ Raised if data distribution amongst MPI ranks fails. """
