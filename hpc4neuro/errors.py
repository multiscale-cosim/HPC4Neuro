# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for
# details)

"""
    A collection of custom error classes.

"""


class BaseHpc4neuroError(Exception):
    """ Base class for all error classes in the module. """


class MpiInitError(BaseHpc4neuroError):
    """ Raised if MPI initialization fails. """


class DataDistributionError(BaseHpc4neuroError):
    """ Raised if data distribution amongst MPI ranks fails. """
