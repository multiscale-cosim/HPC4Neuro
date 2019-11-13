# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for details)

"""
    A collection of custom error classes.

"""


class BaseHpcnsError(Exception):
    """ Base class for all error classes in the module. """


class MpiInitError(BaseHpcnsError):
    """ Raised if MPI initialization fails. """


class DataDistributionError(BaseHpcnsError):
    """ Raised if data distribution amongst MPI ranks fails. """
