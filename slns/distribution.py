# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for details)

"""
    Utilities for seamless distribution of data across multiple MPI ranks.

"""

import sys
import enum
import random
import itertools
import functools

from slns.errors import DataDistributionError


@enum.unique
class Constants(enum.Enum):
    """
    Constants useful for communicating result status amongst
    MPI ranks.

    """

    STATUS = enum.auto()
    MESSAGE = enum.auto()
    SUCCESS = enum.auto()
    FAILURE = enum.auto()


class ErrorHandler:
    # pylint: disable=too-few-public-methods
    """
    Set of functions that can gracefully handle exceptions/errors
    in MPI programs running with one or more ranks.

    """

    @staticmethod
    def exit_all_ranks(error, mpi_comm):
        """
        Ensures a clean program exit by synchronizing all MPI ranks
        available via the given communicator. Also, writes the
        message string from the given error object to stderr.

        :param error: Object, either of type Exception, or a type derived
                      from Exception.
        :param mpi_comm: Object of type mpi4py.MPI.Comm that represents the
                         communicator for all the ranks to which the given
                         error matters.

        """

        mpi_comm.Barrier()

        if mpi_comm.Get_rank() == 0:
            print(f'\n***** Fatal error: {error} *****', file=sys.stderr)

        mpi_comm.Barrier()

        sys.exit(1)


class DataDistributor:
    """
    A callable that serves as a decorator for any function that returns an
    iterable of data items, and needs its functionality to be extended such that
    the iterable returned is not the entire collection, but rather only
    a part of the collection that is to be processed by the local MPI rank.

    Therefore, this decorator converts a function into one that can be used
    for distributed processing with MPI.

    """

    def __init__(self, mpi_comm, shutdown_on_error=False):
        """
        Initialize the object.

        :param mpi_comm: Object of type mpi4py.MPI.Comm to be used as the MPI
                         communicator.
        :param shutdown_on_error: If True, any exception raised during data
                                  distributions will be handled internally,
                                  resulting in a clean exit of all MPI ranks.

        """

        self._comm = mpi_comm

        self._rank = self._comm.Get_rank()
        self._num_ranks = self._comm.Get_size()

        self._is_root = self._rank == 0

        self._shutdown_on_error = shutdown_on_error

    def __str__(self):
        string = f'\nRank: {self._rank}'
        string += f'\nNo. of ranks: {self._num_ranks}'
        string += f'\nIs root: {self._is_root}'
        string += '\n'

        return string

    def __repr__(self):
        representation = f'\nObject of class: {self.__class__.__name__}'
        representation += self.__str__()
        representation += f'\nCommunicator: {repr(self._comm)}'
        representation += '\n'

        return representation

    def __call__(self, data_loader, shuffle=False):
        """
        Decorates the given function so that instead of returning all the
        data_items, the given function returns only those data_items that are
        to be processed by the local MPI rank. All the MPI communication
        details are handled internally, so that the caller need not
        implement these.

        :param data_loader: Any function that returns a sized iterable of data_items,
                            i.e., it should be possible to call iter() and len() on
                            the object returned from the function.
                            A function that does not return a sized iterable causes
                            an error. Also, if the number of items in the iterable
                            is less than the No. of MPI ranks, an error is
                            generated.
        :param shuffle: If True, the iterable of all data items is shuffled before
                        distribution amongst MPI ranks. RNG used is from the
                        'random' package in the Python standard library.

        :raises slns.errors import DataDistributionError: can be raised if
                   'shutdown on error' is not requested at the time of object
                   creation.

        :returns: Decorated function. The function when called returns a list
                of data items to be processed by the local MPI rank.

        """

        @functools.wraps(data_loader)
        def wrapper(*args, **kwargs):
            if self._is_root:
                data_items, error = self._load_data(data_loader, *args, **kwargs)
            else:
                data_items, error = None, None

            self._manage_error_status(error)

            self._manage_error_status(
                self._check_if_iterable(data_items)
            )

            self._manage_error_status(
                self._check_if_sized(data_items)
            )

            self._manage_error_status(
                self._check_if_enough_items(data_items)
            )

            if self._is_root:
                if shuffle:
                    data_items = random.sample(population=data_items, k=len(data_items))

                # Data to send
                bcast_data = self._create_rank_to_items_map(data_items)
            else:
                bcast_data = None

            # Broadcast
            received_data = self._comm.bcast(bcast_data, root=0)

            # Data items for the local rank
            items = received_data[self._rank]

            return items

        return wrapper

    @staticmethod
    def _load_data(loader, *args, **kwargs):
        data_items = None

        try:
            data_items = loader(*args, **kwargs)
        except Exception as error:  # pylint: disable=broad-except
            result = {
                Constants.STATUS: Constants.FAILURE,
                Constants.MESSAGE: f'Error during data loading: {str(error)}'
            }
        else:
            result = {
                Constants.STATUS: Constants.SUCCESS,
                Constants.MESSAGE: None
            }

        return data_items, result

    def _check_type(self, obj, check_func, err_msg):
        result = None

        if self._is_root:
            try:
                check_func(obj)
            except TypeError:
                result = {
                    Constants.STATUS: Constants.FAILURE,
                    Constants.MESSAGE: err_msg
                }
            else:
                result = {
                    Constants.STATUS: Constants.SUCCESS,
                    Constants.MESSAGE: None
                }

        return result

    def _check_if_iterable(self, data_items):
        err_msg = f'Value returned by the provided data loader is not an' \
                  f'iterable, i.e., it is not possible to call iter() on it. ' \
                  f'The function to be decorated by {self.__class__.__name__} ' \
                  f'must return an iterable.'

        return self._check_type(data_items, iter, err_msg)

    def _check_if_sized(self, data_items):
        err_msg = f'Value returned by the provided data loader is not a sized' \
                  f'object, i.e., it is not possible to call len() on it. ' \
                  f'The function to be decorated by {self.__class__.__name__} ' \
                  f'must return a sized object.'

        return self._check_type(data_items, len, err_msg)

    def _check_if_enough_items(self, data_items):
        result = None

        if self._is_root:
            # Make sure there are at least as many data items as MPI ranks,
            # so that each MPI rank has at least one item to process.
            if len(data_items) < self._num_ranks:
                err_msg = f'No. of input data items ({len(data_items)}) is less than the ' \
                          f'No. of MPI ranks ({self._num_ranks}). ' \
                          'No. of input data items must be equal to or greater than ' \
                          'the No. of MPI ranks.'

                result = {
                    Constants.STATUS: Constants.FAILURE,
                    Constants.MESSAGE: err_msg
                }
            else:
                result = {
                    Constants.STATUS: Constants.SUCCESS,
                    Constants.MESSAGE: None
                }

        return result

    def _manage_error_status(self, result):
        # Broadcast the result
        result = self._comm.bcast(result, root=0)

        # If result indicates failure,
        if result[Constants.STATUS] != Constants.SUCCESS:
            error = DataDistributionError(result[Constants.MESSAGE])

            if self._shutdown_on_error:
                ErrorHandler.exit_all_ranks(error, self._comm)
            else:
                raise error

    def _create_rank_to_items_map(self, data_items):
        """
        Creates and returns a dictionary where each key is one of the
        MPI rank IDs being used, and the value is a list of data items
        to be processed by the corresponding rank.

        The data items are assigned to ranks in a round-robin fashion.
        This ensures that no two ranks will eventually differ in the
        No. of assigned items by more than one.

        :param data_items: Iterable of all items to be partitioned
                          amongst the ranks.

        :returns: Dictionary of the form {rank id: [assigned items]}

        """

        # Initialize the dict {rank: [assigned items]}, with
        # ranks as keys and empty lists as values.
        rank_to_items = {rank: [] for rank in range(self._num_ranks)}

        # Cyclic iterator over rank IDs
        rank_it = itertools.cycle(range(self._num_ranks))

        # For each item
        for item in data_items:
            # Get the next available rank from the iterator
            rank = next(rank_it)

            # Append the item to the list of items corresponding
            # to 'rank'
            rank_to_items[rank].append(item)

        return rank_to_items
