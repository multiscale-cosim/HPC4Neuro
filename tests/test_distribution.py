# Copyright (c) 2019 Forschungszentrum Juelich GmbH
# This code is licensed under MIT license (see the LICENSE file for details)

"""
    Test suite to test the slns.distribution module.

"""

import os
import tempfile
import functools
import itertools

import pytest
import numpy as np
from mpi4py import MPI

from slns.distribution import DataDistributor
from slns.errors import DataDistributionError


# MPI communicator info required by multiple test classes
rank = MPI.COMM_WORLD.Get_rank()
num_ranks = MPI.COMM_WORLD.Get_size()


class SizedIterable:
    """
    A type that is both iterable and sized.

    """

    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def __len__(self):
        return len(self._iterable)


class IterableNotSized:
    """
    An iterable type that is not sized.

    """

    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)


class SizedNotIterable:
    """
    A sized type that is not iterable.

    """

    def __init__(self, iterable):
        self._iterable = iterable

    def __len__(self):
        return len(self._iterable)


def change_result_type(func, custom_type):
    """
    Decorator that converts the type of the object returned by
    the function to be decorated, to the given type.

    :param func: Function to be decorated.
    :param custom_type: Type of the object to be returned by the
                decorated function.

    :return: Decorated function.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        result_with_new_type = custom_type(result)

        return result_with_new_type

    return wrapper


def generate_files(target_dir, num_files):
    """
    Generated the given number of files in the given directory.

    :param target_dir: Directory in which to create files.
    :param num_files: No. of the files to be created.

    """

    if rank == 0:
        filenames = [f'file_{i}.txt' for i in range(num_files)]

        for filename in filenames:
            with open(os.path.join(target_dir, filename), 'w') as f:
                f.write(filename)


def get_decorated_func(func, shutdown_on_error):
    """
    Decorate the given function with the DataDistributor and
    return the decorated function.

    :param func: Function to be decorated.
    :param shutdown_on_error: Parameter to be passed to the DataDistributor.

    :return: Decorated function.

    """

    decorator = DataDistributor(
        MPI.COMM_WORLD, shutdown_on_error=shutdown_on_error)

    return decorator(func)


@pytest.fixture(scope='class', params=['list', 'custom', 'numpy'])
def distributed_filenames(request):
    """
    Setup for testing distribution of iterables of filenames.

    :param request: Parameters.

    :return: Tuple. (List of lists of all filenames received by all
                ranks after distribution, name of the temporary
                directory in which files are created)

    """

    # Create a temporary data directory
    with tempfile.TemporaryDirectory(prefix='slns_test_') as data_dir:
        # Generate files in the created directory
        generate_files(data_dir, num_files=num_ranks)

        # Use the iterable's data type as per the received parameter
        if request.param == 'list':
            data_loader = os.listdir
        elif request.param == 'custom':
            data_loader = change_result_type(os.listdir, SizedIterable)
        elif request.param == 'numpy':
            data_loader = change_result_type(os.listdir, np.char.array)

        # Decorate the data loader with distribution functionality
        decorated_func = get_decorated_func(data_loader, shutdown_on_error=False)

        # Call the decorated function to get the list of rank-local
        # filenames
        rank_local_filenames = decorated_func(data_dir)

        # Collect a list of all lists of filenames from all ranks
        gathered_filename_lists = MPI.COMM_WORLD.allgather(rank_local_filenames)

        yield gathered_filename_lists, data_dir


class TestExceptionHandling:
    """
    Tests to verify proper raising of exceptions in error scenarios.

    """

    def test_error_from_data_loader(self):
        """
        An error thrown by the decorated data loader function should
        be raised as slns.errors.DataDistributionError when
        automatic shutdown is not requested.
        """

        # Decorate the data loader with distribution functionality
        decorated_func = get_decorated_func(func=os.listdir, shutdown_on_error=False)

        # Distribution error should be thrown because the loader
        # function should generated an error when trying to list
        # a non-existent directory.
        with pytest.raises(DataDistributionError):
            decorated_func('./non-existent-dir')

    def test_error_for_non_iterable(self, tmpdir):
        """
        When the function to be decorated returns an object that is
        not iterable, the slns.errors.DataDistributionError should
        be raised (when automatic shutdown is not requested).

        :param tmpdir: Temporary directory created by Pytest fixture.
        """

        # Generate files in a temporary directory
        generate_files(tmpdir, num_files=num_ranks)

        # Create a data loader function that is iterable but not sized
        data_loader = change_result_type(os.listdir, SizedNotIterable)

        # Decorate the data loader with distribution functionality
        decorated_func = get_decorated_func(data_loader, shutdown_on_error=False)

        with pytest.raises(DataDistributionError):
            decorated_func(tmpdir)

    def test_error_for_non_sized(self, tmpdir):
        """
        When the function to be decorated returns an object that is
        not sized, the slns.errors.DataDistributionError should
        be raised (when automatic shutdown is not requested).

        :param tmpdir: Temporary directory created by Pytest fixture.
        """

        # Generate files in a temporary directory
        generate_files(tmpdir, num_files=num_ranks)

        # Create a data loader function that is iterable but not sized
        data_loader = change_result_type(os.listdir, IterableNotSized)

        # Decorate the data loader with distribution functionality
        decorated_func = get_decorated_func(data_loader, shutdown_on_error=False)

        with pytest.raises(DataDistributionError):
            decorated_func(tmpdir)

    def test_error_for_size_mismatch(self, tmpdir):
        """
        If the number of items to be distributed returned by the
        decorated function is less the number of MPI ranks, the
        slns.errors.DataDistributionError should be raised (when
        automatic shutdown is not requested).

        :param tmpdir: Temporary directory created by Pytest fixture.
        """

        # Generate files in a temporary directory
        generate_files(tmpdir, num_files=num_ranks - 1)

        # Decorate the data loader with distribution functionality
        decorated_func = get_decorated_func(os.listdir, shutdown_on_error=False)

        with pytest.raises(DataDistributionError):
            decorated_func(tmpdir)


class TestFilenamesDistribution:
    """
    Tests to verify correct data distribution with different
    types of sized iterables.

    The items being distributed are names of files created in
    a temporary directory.

    """

    def test_all_ranks_get_items(self, distributed_filenames):
        """
        Verify that each rank receives at least one item as a
        result of distribution.

        :param distributed_filenames: Tuple generated by the
                    corresponding fixture.

        """

        gathered_filename_lists, _ = distributed_filenames

        # The No. of lists returned should equal the No. of ranks
        assert len(gathered_filename_lists) == num_ranks

        # Every sublist must contain at least one element
        assert all(filenames for filenames in gathered_filename_lists)

    def test_ranks_get_disjoint_subsets(self, distributed_filenames):
        """
        Verify that ranks receive disjoint subsets of items, i.e., no
        two ranks receive the same item.

        :param distributed_filenames: Tuple generated by the
                    corresponding fixture.

        """

        gathered_filename_lists, _ = distributed_filenames

        # Fuse the received list of lists into one list
        all_filenames = list(itertools.chain(*gathered_filename_lists))

        # Verify that all items in the list are unique
        assert len(all_filenames) == len(set(all_filenames))

    def test_all_items_are_distributed(self, distributed_filenames):
        """
        Verify that items from the iterable are all distributed, i.e.,
        no item is left out.

        :param distributed_filenames: Tuple generated by the
                    corresponding fixture.

        """

        gathered_filename_lists, data_dir = distributed_filenames

        # Fuse the received list of lists into one list
        all_filenames = list(itertools.chain(*gathered_filename_lists))

        if rank == 0:
            # Use the un-decorated loader to get a list of filenames
            read_filenames = os.listdir(data_dir)

            # Verify that contents are the same
            assert not (set(all_filenames) ^ set(read_filenames))
