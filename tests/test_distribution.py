
import os
import functools

import pytest
from mpi4py import MPI

from slns.distribution import DataDistributor
from slns.errors import DataDistributionError

# TODO: We can start by testing exception handling.
#       1. Exception thrown by data loader
#           i. Define the data loader
#           ii. Pass path to a non-existent directory
#           iii. Check exception is raised with shutdown_on_error=False
#       2. Check iterable
#       3. Check sized
#       4. Check No. of items
#       How can one test proper execution with shutdown_on_error=True ??


class IterableNotSized:
    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)


class SizedNotIterable:
    def __init__(self, iterable):
        self._iterable = iterable

    def __len__(self):
        return len(self._iterable)


def make_not_sized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sized_iterable = func(*args, **kwargs)

        iterable = IterableNotSized(sized_iterable)

        return iterable

    return wrapper


def make_not_iterable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sized_iterable = func(*args, **kwargs)

        sized = SizedNotIterable(sized_iterable)

        return sized

    return wrapper


def generate_filenames(num_files):
    return [f'file_{i}.txt' for i in range(num_files)]


class TestExceptionHandling:
    def test_data_loader_error(self):
        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(os.listdir)

        with pytest.raises(DataDistributionError):
            get_rank_local_filenames('./this_is_not_a_dir')

    def test_non_iterable_error(self, tmpdir):
        # Create file in a temporary directory
        filenames = ['file_1.txt', 'file_2.txt', 'file_3.txt', 'file_4.txt']
        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        callable_ = make_not_sized(os.listdir)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(callable_)

        with pytest.raises(DataDistributionError):
            get_rank_local_filenames(tmpdir)

    def test_non_sized_error(self, tmpdir):
        # Create file in a temporary directory
        filenames = ['file_1.txt', 'file_2.txt', 'file_3.txt', 'file_4.txt']
        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        callable_ = make_not_iterable(os.listdir)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(callable_)

        with pytest.raises(DataDistributionError):
            get_rank_local_filenames(tmpdir)

    def test_size_mismatch_error(self, tmpdir):
        # Create file in a temporary directory
        num_ranks = MPI.COMM_WORLD.Get_size()
        filenames = generate_filenames(num_ranks - 1)

        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(os.listdir)

        with pytest.raises(DataDistributionError):
            get_rank_local_filenames(tmpdir)
