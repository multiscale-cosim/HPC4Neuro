
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


class SizedIterable:
    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def __len__(self):
        return len(self._iterable)


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


def make_sized_iterable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        sized_iterable = SizedIterable(result)

        return sized_iterable

    return wrapper


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
        num_ranks = MPI.COMM_WORLD.Get_size()

        # Create file in a temporary directory
        filenames = generate_filenames(num_files=num_ranks)
        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        callable_ = make_not_sized(os.listdir)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(callable_)

        with pytest.raises(DataDistributionError):
            get_rank_local_filenames(tmpdir)

    def test_non_sized_error(self, tmpdir):
        num_ranks = MPI.COMM_WORLD.Get_size()

        # Create file in a temporary directory
        filenames = generate_filenames(num_files=num_ranks)
        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        callable_ = make_not_iterable(os.listdir)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(callable_)

        with pytest.raises(DataDistributionError):
            get_rank_local_filenames(tmpdir)

    def test_size_mismatch_error(self, tmpdir):
        num_ranks = MPI.COMM_WORLD.Get_size()
        filenames = generate_filenames(num_ranks - 1)

        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(os.listdir)

        with pytest.raises(DataDistributionError):
            get_rank_local_filenames(tmpdir)


class TestDistribution:
    def test_distribution_happened(self, tmpdir):
        rank = MPI.COMM_WORLD.Get_rank()
        num_ranks = MPI.COMM_WORLD.Get_size()

        filenames = generate_filenames(num_files=num_ranks)

        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(os.listdir)

        updated_filenames = get_rank_local_filenames(tmpdir)

        assert len(updated_filenames) >= 1

    def test_custom_iterable(self, tmpdir):
        num_ranks = MPI.COMM_WORLD.Get_size()

        filenames = generate_filenames(num_files=num_ranks)

        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        callable_ = make_sized_iterable(os.listdir)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(callable_)

        updated_filenames = get_rank_local_filenames(tmpdir)

        assert len(updated_filenames) >= 1

    def test_distributed_items_are_disjoint(self, tmpdir):
        # rank = MPI.COMM_WORLD.Get_rank()
        num_ranks = MPI.COMM_WORLD.Get_size()

        filenames = generate_filenames(num_files=num_ranks)

        for filename in filenames:
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(filename)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(os.listdir)

        updated_filenames = get_rank_local_filenames(tmpdir)

        gathered_filenames = MPI.COMM_WORLD.allgather(updated_filenames)

        assert len(gathered_filenames) == num_ranks

        count = 0
        for f_names in gathered_filenames:
            count += len(f_names)

        unique_filenames = set()
        for f_names in gathered_filenames:
            unique_filenames.add(*f_names)

        assert count == len(unique_filenames)
