
import os
import tempfile
import functools
import itertools

import pytest
from mpi4py import MPI

from slns.distribution import DataDistributor
from slns.errors import DataDistributionError


rank = MPI.COMM_WORLD.Get_rank()
num_ranks = MPI.COMM_WORLD.Get_size()


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


def change_result_type(func, custom_type):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        result_with_new_type = custom_type(result)

        return result_with_new_type

    return wrapper


def generate_filenames(num_files):
    return [f'file_{i}.txt' for i in range(num_files)]


def generate_files(target_dir, num_files):
    if rank == 0:
        filenames = generate_filenames(num_files)

        for filename in filenames:
            with open(os.path.join(target_dir, filename), 'w') as f:
                f.write(filename)


def get_decorated_func(func, shutdown_on_error):
    decorator = DataDistributor(
        MPI.COMM_WORLD, shutdown_on_error=shutdown_on_error)

    return decorator(func)


class TestExceptionHandling:
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


@pytest.fixture(scope='class')
def distributed_filenames():
    # Create a temporary data directory
    with tempfile.TemporaryDirectory(prefix='slns_test_') as data_dir:
        # Generate files in a temporary directory
        generate_files(data_dir, num_files=num_ranks)

        # Decorate the data loader with distribution functionality
        decorated_func = get_decorated_func(os.listdir, shutdown_on_error=False)

        # Call the decorated function to get the list of rank-local
        # filenames
        rank_local_filenames = decorated_func(data_dir)

        # Collect a list of all lists of filenames from all ranks
        gathered_filename_lists = MPI.COMM_WORLD.allgather(rank_local_filenames)

        yield gathered_filename_lists, data_dir


# TODO: Test distribution of numpy arrays ...
class TestFilenamesDistribution:
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

        # data_loader = make_sized_iterable(os.listdir)
        data_loader = change_result_type(os.listdir, SizedIterable)

        dist_decorator = DataDistributor(MPI.COMM_WORLD, shutdown_on_error=False)
        get_rank_local_filenames = dist_decorator(data_loader)

        updated_filenames = get_rank_local_filenames(tmpdir)

        assert len(updated_filenames) >= 1

    def test_all_ranks_get_items(self, distributed_filenames):
        gathered_filename_lists, _ = distributed_filenames

        # The No. of lists returned should equal the No. of ranks
        assert len(gathered_filename_lists) == num_ranks

        # Every sublist must contain at least one element
        assert all(filenames for filenames in gathered_filename_lists)

    def test_ranks_get_disjoint_subsets(self, distributed_filenames):
        gathered_filename_lists, _ = distributed_filenames

        # Fuse the received list of lists into one list
        all_filenames = list(itertools.chain(*gathered_filename_lists))

        # Verify that all items in the list are unique
        assert len(all_filenames) == len(set(all_filenames))

    def test_all_items_are_distributed(self, distributed_filenames):
        gathered_filename_lists, data_dir = distributed_filenames

        # Fuse the received list of lists into one list
        all_filenames = list(itertools.chain(*gathered_filename_lists))

        if rank == 0:
            # Use the un-decorated loader to get a list of filenames
            read_filenames = os.listdir(data_dir)

            # Verify that contents are the same
            assert not (set(all_filenames) ^ set(read_filenames))
