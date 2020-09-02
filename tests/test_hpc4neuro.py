"""
    Tests for module-wide parameters.

"""

from hpc4neuro import __version__


def test_version():
    """
    Tests the hpc4neuro module version.

    """

    assert __version__ == '0.1.0'
