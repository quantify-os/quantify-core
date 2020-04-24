"""
Module for handling data.

This module contains a specification for the dataset as well as utilities to
handle the data.

Utility functions include
- Finding a dataset
-
"""
import xarray as xr


def is_valid_dset(dset):
    """
    Asserts if dset adheres to quantify Dataset specification.

    Args:
        dset: an xarray dset object

    Returns:
        is_valid (bool)
    """
    if not isinstance(dset, xr.Dataset):
        raise TypeError


    return True

def gen_time_uid():
    """
    Generates a human readable unique identifier based on the current time.
    """

    return