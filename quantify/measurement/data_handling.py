"""
Module for handling data.

This module contains a specification for the dataset as well as utilities to
handle the data.

Utility functions include
- Finding a dataset
-
"""

import xarray as xr
from datetime import datetime
from uuid import uuid4


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


def gen_tuid(ts=None):
    """
    Generates a human readable unique identifier based on the current time.

    Args:
        ts (datetime) : optional datetime object can be passed to ensure the
            tuid is based on a specific timestamp.

    Returns:
        tuid (str): timestamp based uid formatted as YYMMDD-HHMMSS-****
    """
    ts = datetime.now()
    tuid = ts.strftime('%Y%m%d-%H%M%S-')+str(uuid4())[:4]

    return tuid

