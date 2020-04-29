"""
Module for handling data.

This module contains a specification for the dataset as well as utilities to
handle the data.

quantify datasets are based on the xarray Dataset.

Utility functions include
- Finding a dataset
-
"""
import os
import numpy as np
import xarray as xr
from datetime import datetime
from uuid import uuid4


def get_datadir():
    """Returns the current data directory."""

    # TODO: make configurable, currently hardcoded to
    # TODO: make configurable.
    # quantify/data

    datadir = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'data'))

    return datadir


def create_exp_folder(tuid,  name='', datadir=None):
    """
    Creates an empty folder to store an experiment container.

    Args:
        tuid (str) : a timestamp based human-readable unique identifier.
        name (str) : optional name to identify the folder
        datadir (str):
            path of the data directory. If `None`, uses `get_datadir()` to
            determine the data directory.

    Returns:
        exp_folder (str):
            the full path of the experiment folder following the following
            convention: /datadir/YYMMDD/HHMMSS-******-name/
    """
    if datadir is None:
        datadir = get_datadir()
    exp_folder = os.path.join(datadir, tuid[:8], tuid[9:])
    if name != '':
        exp_folder += '-'+name

    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder


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
        tuid (str): timestamp based uid formatted as YYMMDD-HHMMSS-******
    """
    ts = datetime.now()
    tuid = ts.strftime('%Y%m%d-%H%M%S-')+str(uuid4())[:6]

    return tuid


def append_to_Xarr():
    pass


def initialize_dataset(setable_pars, setpoints, getable_pars):
    """
    Initialize an empty dataset based on
        setable_pars, setpoints and getable_pars

    Args:
        setable_pars (list):    a list of M setables
        setpoints (np.array):   an (N*M) array
        getable_pars (list):    a list of getables

    Returns:
        Dataset

    """
    darrs = []
    for i, setpar in enumerate(setable_pars):
        darrs.append(xr.DataArray(
            data=setpoints[:, i],
            name='x{}'.format(i),
            attrs={'name': setpar.name, 'long_name': setpar.label,
                   'unit': setpar.unit}))

    numpoints = len(setpoints[:, 0])
    for j, getpar in enumerate(getable_pars):
        empty_arr = np.empty(numpoints)
        empty_arr[:] = np.nan
        darrs.append(xr.DataArray(
            data=empty_arr,
            name='y{}'.format(i),
            attrs={'name': getpar.name, 'long_name': getpar.label,
                   'unit': getpar.unit}))

    dataset = xr.merge(darrs)
    dataset.attrs['tuid'] = gen_tuid()
    return dataset
