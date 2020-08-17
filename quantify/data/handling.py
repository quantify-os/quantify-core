"""
-----------------------------------------------------------------------------
Description:    Utilities for handling data.
Repository:     https://gitlab.com/qblox/packages/software/quantify/
Copyright (C) Qblox BV (2020)
-----------------------------------------------------------------------------
"""
import os
import sys
import json
from datetime import datetime
from uuid import uuid4
import numpy as np
import xarray as xr
from qcodes import Instrument
from quantify.data.types import TUID
from quantify.utilities.general import delete_keys_from_dict


# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

_default_datadir = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'data'))

this._datadir = None


def gen_tuid(ts=None):
    """
    Generates a :class:`~quantify.data.types.TUID` based on current time.

    Args:
        ts (:class:`datetime.datetime`) : optional, can be passed to ensure the tuid is based on a specific time.

    Returns:
        :class:`~quantify.data.types.TUID`: timestamp based uid.
    """
    if ts is None:
        ts = datetime.now()
    # ts gives microsecs by default
    (dt, micro) = ts.strftime('%Y%m%d-%H%M%S-.%f').split('.')
    # this ensures the string is formatted correctly as some systems return 0 for micro
    dt = "%s%03d-" % (dt, int(micro) / 1000)
    # the tuid is composed of the timestamp and a 6 character uuid.
    tuid = TUID(dt + str(uuid4())[:6])

    return tuid


def get_datadir():
    """
    Returns the current data directory.

    The data directory can be changed using :func:`~quantify.data.handling.set_datadir`
    """

    if this._datadir is None:
        this._datadir = this._default_datadir

    return this._datadir


def set_datadir(datadir):
    """
    Sets the data directory.

    Args:
        datadir (str):
            path of the data directory. If set to None, resets the datadir to the default datadir (quantify/data).
    """
    this._datadir = datadir


def _locate_experiment_file(tuid: TUID, datadir: str, name: str) -> str:
    if datadir is None:
        datadir = get_datadir()

    daydir = os.path.join(datadir, tuid[:8])

    # This will raise a file not found error if no data exists on the specified date
    exp_folders = list(filter(lambda x: tuid[9:] in x, os.listdir(daydir)))
    if len(exp_folders) == 0:
        print(os.listdir(daydir))
        raise FileNotFoundError("File with tuid: {} was not found.".format(tuid))

    # We assume that the length is 1 as tuid is assumed to be unique
    exp_folder = exp_folders[0]

    return os.path.join(daydir, exp_folder, name)


def load_dataset(tuid: TUID, datadir: str = None) -> xr.Dataset:
    """
    Loads a dataset specified by a tuid.

    Args:
        tuid (str): a :class:`~quantify.data.types.TUID` string.
            It is also possible to specify only the first part of a tuid.

        datadir (str): path of the data directory. If `None`, uses `get_datadir()` to determine the data directory.

    Returns:
        :class:`xarray.Dataset`: The dataset.

    Raises:
        FileNotFoundError: No data found for specified date.

    .. tip::

        This method also works when specifying only the first part of a :class:`~quantify.data.types.TUID`.

    .. note::

        This method uses :func:`xarray.load_dataset` to ensure the file is closed after loading as datasets are
        intended to be immutable after performing the initial experiment.
    """
    return xr.load_dataset(_locate_experiment_file(tuid, datadir, 'dataset.hdf5'))


def load_snapshot(tuid: TUID, datadir: str = None, file: str = 'snapshot.json') -> dict:
    with open(_locate_experiment_file(tuid, datadir, file)) as snap:
        return json.load(snap)


def create_exp_folder(tuid, name='', datadir=None):
    """
    Creates an empty folder to store an experiment container.

    If the folder already exists, simple return the experiment folder corresponding to the
    :class:`~quantify.data.types.TUID`.

    Args:
        tuid (:class:`~quantify.data.types.TUID`) : a timestamp based human-readable unique identifier.
        name (str) : optional name to identify the folder
        datadir (str) : path of the data directory. If `None`, uses `get_datadir()` to determine the data directory.

    Returns:
        str: full path of the experiment folder following format: /datadir/YYMMDD/HHMMSS-******-name/.
    """
    assert TUID.is_valid(tuid)

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
        dset (:class:`xarray.Dataset`): the dataset

    Returns:
        bool

    Raises:
        TypeError: the dataset is not of type :class:`xarray.Dataset`
    """
    if not isinstance(dset, xr.Dataset):
        raise TypeError
    assert TUID.is_valid(dset.attrs['tuid'])

    return True


def initialize_dataset(setable_pars, setpoints, getable_pars):
    """
    Initialize an empty dataset based on setable_pars, setpoints and getable_pars

    Args:
        setable_pars (list):                    a list of M settables
        setpoints (:class:`numpy.ndarray`):     an (N*M) array
        getable_pars (list):                    a list of gettables

    Returns:
        :class:`xarray.Dataset`: the dataset

    """
    darrs = []
    for i, setpar in enumerate(setable_pars):
        darrs.append(xr.DataArray(
            data=setpoints[:, i],
            name='x{}'.format(i),
            attrs={'name': setpar.name, 'long_name': setpar.label, 'unit': setpar.unit})
        )

    numpoints = len(setpoints[:, 0])
    j = 0
    for getpar in getable_pars:
        #  it's possible for one Gettable to return multiple axes. to handle this, zip the axis info together
        #  so we can iterate through when defining the axis in the dataset
        if not isinstance(getpar.name, list):
            itrbl = zip([getpar.name], [getpar.label], [getpar.unit])
        else:
            itrbl = zip(getpar.name, getpar.label, getpar.unit)

        count = 0
        for idx, info in enumerate(itrbl):
            empty_arr = np.empty(numpoints)
            empty_arr[:] = np.nan
            darrs.append(xr.DataArray(
                data=empty_arr,
                name='y{}'.format(j + idx),
                attrs={'name': info[0], 'long_name': info[1], 'unit': info[2]})
            )
            count += 1
        j += count

    dataset = xr.merge(darrs)
    dataset.attrs['tuid'] = gen_tuid()
    return dataset


def grow_dataset(dataset):
    """
    Resizes the dataset by doubling the current length of all arrays.

    Args:
        dataset (:class:`xarray.Dataset`): the dataset to resize.

    Returns:
        The resized dataset.
    """
    darrs = []
    for col in dataset:
        data = dataset[col].values
        darrs.append(xr.DataArray(
            name=dataset[col].name,
            data=np.resize(data, len(data) * 2),
            attrs=dataset[col].attrs
        ))
    dataset = dataset.drop_dims(['dim_0'])
    new_data = xr.merge(darrs)
    return dataset.merge(new_data)


def trim_dataset(dataset):
    """
    Trim NaNs from a dataset, useful in the case of a dynamically resized dataset (eg. adaptive loops).

    Args:
        dataset (:class:`xarray.Dataset`): the dataset to trim.

    Returns:
        The dataset, trimmed and resized if necessary or unchanged.
    """
    for i, val in enumerate(reversed(dataset['y0'].values)):
        if not np.isnan(val):
            finish_idx = len(dataset['y0'].values) - i
            darrs = []
            for col in dataset:
                data = dataset[col].values[:finish_idx]
                darrs.append(xr.DataArray(
                    name=dataset[col].name,
                    data=data,
                    attrs=dataset[col].attrs
                ))
            dataset = dataset.drop_dims(['dim_0'])
            new_data = xr.merge(darrs)
            return dataset.merge(new_data)
    return dataset


########################################################################


def get_latest_tuid(contains='') -> TUID:
    """
    Returns the most recent tuid.

    Args:
        contains (str): an optional string contained in the experiment name.

    Returns:
        :class:`~quantify.data.types.TUID`: the latest TUID

    Raises:
        FileNotFoundError: No data found

    .. tip::

        This function is similar to :func:`~get_tuids_containing` but is preferred if one is only interested in the
        most recent :class:`~quantify.data.types.TUID` for performance reasons.

    """
    return get_tuids_containing(contains, 1)[0]


def get_tuids_containing(contains, max_results=sys.maxsize) -> list:
    """
    Returns a list of tuids containing a specific label

    Args:
        contains (str): a string contained in the experiment name.
        max_results (int): maximum number of results to return. Defaults to unlimited.

    Returns:
        A list of  :class:`~quantify.data.types.TUID`: objects

    Raises:
        FileNotFoundError: No data found

    .. tip::

        If one is only interested in the most recent :class:`~quantify.data.types.TUID`,
        :func:`~get_latest_tuid` is preferred for performance reasons.
    """
    datadir = get_datadir()
    daydirs = list(filter(lambda x: (x.isdigit() and len(x) == 8), os.listdir(datadir)))
    daydirs.sort(reverse=True)
    if len(daydirs) == 0:
        raise FileNotFoundError('There are no valid day directories in the data folder "{}".'.format(datadir))
    tuids = []
    for dd in daydirs:
        expdirs = list(filter(lambda x: (x[:6].isdigit() and x[6] == '-' and len(x) > 12 and contains in x),
                              os.listdir(os.path.join(datadir, dd))))
        expdirs.sort(reverse=True)
        for expname in expdirs:
            tuids.append(TUID('{}-{}'.format(dd, expname[:17])))
            if len(tuids) == max_results:
                return tuids
    if len(tuids) == 0:
        raise FileNotFoundError('No experiment found containing "{}"'.format(contains))
    return tuids


def snapshot(update: bool = False, clean: bool = True) -> dict:
    """
    State of all instruments setup as a JSON-compatible dictionary (everything that the custom JSON encoder class
    :class:`qcodes.utils.helpers.NumpyJSONEncoder` supports).

    Args:
        update (bool) : if True, first gets all values before filling the snapshot.
        clean (bool)  : if True, removes certain keys from the snapshot to create a more readible and compact snapshot.

    """

    snap = {
        'instruments': {},
        'parameters': {},
    }
    for ins_name, ins_ref in Instrument._all_instruments.items():
        snap['instruments'][ins_name] = ins_ref().snapshot(update=update)

    if clean:
        exclude_keys = {
            "inter_delay",
            "post_delay",
            "vals",
            "instrument",
            "submodules",
            "functions",
            "__class__",
            "raw_value",
            "instrument_name",
            "full_name",
            "val_mapping",
        }
        snap = delete_keys_from_dict(snap, exclude_keys)

    return snap
