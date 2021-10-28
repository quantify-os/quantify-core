# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities for handling data."""
from __future__ import annotations

import datetime
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Union
from uuid import uuid4

import numpy as np
import xarray as xr
from dateutil.parser import parse
from qcodes import Instrument

import quantify_core.data.dataset_adapters as da
from quantify_core.data.types import TUID
from quantify_core.utilities.general import delete_keys_from_dict

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
this._datadir = None

DATASET_NAME = "dataset.hdf5"
QUANTITIES_OF_INTEREST_NAME = "quantities_of_interest.json"
PROCESSED_DATASET_NAME = "dataset_processed.hdf5"


def gen_tuid(time_stamp: datetime.datetime = None) -> TUID:
    """
    Generates a :class:`~quantify_core.data.types.TUID` based on current time.

    Parameters
    ----------
    time_stamp
        Optional, can be passed to ensure the tuid is based on a specific time.

    Returns
    -------
    :
        Timestamp based uid.
    """
    if time_stamp is None:
        time_stamp = datetime.datetime.now()
    # time_stamp gives microsecs by default
    (date_time, micro) = time_stamp.strftime("%Y%m%d-%H%M%S-.%f").split(".")
    # this ensures the string is formatted correctly as some systems return 0 for micro
    date_time = f"{date_time}{int(int(micro) / 1000):03d}-"
    # the tuid is composed of the timestamp and a 6 character uuid.
    tuid = TUID(date_time + str(uuid4())[:6])

    return tuid


def get_datadir() -> str:
    """
    Returns the current data directory.
    The data directory can be changed using
    :func:`~quantify_core.data.handling.set_datadir`.

    Returns
    -------
    :
        The current data directory.
    """
    set_datadir_import = "from " + this.__name__ + " import set_datadir"

    if this._datadir is None or not os.path.isdir(this._datadir):
        raise NotADirectoryError(
            "The datadir is not valid. Please set the datadir after importing Quantify."
            "\nWe recommend to settle for a single common data directory for all \n"
            "notebooks/experiments within your measurement setup/PC.\n"
            "E.g. '~/quantify-data' (unix), or 'D:\\Data\\quantify-data' (Windows).\n"
            "The datadir can be changed as follows:\n\n"
            f"    {set_datadir_import}\n"
            "    set_datadir('path_to_datadir')"
        )

    return this._datadir


def set_datadir(datadir: str) -> None:
    """
    Sets the data directory.

    Parameters
    ----------
    datadir
        Path of the data directory. If set to ``None``, resets the datadir to the
        default datadir (``<top_level>/data``).
    """
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    this._datadir = datadir


def locate_experiment_container(tuid: TUID, datadir: str = None) -> str:
    """
    Returns the path to the experiment container of the specified tuid.

    Parameters
    ----------
    tuid
        A :class:`~quantify_core.data.types.TUID` string. It is also possible to specify
        only the first part of a tuid.
    datadir
        Path of the data directory. If ``None``, uses :meth:`~get_datadir` to determine
        the data directory.
    Returns
    -------
    :
        The path to the experiment container
    Raises
    ------
    FileNotFoundError
        Experiment container not found.
    """
    if datadir is None:
        datadir = get_datadir()

    daydir = os.path.join(datadir, tuid[:8])

    # This will raise a file not found error if no data exists on the specified date
    exp_folders = list(filter(lambda x: tuid in x, os.listdir(daydir)))
    if len(exp_folders) == 0:
        raise FileNotFoundError(f"File with tuid: {tuid} was not found.")

    # We assume that the length is 1 as tuid is assumed to be unique
    exp_folder = exp_folders[0]

    return os.path.join(daydir, exp_folder)


def _locate_experiment_file(
    tuid: TUID, datadir: str = None, name: str = DATASET_NAME
) -> str:
    exp_container = locate_experiment_container(tuid=tuid, datadir=datadir)
    return os.path.join(exp_container, name)


def load_dataset(
    tuid: TUID, datadir: str = None, name: str = DATASET_NAME
) -> xr.Dataset:
    """
    Loads a dataset specified by a tuid.

    .. tip::

        This method also works when specifying only the first part of a
        :class:`~quantify_core.data.types.TUID`.

    .. note::

        This method uses :func:`~.load_dataset` to ensure the file is closed after
        loading as datasets are intended to be immutable after performing the initial
        experiment.

    Parameters
    ----------
    tuid
        A :class:`~quantify_core.data.types.TUID` string. It is also possible to specify
        only the first part of a tuid.
    datadir
        Path of the data directory. If ``None``, uses :meth:`~get_datadir` to determine
        the data directory.
    Returns
    -------
    :
        The dataset.
    Raises
    ------
    FileNotFoundError
        No data found for specified date.
    """
    return load_dataset_from_path(_locate_experiment_file(tuid, datadir, name))


def load_dataset_from_path(path: Union[Path, str]) -> xr.Dataset:
    """
    Loads a :class:`~xarray.Dataset` with a specific engine preference.

    Before returning the dataset
    :meth:`AdapterH5NetCDF.recover() <quantify_core.data.dataset_adapters.AdapterH5NetCDF.recover>`
    is applied.

    This function tries to load the dataset until success with the following engine
    preference:

    - ``"h5netcdf"``
    - ``"netcdf4"``
    - No engine specified (:func:`~xarray.load_dataset` default)

    Parameters
    ----------
    path
        Path to the dataset.

    Returns
    -------
    :
        The loaded dataset.
    """  # pylint: disable=line-too-long
    exceptions = []
    engines = ["h5netcdf", "netcdf4", None]
    for engine in engines:
        try:
            dataset = xr.load_dataset(path, engine=engine)
        except Exception as exception:
            exceptions.append(exception)
        else:
            # Only quantify_dataset_version=>2.0.0 requires the adapter
            if "quantify_dataset_version" in dataset.attrs:
                dataset = da.AdapterH5NetCDF.recover(dataset)
            return dataset

    # Do not let exceptions pass silently
    for exception, engine in zip(exceptions, engines[: engines.index(engine)]):
        print(
            f"Failed loading dataset with '{engine}' engine. "
            f"Raised '{exception.__class__.__name__}':\n    {exception}"
        )
    # raise the last exception
    raise exception


def load_quantities_of_interest(tuid: TUID, analysis_name: str) -> dict:
    """
    Given an experiment TUID and the name of an analysis previously run on it,
    retrieves the corresponding "quantities of interest" data.

    Parameters
    ----------
    tuid
        TUID of the experiment.
    analysis_name
        Name of the Analysis from which to load the data.

    Returns
    -------
    :
        A dictionary containing the loaded quantities of interest.
    """

    # Get Analysis directory from TUID
    exp_folder = Path(locate_experiment_container(tuid, get_datadir()))
    analysis_dir = exp_folder / f"analysis_{analysis_name}"

    if not analysis_dir.is_dir():
        raise FileNotFoundError("Analysis not found in current experiment.")

    # Load JSON file and return
    with open(os.path.join(analysis_dir, QUANTITIES_OF_INTEREST_NAME), "r") as file:
        quantities_of_interest = json.load(file)

    return quantities_of_interest


def load_processed_dataset(tuid: TUID, analysis_name: str) -> xr.Dataset:
    """
    Given an experiment TUID and the name of an analysis previously run on it,
    retrieves the processed dataset resulting from that analysis.

    Parameters
    ----------
    tuid
        TUID of the experiment from which to load the data.
    analysis_name
        Name of the Analysis from which to load the data.

    Returns
    -------
    :
        A dataset containing the results of the analysis.
    """

    # Get Analysis directory from TUID
    exp_folder = Path(locate_experiment_container(tuid, get_datadir()))
    analysis_dir = exp_folder / f"analysis_{analysis_name}"

    if not analysis_dir.is_dir():
        raise FileNotFoundError("Analysis not found in current experiment.")

    # Load dataset and return
    return load_dataset_from_path(analysis_dir / PROCESSED_DATASET_NAME)


def _xarray_numpy_bool_patch(dataset: xr.Dataset) -> None:
    """
    Converts any attribute of :obj:`~numpy.bool_` type to a :obj:`~bool`.

    This is a patch to a bug in xarray 0.17.0.

    .. seealso::

        See issue #161 in quantify-core.
        Our (accepted) pull request https://github.com/pydata/xarray/pull/4986
        Version >0.17.0 will fix the problem but will have breaking changes,
        for now we use this patch.

    Parameters
    ----------
    dataset
        The dataset to be patched in-place.

    """

    def bool_cast_attributes(attrs: dict) -> None:
        for attr_name, attr_val in attrs.items():
            if isinstance(attr_val, np.bool_):
                # cast to bool to avoid xarray 0.17.0 type exception
                # for engine="h5netcdf"
                attrs[attr_name] = bool(attr_val)

    for data_array in dataset.variables.values():
        bool_cast_attributes(data_array.attrs)

    bool_cast_attributes(dataset.attrs)


def write_dataset(path: Union[Path, str], dataset: xr.Dataset) -> None:
    """
    Writes a :class:`~xarray.Dataset` to a file with the `h5netcdf` engine.

    Before writing the
    :meth:`AdapterH5NetCDF.adapt() <quantify_core.data.dataset_adapters.AdapterH5NetCDF.adapt>`
    is applied.

    To accommodate for complex-type numbers and arrays ``invalid_netcdf=True`` is used.

    Parameters
    ----------
    path
        Path to the file including filename and extension
    dataset
        The :class:`~xarray.Dataset` to be written to file.
    """  # pylint: disable=line-too-long
    _xarray_numpy_bool_patch(dataset)  # See issue #161 in quantify-core
    # Only quantify_dataset_version=>2.0.0 requires the adapter
    if "quantify_dataset_version" in dataset.attrs:
        dataset = da.AdapterH5NetCDF.adapt(dataset)
    dataset.to_netcdf(path, engine="h5netcdf", invalid_netcdf=True)


def load_snapshot(tuid: TUID, datadir: str = None, file: str = "snapshot.json") -> dict:
    """
    Loads a snapshot specified by a tuid.

    Parameters
    ----------
    tuid
        A :class:`~quantify_core.data.types.TUID` string. It is also possible to specify
        only the first part of a tuid.
    datadir
        Path of the data directory. If ``None``, uses :meth:`~get_datadir` to determine
        the data directory.
    file
        Filename to load.
    Returns
    -------
    :
        The snapshot.
    Raises
    ------
    FileNotFoundError
        No data found for specified date.
    """
    with open(_locate_experiment_file(tuid, datadir, file)) as snap:
        return json.load(snap)


def create_exp_folder(tuid: TUID, name: str = "", datadir: str = None):
    """
    Creates an empty folder to store an experiment container.

    If the folder already exists, simply returns the experiment folder corresponding to
    the :class:`~quantify_core.data.types.TUID`.

    Parameters
    ----------
    tuid
        A timestamp based human-readable unique identifier.
    name
        Optional name to identify the folder.
    datadir
        path of the data directory.
        If ``None``, uses :meth:`~get_datadir` to determine the data directory.
    Returns
    -------
    :
        Full path of the experiment folder following format:
        ``/datadir/YYYYmmDD/YYYYmmDD-HHMMSS-sss-******-name/``.
    """
    TUID.is_valid(tuid)

    if datadir is None:
        datadir = get_datadir()
    exp_folder = os.path.join(datadir, tuid[:8], tuid)
    if name != "":
        exp_folder += "-" + name

    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder


# pylint: disable=too-many-locals
def initialize_dataset(
    settable_pars: Iterable, setpoints: np.ndarray, gettable_pars: Iterable
):
    """
    Initialize an empty dataset based on settable_pars, setpoints and gettable_pars

    Parameters
    ----------
    settable_pars
        A list of M settables.
    setpoints
        An (N*M) array.
    gettable_pars
        A list of gettables.
    Returns
    -------
    :
        The dataset.
    """

    darrs = []
    coords = []
    for i, setpar in enumerate(settable_pars):
        attrs = {
            "name": setpar.name,
            "long_name": setpar.label,
            "units": setpar.unit,
            "batched": _is_batched(setpar),
        }
        if attrs["batched"] and hasattr(setpar, "batch_size"):
            attrs["batch_size"] = getattr(setpar, "batch_size")
        coords.append(f"x{i}")
        darrs.append(xr.DataArray(data=setpoints[:, i], name=coords[-1], attrs=attrs))

    numpoints = len(setpoints[:, 0])
    j = 0
    for getpar in gettable_pars:
        # it's possible for one Gettable to return multiple axes. to handle this, zip
        # the axis info together
        # so we can iterate through when defining the axis in the dataset
        if not isinstance(getpar.name, list):
            itrbl = zip([getpar.name], [getpar.label], [getpar.unit])
        else:
            itrbl = zip(getpar.name, getpar.label, getpar.unit)

        count = 0
        for idx, info in enumerate(itrbl):
            attrs = {
                "name": info[0],
                "long_name": info[1],
                "units": info[2],
                "batched": _is_batched(getpar),
            }
            if attrs["batched"] and hasattr(getpar, "batch_size"):
                attrs["batch_size"] = getattr(getpar, "batch_size")
            empty_arr = np.empty(numpoints)
            empty_arr[:] = np.nan
            darrs.append(
                xr.DataArray(
                    data=empty_arr,
                    name=f"y{j + idx}",
                    attrs=attrs,
                )
            )
            count += 1
        j += count

    dataset = xr.merge(darrs)
    dataset = dataset.set_coords(coords)
    # xarray>=0.18.0 tries to combine attrs which we do not want at all
    dataset.attrs = {}
    dataset.attrs["tuid"] = gen_tuid()
    return dataset


def grow_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Resizes the dataset by doubling the current length of all arrays.

    Parameters
    ----------
    dataset
        The dataset to resize.
    Returns
    -------
    :
        The resized dataset.
    """
    darrs = []

    # coords will also be grown
    for vname in dataset.variables.keys():
        data = dataset[vname].values
        darrs.append(
            xr.DataArray(
                name=dataset[vname].name,
                data=np.pad(data, (0, len(data)), "constant", constant_values=np.nan),
                attrs=dataset[vname].attrs,
            )
        )
    coords = tuple(dataset.coords.keys())
    dataset = dataset.drop_dims(["dim_0"])
    merged_data_arrays = xr.merge(darrs)
    merged_data_arrays.attrs = {}  # xarray>=0.18.0 tries to merge attrs
    dataset = dataset.merge(merged_data_arrays)
    dataset = dataset.set_coords(coords)
    return dataset


def trim_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Trim NaNs from a dataset, useful in the case of a dynamically
    resized dataset (e.g. adaptive loops).

    Parameters
    ----------
    dataset
        The dataset to trim.
    Returns
    -------
    :
        The dataset, trimmed and resized if necessary or unchanged.
    """
    coords = tuple(dataset.coords.keys())
    for i, val in enumerate(reversed(dataset["y0"].values)):
        if not np.isnan(val):
            finish_idx = len(dataset["y0"].values) - i
            darrs = []
            # coords will also be trimmed
            for vname in dataset.variables.keys():
                data = dataset[vname].values[:finish_idx]
                darrs.append(
                    xr.DataArray(
                        name=dataset[vname].name, data=data, attrs=dataset[vname].attrs
                    )
                )
            dataset = dataset.drop_dims(["dim_0"])
            merged_data_arrays = xr.merge(darrs)
            merged_data_arrays.attrs = {}  # xarray>=0.18.0 tries to merge attrs
            dataset = dataset.merge(merged_data_arrays)
            dataset = dataset.set_coords(coords)
            break

    return dataset


def to_gridded_dataset(
    quantify_dataset: xr.Dataset,
    dimension: str = "dim_0",
    coords_names: Iterable = None,
) -> xr.Dataset:
    """
    Converts a flattened (a.k.a. "stacked") dataset as the one generated by the
    :func:`~initialize_dataset` to a dataset in which the measured values are mapped
    onto a grid in the `xarray` format.

    This will be meaningful only if the data itself corresponds to a gridded
    measurement.

    .. note::

        Each individual :code:`(x0[i], x1[i], x2[i], ...)` setpoint must be unique.

    Conversions applied:

    - The names :code:`"x0", "x1", ...` will correspond to the names of the Dimensions.
    - The unique values for each of the :code:`x0, x1, ...` Variables are converted to
        Coordinates.
    - The :code:`y0, y1, ...` Variables are reshaped into a (multi-)dimensional grid
        and associated to the Coordinates.

    .. seealso:: :meth:`.MeasurementControl.setpoints_grid`

    Parameters
    ----------
    quantify_dataset
        Input dataset in the format generated by the :class:`~initialize_dataset`.
    dimension
        The flattened xarray Dimension.
    coords_names
        Optionally specify explicitly which Variables correspond to orthogonal
        coordinates, e.g. datasets holds values for :code:`("x0", "x1")` but only "x0"
        is independent: :code:`to_gridded_dataset(dset, coords_names=["x0"])`.

    Returns
    -------
    :
        The new dataset.


    .. include:: examples/data.handling.to_gridded_dataset.py.rst.txt
    """
    if dimension not in quantify_dataset.dims:
        dims = tuple(quantify_dataset.dims.keys())
        raise ValueError(f"Dimension {dimension} not in dims {dims}.")

    if coords_names is None:
        # for compatibility with older datasets we use `variables` instead of `coords`
        coords_names = sorted(
            v for v in quantify_dataset.variables.keys() if v.startswith("x")
        )
    else:
        for coord in coords_names:
            vars_ = tuple(quantify_dataset.variables.keys())
            if coord not in vars_:
                raise ValueError(f"Coordinate {coord} not in coordinates {vars_}.")

    # Because xarray in general creates new objects and
    # due to https://github.com/pydata/xarray/issues/2245
    # the attributes need to be saved and restored in the new object
    attrs_coords = tuple(quantify_dataset[name].attrs for name in coords_names)
    # Convert "xi" variables to Coordinates
    dataset = quantify_dataset.set_coords(coords_names)

    # Convert to a gridded xarray dataset format

    if len(coords_names) == 1:
        # No unstacking needed just swap the dimension
        for var in quantify_dataset.data_vars.keys():
            if dimension in dataset[var].dims:
                dataset = dataset.update(
                    {var: dataset[var].swap_dims({dimension: coords_names[0]})}
                )
    else:
        # Make the Dimension `dimension` a MultiIndex(x0, x1, ...)
        dataset = dataset.set_index({dimension: coords_names})
        # See also: http://xarray.pydata.org/en/stable/reshaping.html#stack-and-unstack
        dataset = dataset.unstack(dim=dimension)
    for name, attrs in zip(coords_names, attrs_coords):
        dataset[name].attrs = attrs

    return dataset


# ######################################################################


def get_latest_tuid(contains: str = "") -> TUID:
    """
    Returns the most recent tuid.

    .. tip::

        This function is similar to :func:`~get_tuids_containing` but is preferred if
        one is only interested in the most recent
        :class:`~quantify_core.data.types.TUID` for performance reasons.

    Parameters
    ----------
    contains
        An optional string contained in the experiment name.
    Returns
    -------
    :
        The latest TUID.
    Raises
    ------
    FileNotFoundError
        No data found.
    """
    # `max_results=1, reverse=True` makes sure the tuid is found efficiently asap
    return get_tuids_containing(contains, max_results=1, reverse=True)[0]


# pylint: disable=too-many-locals
def get_tuids_containing(
    contains: str,
    t_start: Union[datetime.datetime, str] = None,
    t_stop: Union[datetime.datetime, str] = None,
    max_results: int = sys.maxsize,
    reverse: bool = False,
) -> list:
    """
    Returns a list of tuids containing a specific label.

    .. tip::

        If one is only interested in the most recent
        :class:`~quantify_core.data.types.TUID`, :func:`~get_latest_tuid` is preferred
        for performance reasons.

    Parameters
    ----------
    contains
        A string contained in the experiment name.
    t_start
        datetime to search from, inclusive. If a string is specified, it will be
        converted to a datetime object using :obj:`~dateutil.parser.parse`.
        If no value is specified, will use the year 1 as a reference t_start.
    t_stop
        datetime to search until, exclusive. If a string is specified, it will be
        converted to a datetime object using :obj:`~dateutil.parser.parse`.
        If no value is specified, will use the current time as a reference t_stop.
    max_results
        Maximum number of results to return. Defaults to unlimited.
    reverse
        If False, sorts tuids chronologically, if True sorts by most recent.
    Returns
    -------
    list
        A list of :class:`~quantify_core.data.types.TUID`: objects.
    Raises
    ------
    FileNotFoundError
        No data found.
    """
    datadir = get_datadir()
    if isinstance(t_start, str):
        t_start = parse(t_start)
    elif t_start is None:
        t_start = datetime.datetime(1, 1, 1)
    if isinstance(t_stop, str):
        t_stop = parse(t_stop)
    elif t_stop is None:
        t_stop = datetime.datetime.now()

    # date range filters, define here to make the next line more readable
    d_start = t_start.strftime("%Y%m%d")
    d_stop = t_stop.strftime("%Y%m%d")

    def lower_bound(dir_name):
        return dir_name >= d_start if d_start else True  # noqa: E731

    def upper_bound(dir_name):
        return dir_name <= d_stop if d_stop else True  # noqa: E731

    daydirs = list(
        filter(
            lambda x: (
                x.isdigit() and len(x) == 8 and lower_bound(x) and upper_bound(x)
            ),
            os.listdir(datadir),
        )
    )
    daydirs.sort(reverse=reverse)
    if len(daydirs) == 0:
        err_msg = f"There are no valid day directories in the data folder '{datadir}'"
        if t_start or t_stop:
            err_msg += f", for the range {t_start or ''} to {t_stop or ''}"
        raise FileNotFoundError(err_msg)

    tuids = []
    for daydir in daydirs:
        expdirs = list(
            filter(
                lambda x: (
                    len(x) > 25
                    and TUID.is_valid(x[:26])  # tuid is valid
                    and (contains in x)  # label is part of exp_name
                    and (t_start <= parse(x[:15]))  # tuid is after t_start
                    and (parse(x[:15]) < t_stop)  # tuid is before t_stop
                ),
                os.listdir(os.path.join(datadir, daydir)),
            )
        )
        expdirs.sort(reverse=reverse)
        for expname in expdirs:
            # Check for inconsistent folder structure for datasets portability
            if daydir != expname[:8]:
                raise FileNotFoundError(
                    f"Experiment container '{expname}' is in wrong day directory "
                    f"'{daydir}'"
                )
            tuids.append(TUID(expname[:26]))
            if len(tuids) == max_results:
                return tuids
    if len(tuids) == 0:
        raise FileNotFoundError(f"No experiment found containing '{contains}'")
    return tuids


def snapshot(update: bool = False, clean: bool = True) -> dict:
    """
    State of all instruments setup as a JSON-compatible dictionary (everything that the
    custom JSON encoder class :class:`qcodes.utils.helpers.NumpyJSONEncoder` supports).

    Parameters
    ----------
    update
        If True, first gets all values before filling the snapshot.
    clean
        If True, removes certain keys from the snapshot to create a more readable and
        compact snapshot.
    """

    snap = {"instruments": {}, "parameters": {}}
    for ins_name, ins_ref in Instrument._all_instruments.items():
        ref = ins_ref()
        # Check for dead weakrefs
        if ref is not None:
            snap["instruments"][ins_name] = ref.snapshot(update=update)

    if clean:
        exclude_keys = {
            "inter_delay",
            "post_delay",
            "vals",
            "instrument",
            "functions",
            "__class__",
            "raw_value",
            "instrument_name",
            "full_name",
            "val_mapping",
        }
        snap = delete_keys_from_dict(snap, exclude_keys)

    return snap


# ######################################################################
# Private utilities
# ######################################################################


def _xi_and_yi_match(dsets: Iterable) -> bool:
    """
    Checks if all xi and yi data variables in `dsets` match:

    Returns `True` only when all these conditions are met:

    - Same number of xi's
    - Same number of yi's
    - Same attributes for xi's across `dsets`
    - Same attributes for yi's across `dsets`
    - Same order of the xi's across `dsets`
    - Same order of the yi's across `dsets`

    Otherwise returns `False`.
    """
    return _vars_match(dsets, var_type="x") and _vars_match(dsets, var_type="y")


def _vars_match(dsets: Iterable, var_type="x") -> bool:
    """
    Checks if all the datasets have matching xi or yi.
    """

    def get_xi_attrs(dset):
        # Hash is used in order to ensure everything matches:
        # name, long_name, unit, number of xi
        return tuple(dset[xi].attrs for xi in _get_parnames(dset, var_type))

    iterator = map(get_xi_attrs, dsets)
    # We can compare to the first one always
    tup0 = next(iterator, None)

    for tup in iterator:
        if tup != tup0:
            return False

    # Also returns true if the dsets is empty
    return True


def _get_parnames(dset, par_type):
    attr = "coords" if par_type == "x" else "data_vars"
    return sorted(key for key in getattr(dset, attr).keys() if key.startswith(par_type))


def _is_batched(obj) -> bool:
    """
    N.B. This function cannot be imported from quantify_core.measurement.type due to
    some circular dependencies that it would create in the
    quantify_core.measurement.__init__

    Returns
    -------
    :
        The `.batched` attribute of the settable/gettable `obj`, `False` if not present.
    """
    return getattr(obj, "batched", False)


def _is_uniformly_spaced_array(points: np.ndarray, rel_tolerance: float = 0.001):
    """
    Determines if the points in the array are spaced uniformly.
    Intended mainly for `plotmon` to detect if it needs to interpolate the data first,
    otherwise `pyqtgraph` cannot handle the non-uniform case.

    Usually the points have been generated with `numpy.linspace()` or `numpy.arange`.

    This function is intended to be detect cases such as adaptively sampled datasets,
    logspace, etc..

    .. note::

        Assumes unique values. This means that if there are duplicates in `points`
        this function will return `False`. E.g.,

        .. jupyter-execute::

            import quantify_core.data.handling as dh

            assert dh._is_uniformly_spaced_array([1, 2, 2, 3, 4]) == False

        Additionally, assumes monotonously increasing or decreasing values.

    Parameters
    ----------
    points
        A 1-dimensional array of points (usually the setpoints in an experiment).
    rel_tolerance
        Maximum relative tolerance with respect to the size of a segment that would be
        generated by a :code:`numpy.linspace(min(points), max(points), len(points) - 1).
        The function returns :code:`False` if any segment in `points` violates this
        tolerance.
    """
    points = np.asarray(points)
    assert len(np.shape(points)) == 1, "Points must be 1-dimensional."

    # at least 3 points required
    if len(points) <= 2:
        return True

    max_, min_ = np.max(points), np.min(points)
    abs_tolerance = (max_ - min_) / (len(points) - 1) * rel_tolerance

    # Very likely by looking at the first and last segment we already know if it
    # is not uniform and the check is cheap to evaluate
    first_segment = np.abs(points[1] - points[0])
    last_segment = np.abs(points[-2] - points[-1])
    diff_first_last = np.abs(last_segment - first_segment)
    if diff_first_last > abs_tolerance:
        return False

    linspace = np.linspace(points[0], points[-1], len(points))
    diff_square = np.square(linspace[1:-1] - points[1:-1])
    if np.any(diff_square > np.square(abs_tolerance)):
        return False

    return True
