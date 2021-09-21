# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""
Utilities for handling the attributes of :class:`xarray.Dataset` and
:class:`xarray.DataArray` (python objects) handling.
"""
# pylint: disable=too-many-instance-attributes
from __future__ import annotations

from typing import List, Union, Dict, Any, Tuple
from dataclasses import dataclass, field
from typing_extensions import Literal

import xarray as xr
from dataclasses_json import DataClassJsonMixin


@dataclass
class QDatasetIntraRelationship(DataClassJsonMixin):
    """
    A dataset representing a dictionary that specifies a relationship between dataset
    variables. A prominent example are calibration points contained within one variable
    or several variables that are necessary to interpret correctly the data of another
    variable.
    """

    item_name: str = None
    """
    The name of the coordinate/variable to which we want to relate other
    coordinates/variables.
    """
    relation_type: str = None
    """A string specifying the type of relationship.

    Reserved relation types:

    ``"calibration"`` - Specifies a list of experiment variables used as calibration
    data for the experiment variables whose name is specified by the ``item_name``.
    """
    related_names: List[str] = field(default_factory=list)
    """A list of names related to the ``item_name``."""
    relation_metadata: Dict[str, Any] = field(default_factory=dict)
    """
    A free-form dictionary to store additional information relevant to this
    relationship.
    """


@dataclass
class QCoordAttrs(DataClassJsonMixin):
    """
    A dataclass representing the :attr:`~xarray.DataArray.attrs` attribute of
    experiment and calibration coordinates.

    All attributes are mandatory to be present but can be ``None``.
    """

    units: str = ""
    """The units of the values."""
    long_name: str = ""
    """A long name for this coordinate."""
    is_experiment_coord: bool = True
    """A convenient attribute, set to ``True``, to flag xarray coordinates that
    correspond to experiment coordinates."""
    is_calibration_coord: bool = False
    """If ``True``, this experiment coordinates is intended to be a coordinate for
    an experiment variable that corresponds to calibration data."""
    batched: Union[bool, None] = None
    """True if this coordinates corresponds to a batched settable."""
    batch_size: Union[int, None] = None
    """The (maximum) size of a batch supported by the corresponding settable."""
    uniformly_spaced: Union[bool, None] = None
    """Indicates if the values are uniformly spaced."""
    is_dataset_ref: bool = False
    """Flags if it is an array of :class:`quantify_core.data.types.TUID` s of other
    dataset."""

    json_attrs: List[str] = field(
        # ``None`` and ``Dict``
        default_factory=lambda: ["batched", "batch_size", "uniformly_spaced"]
    )
    """A list of strings corresponding to the names of other attributes that require to
    be json-serialized in order to be able to write them to disk using the ``h5netcdf``
    engine.

    Note that the default values in this list should be included as well."""


@dataclass
class QVarAttrs(DataClassJsonMixin):
    """
    A dataclass representing the :attr:`~xarray.DataArray.attrs` attribute of
    experiment and calibration variables.

    All attributes are mandatory to be present but can be ``None``.
    """

    units: str = ""
    """The units of the values."""
    long_name: str = ""
    """A long name for this coordinate."""
    is_experiment_var: bool = True
    """A convenient attribute, set to ``True``, to flag xarray data variables that
    correspond to experiment variables."""
    is_calibration_var: bool = False
    """If ``True``, the data of this experiment variable is intended to be used to
    calibrate the data of another experiment variable. E.g., this experiment variable
    could contain the amplitude of a signal for when a qubit is in the excited and
    ground states."""
    batched: Union[bool, None] = None
    """True if this coordinates corresponds to a batched settable."""
    batch_size: Union[int, None] = None
    """The (maximum) size of a batch supported by the corresponding settable."""
    uniformly_spaced: Union[bool, None] = None
    """Indicates if the values are uniformly spaced.
    This does not apply to 'true' experiment variables but, because a MultiIndex is not
    supported yet by xarray, some coordinate variables have to be stored as experiment
    variables instead.
    """

    # This attribute only makes sense to have for each exp. variable instead of
    # attaching it to the full dataset.
    # In case we later make use of more dimensions this will be specially relevant.
    grid: Union[bool, None] = None
    """Indicates if the variables data are located on a grid, which does not need to be
    uniformly spaced along all dimensions. In other words, specifies if the
    corresponding experiment coordinates are the 'unrolled' points (also known as
    'unstacked') corresponding to a grid.

    If ``True`` than it is possible to use
    :func:`quantify_core.data.handling.to_gridded_dataset()` to convert the variables to
    a 'stacked' version.
    """
    experiment_coords: List[str] = None
    """The experiment coordinates that index this experiment variable. This is
    information is necessary to avoid ambiguities in the dataset.

    E.g. if we would measure a signal ``amplitude`` as a function of ``time``. The
    experiment variable is ``amplitude`` and we would have
    ``experiment_coords=["time"]``. For a 2D dataset we could have
    ``experiment_coords=["time", "frequency"]``.
    """
    is_dataset_ref: bool = False
    """Flags if it is an array of :class:`quantify_core.data.types.TUID` s of other
    dataset."""

    json_attrs: List[str] = field(
        # ``None`` and ``Dict``
        default_factory=lambda: [
            "batched",
            "batch_size",
            "uniformly_spaced",
            "grid",
            "experiment_coords",
        ]
    )
    """A list of strings corresponding to the names of other attributes that require to
    be json-serialized in order to be able to write them to disk using the ``h5netcdf``
    engine.

    Note that the default values in this list should be included as well."""


@dataclass
class QDatasetAttrs(DataClassJsonMixin):
    """
    A dataclass representing the :attr:`~xarray.Dataset.attrs` attribute of the
    Quantify dataset.

    All attributes are mandatory to be present but can be ``None``.
    """

    tuid: Union[str, None] = None
    """The time-based unique identifier of the dataset.
    See :class:`quantify_core.data.types.TUID`."""
    experiment_name: str = ""
    """Experiment name, same as the the experiment name included in the name of the
    experiment container."""
    experiment_state: Union[
        Literal["running", "interrupted (safety)", "interrupted (forced)", "done"], None
    ] = None
    """Denotes the last known state of the experiment. Can be used later to filter
    'bad' datasets."""
    experiment_start: Union[str, None] = None
    """Human-readable timestamp, including timezone, format to be defined.
    Specifies when the experiment/data acquisition started.
    """
    experiment_end: Union[str, None] = None
    """Human-readable timestamp, including timezone, format to be defined.
    Specifies when the experiment/data acquisition ended.
    """
    quantify_dataset_version: str = "2.0.0"
    """A string identifying the version of this Quantify dataset for future backwards
    compatibility."""
    software_versions: Dict[str, str] = field(default_factory=dict)
    """A mapping of other relevant software packages that are relevant to log for this
    dataset. Another example is the git tag or hash of a commit of a lab repository."""
    relationships: List[QDatasetIntraRelationship] = field(default_factory=list)
    """A list of relationships within the dataset specified as list of dictionaries
    that comply with the :class:`~.QDatasetIntraRelationship`."""
    repetitions_dims: List[str] = field(default_factory=list)
    """A list of xarray dimension names which correspond to outermost dimensions along
    which experiment coordinates and variables lie on. This attribute is intended to
    allow easy programmatic detection of such dimension. This can be used, for example,
    to average along these dimensions before an automatic live plotting.
    """

    json_attrs: List[str] = field(
        default_factory=lambda: [  # ``None`` and ``Dict``
            "tuid",
            "experiment_state",
            "experiment_start",
            "experiment_end",
            "software_versions",
            "relationships",
        ]
    )
    """A list of strings corresponding to the names of other attributes that require to
    be json-serialized in order to be able to write them to disk using the ``h5netcdf``
    engine.

    Note that the default values in this list should be included as well."""


def _get_dims(dataset: xr.Dataset) -> Tuple[List[str], List[str]]:
    """Return main dimensions separated in non-calibration and calibration ones."""
    main_dims = set()
    calib_dims = set()
    for var_or_coords in (dataset.coords.values(), dataset.data_vars.values()):
        for var in var_or_coords:
            if var.attrs.get("is_experiment_var", False):
                # Check if the outermost dimension is a repetitions dimension
                if var.dims[0] in dataset.attrs.get("repetitions_dims", []):
                    to_add = var.dims[1]
                else:
                    to_add = var.dims[0]

                is_cal = var.attrs.get(
                    "is_calibration_var", var.attrs.get("is_calibration_coord", False)
                )
                if is_cal:
                    calib_dims.add(to_add)
                else:
                    main_dims.add(to_add)

    return list(main_dims), list(calib_dims)


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_main_dims(dataset: xr.Dataset) -> List[str]:
    """Determines the 'main' dimensions in the dataset.

    Each of the dimensions returned is the outermost dimension for an experiment
    coordinate/variable, or the one after a dimension listed in
    ``~QDatasetAttrs.repetition_dims``.

    These dimensions are detected based on :attr:`~.QCoordAttrs.is_experiment_coord`
    and :attr:`~.QVarAttrs.is_experiment_var` attributes.

    .. warning::

        The dimensions listed in this list should be considered "incompatible" in the
        sense that the experiment coordinate/variables must lie on one and only one of
        such dimension.

    .. note::

        The dimensions, on which the calibration coordinates/variables lie, are not
        included in this list.

    Parameters
    ----------
    dataset
        The dataset from which to extract the main dimensions.

    Returns
    -------
    :
        The names of the 'main' experiment dimensions in the dataset.
    """

    main_dims, _ = _get_dims(dataset)

    return main_dims


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_main_calibration_dims(dataset: xr.Dataset) -> List[str]:
    """Returns the 'main' calibration dimensions.

    For details see :func:`~.get_main_dims`, :attr:`~.QVarAttrs.is_calibration_var`
    and :attr:`~.QCoordAttrs.is_calibration_coord`.

    Parameters
    ----------
    dataset
        The dataset from which to extract the main dimensions.

    Returns
    -------
    :
        The names of the 'main' dimensions of calibration coordinates/variables in the
        dataset.
    """

    _, calib_dims = _get_dims(dataset)

    return calib_dims


def _get_all_variables(
    dataset: xr.Dataset,
    var_type: str = Literal["coord", "var"],
    attr_type: str = Literal["experiment", "calibration"],
) -> Tuple[List[str], List[str]]:
    """Shared internal logic used to retrieve variables/coordinates names."""

    variables = dataset.data_vars if var_type == "var" else dataset.coords

    var_names = []
    for var_name, var in variables.items():
        if var.attrs.get(f"is_{attr_type}_{var_type}", False):
            var_names.append(var_name)

    return var_names


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_experiment_vars(dataset: xr.Dataset) -> List[str]:
    """
    Finds the experiment variables in the dataset (except calibration variables).

    Finds the xarray data variables in the dataset that have their attributes
    :attr:`~.QVarAttrs.is_experiment_var` set to ``True`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the experiment variables.
    """
    return _get_all_variables(dataset, var_type="var", attr_type="experiment")


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_calibration_vars(dataset: xr.Dataset) -> List[str]:
    """
    Finds the experiment calibration variables in the dataset.

    Finds the xarray data variables in the dataset that have their attributes
    :attr:`~.QVarAttrs.is_calibration_var` set to ``True`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the experiment calibration variables.
    """
    return _get_all_variables(dataset, var_type="var", attr_type="calibration")


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_experiment_coords(dataset: xr.Dataset) -> List[str]:
    """
    Finds the experiment coordinates in the dataset (except calibration coordinates).

    Finds the xarray coordinates in the dataset that have their attributes
    :attr:`~.QCoordAttrs.is_experiment_coord` set to ``True`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the experiment coordinates.
    """
    return _get_all_variables(dataset, var_type="coord", attr_type="experiment")


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_calibration_coords(dataset: xr.Dataset) -> List[str]:
    """
    Finds the experiment calibration coordinates in the dataset.

    Finds the xarray coordinates in the dataset that have their attributes
    :attr:`~.QCoordAttrs.is_calibration_coord` set to ``True`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the experiment calibration coordinates.
    """
    return _get_all_variables(dataset, var_type="coord", attr_type="calibration")
