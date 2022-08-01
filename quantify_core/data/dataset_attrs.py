# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""
Utilities for handling the attributes of :class:`xarray.Dataset` and
:class:`xarray.DataArray` (python objects) handling.
"""
# pylint: disable=too-many-instance-attributes
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Union

import xarray as xr
from dataclasses_json import DataClassJsonMixin


@dataclass
class QDatasetIntraRelationship(DataClassJsonMixin):
    """
    A dataclass representing a dictionary that specifies a relationship between dataset
    variables.

    A prominent example are calibration points contained within one variable
    or several variables that are necessary to interpret correctly the data of another
    variable.

    .. admonition:: Examples

        .. include:: /examples/data.dataset_attrs.QDatasetIntraRelationship.rst.txt
    """

    item_name: str = None
    """
    The name of the coordinate/variable to which we want to relate other
    coordinates/variables.
    """
    relation_type: str = None
    """A string specifying the type of relationship.

    Reserved relation types:

    ``"calibration"`` - Specifies a list of main variables used as calibration
    data for the main variables whose name is specified by the ``item_name``.
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
    main and secondary coordinates.

    All attributes are mandatory to be present but can be ``None``.

    .. admonition:: Examples

        .. include:: /examples/data.dataset_attrs.QCoordAttrs.rst.txt
    """

    unit: str = ""
    """The units of the values."""
    long_name: str = ""
    """A long name for this coordinate."""
    is_main_coord: bool = None
    """When set to ``True``, flags the xarray coordinate to correspond to a main
    coordinate, otherwise (``False``) it corresponds to a secondary coordinate."""
    uniformly_spaced: Union[bool, None] = None
    """Indicates if the values are uniformly spaced."""
    is_dataset_ref: bool = False
    """Flags if it is an array of :class:`quantify_core.data.types.TUID` s of other
    dataset."""

    json_serialize_exclude: List[str] = field(default_factory=list)
    """A list of strings corresponding to the names of other attributes that should not
    be json-serialized when writing the dataset to disk. Empty by default.
    """


@dataclass
class QVarAttrs(DataClassJsonMixin):
    """
    A dataclass representing the :attr:`~xarray.DataArray.attrs` attribute of
    main and secondary variables.

    All attributes are mandatory to be present but can be ``None``.

    .. admonition:: Examples

        .. include:: /examples/data.dataset_attrs.QVarAttrs.rst.txt
    """

    unit: str = ""
    """The units of the values."""
    long_name: str = ""
    """A long name for this coordinate."""
    is_main_var: bool = None
    """When set to ``True``, flags this xarray data variable to correspond to a main
    variable, otherwise (``False``) it corresponds to a secondary variable."""
    uniformly_spaced: Union[bool, None] = None
    """Indicates if the values are uniformly spaced.
    This does not apply to 'true' main variables but, because a MultiIndex is not
    supported yet by xarray when writing to disk, some coordinate variables have to be
    stored as main variables instead.
    """

    # This attribute only makes sense to have for each main variable instead of
    # attaching it to the full dataset.
    # In case we later make use of more dimensions this will be specially relevant.
    grid: Union[bool, None] = None
    """Indicates if the variables data are located on a grid, which does not need to be
    uniformly spaced along all dimensions. In other words, specifies if the
    corresponding main coordinates are the 'unrolled' points (also known as
    'unstacked') corresponding to a grid.

    If ``True`` than it is possible to use
    :func:`quantify_core.data.handling.to_gridded_dataset()` to convert the variables to
    a 'stacked' version.
    """
    is_dataset_ref: bool = False
    """Flags if it is an array of :class:`quantify_core.data.types.TUID` s of other
    dataset. See also :ref:`sec-nested-mc-example`."""
    has_repetitions: bool = False
    """Indicates that the outermost dimension of this variable is a repetitions
    dimension. This attribute is intended to allow easy programmatic detection of such
    dimension. It can be used, for example, to average along this dimension before an
    automatic live plotting or analysis.
    """

    json_serialize_exclude: List[str] = field(default_factory=list)
    """A list of strings corresponding to the names of other attributes that should not
    be json-serialized when writing the dataset to disk. Empty by default.
    """


@dataclass
class QDatasetAttrs(DataClassJsonMixin):
    """
    A dataclass representing the :attr:`~xarray.Dataset.attrs` attribute of the
    Quantify dataset.

    All attributes are mandatory to be present but can be ``None``.

    .. admonition:: Example

        .. include:: /examples/data.dataset_attrs.QDatasetAttrs.rst.txt
    """

    tuid: Union[str, None] = None
    """The time-based unique identifier of the dataset.
    See :class:`quantify_core.data.types.TUID`."""
    dataset_name: str = ""
    """The dataset name, usually same as the the experiment name included in the name of
    the experiment container."""
    dataset_state: Literal[
        None, "running", "interrupted (safety)", "interrupted (forced)", "done"
    ] = None
    """Denotes the last known state of the experiment/data acquisition that served to
    'build' this dataset. Can be used later to filter 'bad' datasets.
    """
    timestamp_start: Union[str, None] = None
    """Human-readable timestamp (ISO8601) as returned by
    :code:`pendulum.now().to_iso8601_string()`
    (`docs <https://pendulum.eustace.io/docs/>`_).
    Specifies when the experiment/data acquisition started.
    """
    timestamp_end: Union[str, None] = None
    """Human-readable timestamp (ISO8601) as returned by
    :code:`pendulum.now().to_iso8601_string()`
    (`docs <https://pendulum.eustace.io/docs/>`_).
    Specifies when the experiment/data acquisition ended.
    """
    quantify_dataset_version: str = "2.0.0"
    """A string identifying the version of this Quantify dataset for backwards
    compatibility."""
    software_versions: Dict[str, str] = field(default_factory=dict)
    """A mapping of other relevant software packages that are relevant to log for this
    dataset. Another example is the git tag or hash of a commit of a lab repository.

    .. admonition:: Example

        .. include:: /examples/data.dataset_attrs.QDatasetAttrs.software_versions.rst.txt
    """  # pylint: disable=line-too-long
    relationships: List[QDatasetIntraRelationship] = field(default_factory=list)
    """A list of relationships within the dataset specified as list of dictionaries
    that comply with the :class:`~.QDatasetIntraRelationship`."""

    json_serialize_exclude: List[str] = field(default_factory=list)
    """A list of strings corresponding to the names of other attributes that should not
    be json-serialized when writing the dataset to disk. Empty by default.
    """


def _get_dims(dataset: xr.Dataset, main: bool) -> Tuple[List[str], List[str]]:
    """Return main or secondary dimensions."""
    dims = set()
    for vars_or_coords in (dataset.coords.values(), dataset.data_vars.values()):
        for var in vars_or_coords:
            is_main_var = var.attrs.get("is_main_var", None)
            is_main_coord = var.attrs.get("is_main_coord", None)
            if is_main_var is main or is_main_coord is main:
                # Check if the outermost dimension is a repetitions dimension
                if var.attrs.get("has_repetitions", False):
                    dims.add(var.dims[1])
                else:
                    dims.add(var.dims[0])

    return list(dims)


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_main_dims(dataset: xr.Dataset) -> List[str]:
    """Determines the 'main' dimensions in the dataset.

    Each of the dimensions returned is the outermost dimension for an main
    coordinate/variable, OR the second one when a repetitions dimension is present.
    (see :attr:`~.QVarAttrs.has_repetitions`).

    These dimensions are detected based on :attr:`~.QCoordAttrs.is_main_coord`
    and :attr:`~.QVarAttrs.is_main_var` attributes.

    .. warning::

        The dimensions listed in this list should be considered "incompatible" in the
        sense that the main coordinate/variables must lie on one and only one of
        such dimension.

    .. note::

        The dimensions, on which the secondary coordinates/variables lie, are not
        included in this list. See also :func:`~.get_secondary_dims`.

    Parameters
    ----------
    dataset
        The dataset from which to extract the main dimensions.

    Returns
    -------
    :
        The names of the main dimensions in the dataset.
    """

    return _get_dims(dataset, main=True)


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_secondary_dims(dataset: xr.Dataset) -> List[str]:
    """Returns the 'main' secondary dimensions.

    For details see :func:`~.get_main_dims`, :attr:`~.QVarAttrs.is_main_var`
    and :attr:`~.QCoordAttrs.is_main_coord`.

    Parameters
    ----------
    dataset
        The dataset from which to extract the main dimensions.

    Returns
    -------
    :
        The names of the 'main' dimensions of secondary coordinates/variables in the
        dataset.
    """

    return _get_dims(dataset, main=False)


def _get_all_variables(
    dataset: xr.Dataset, var_type: Literal["coord", "var"], is_main: bool
) -> Tuple[List[str], List[str]]:
    """Shared internal logic used to retrieve variables/coordinates names."""

    variables = dataset.data_vars if var_type == "var" else dataset.coords

    var_names = []
    for var_name, var in variables.items():
        if var.attrs.get(f"is_main_{var_type}", False) is is_main:
            var_names.append(var_name)

    return var_names


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_main_vars(dataset: xr.Dataset) -> List[str]:
    """
    Finds the main variables in the dataset (except secondary variables).

    Finds the xarray data variables in the dataset that have their attributes
    :attr:`~.QVarAttrs.is_main_var` set to ``True`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the main variables.
    """
    return _get_all_variables(dataset, var_type="var", is_main=True)


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_secondary_vars(dataset: xr.Dataset) -> List[str]:
    """
    Finds the secondary variables in the dataset.

    Finds the xarray data variables in the dataset that have their attributes
    :attr:`~.QVarAttrs.is_main_var` set to ``False`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the secondary variables.
    """
    return _get_all_variables(dataset, var_type="var", is_main=False)


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_main_coords(dataset: xr.Dataset) -> List[str]:
    """
    Finds the main coordinates in the dataset (except secondary coordinates).

    Finds the xarray coordinates in the dataset that have their attributes
    :attr:`~.QCoordAttrs.is_main_coord` set to ``True`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the main coordinates.
    """
    return _get_all_variables(dataset, var_type="coord", is_main=True)


# FIXME add as a dataset property to the quantify dataset # pylint: disable=fixme
def get_secondary_coords(dataset: xr.Dataset) -> List[str]:
    """
    Finds the secondary coordinates in the dataset.

    Finds the xarray coordinates in the dataset that have their attributes
    :attr:`~.QCoordAttrs.is_main_coord` set to ``False`` (inside the
    :attr:`xarray.DataArray.attrs` dictionary).

    Parameters
    ----------
    dataset
        The dataset to scan.

    Returns
    -------
    :
        The names of the secondary coordinates.
    """
    return _get_all_variables(dataset, var_type="coord", is_main=False)
