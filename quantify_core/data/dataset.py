# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities for dataset (python object) handling."""
# pylint: disable=too-many-instance-attributes
from __future__ import annotations
from typing import List, Union, Dict
from dataclasses import dataclass, field
from typing_extensions import Literal
from dataclasses_json import DataClassJsonMixin

# TODO: convert to Dataclasses/Traitlets with method # pylint: disable=fixme


@dataclass
class QExpCoordAttrs(DataClassJsonMixin):
    """
    A dataclass representing the attribute of experimental coordinates.

    All attributes are mandatory to be present but can be ``None``.
    """

    units: str = ""
    """The units of the values."""
    long_name: str = ""
    """A long name for this coordinate."""
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
class QExpVarAttrs(DataClassJsonMixin):
    """
    A dataclass representing the attribute of experimental coordinates.

    All attributes are mandatory to be present but can be ``None``.
    """

    units: str = ""
    """The units of the values."""
    long_name: str = ""
    """A long name for this coordinate."""
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
    is_dataset_ref: bool = False
    """Flags if it is an array of :class:`quantify_core.data.types.TUID` s of other
    dataset."""

    json_attrs: List[str] = field(
        # ``None`` and ``Dict``
        default_factory=lambda: ["batched", "batch_size", "uniformly_spaced", "grid"]
    )
    """A list of strings corresponding to the names of other attributes that require to
    be json-serialized in order to be able to write them to disk using the ``h5netcdf``
    engine.

    Note that the default values in this list should be included as well."""


@dataclass
class QDatasetAttrs(DataClassJsonMixin):
    """
    A dataclass representing the attribute of the Quantify dataset.

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
    experiment_coords: List[str] = field(default_factory=list)
    """A list specifying the experiment coordinates.
    See :ref:`sec-experiment-coordinates-and-variables` for terminology.
    """
    experiment_data_vars: List[str] = field(default_factory=list)
    """A list specifying the experiment variables.
    See :ref:`sec-experiment-coordinates-and-variables` for terminology.
    """
    calibration_coords_map: Dict[str, Union[str, List[str]]] = field(
        default_factory=dict
    )
    """A mapping that maps a calibration coordinate or a list of calibration coordinates
    to another calibration coordinate.
    See :ref:`sec-experiment-coordinates-and-variables` for terminology.
    """
    calibration_data_vars_map: Dict[str, Union[str, List[str]]] = field(
        default_factory=dict
    )
    """A mapping that maps a calibration variable or a list of calibration
    variables to another calibration variable.
    See :ref:`sec-experiment-coordinates-and-variables` for terminology.
    """
    quantify_dataset_version: str = "2.0.0"
    """A string identifying the version of this Quantify dataset for future backwards
    compatibility."""
    software_versions: Dict[str, str] = field(default_factory=dict)
    """A mapping of other relevant software packages that are relevant to log for this
    dataset. Another example is the git tag or hash of a commit of a lab repository."""

    json_attrs: List[str] = field(
        default_factory=lambda: [  # ``None`` and ``Dict``
            "tuid",
            "experiment_state",
            "experiment_start",
            "experiment_end",
            "calibration_data_vars_map",
            "calibration_coords_map",
            "software_versions",
        ]
    )
    """A list of strings corresponding to the names of other attributes that require to
    be json-serialized in order to be able to write them to disk using the ``h5netcdf``
    engine.

    Note that the default values in this list should be included as well."""
