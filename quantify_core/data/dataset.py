# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities for dataset (python object) handling."""
# pylint: disable=too-many-instance-attributes
from __future__ import annotations
from typing import Set, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin

# TODO: convert to Dataclasses/Traitlets with method # pylint: disable=fixme


@dataclass
class QExpCoordAttrs(DataClassJsonMixin):
    """A dataclass representing the attribute of experimental coordinates."""

    serialize_to_json: List[str] = field(
        default_factory=lambda: ["batched", "batch_size", "uniformly_spaced"]
    )
    """A list of keys corresponding to the names of other attributes that require to be
    json-serialized in order to be able to write them to disk using the ``h5netcdf``
    engine."""

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
    """flag if it is an array of tuids of other dataset."""


@dataclass
class QExpVarAttrs(DataClassJsonMixin):
    """A dataclass representing the attribute of experimental coordinates."""

    serialize_to_json: List[str] = field(
        # this ones
        default_factory=lambda: ["batched", "batch_size", "uniformly_spaced", "grid"]
    )
    """A list of keys corresponding to the names of other attributes that require to be
    json-serialized in order to be able to write them to disk using the ``h5netcdf``
    engine.

    Note that the default one should be preserved."""

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
    uniformly spaced along all dimensions."""
    is_dataset_ref: bool = False
    """flag if it is an array of tuids of other dataset."""


def mk_default_dataset_attrs(**kwargs) -> dict:

    tuid: str = ""
    experiment_name: str = ""
    experiment_state: str = ""  # running/interrupted (safely)/interrupted (forced)/done
    experiment_start: str = ""  # unambiguous timestamp format to be defined
    experiment_end: str = ""  # optional, unambiguous timestamp format to be defined
    experiment_coords: List[str] = []
    experiment_data_vars: List[str] = []
    # dictionaries are not allowed with the h5netcdf backend so we use tuples
    # entries: (experiment var. name, calibration var. name)
    calibration_data_vars_map: List[Tuple[str, str]] = []
    # entries: (experiment coord. name, calibration coord. name)
    calibration_coords_map: List[Tuple[str, str]] = []
    quantify_dataset_version: str = "2.0.0"
    # entries: (package or repo name, version tag or commit hash)
    software_versions: List[Tuple[str, str]] = []

    attrs = dict(
        tuid=tuid,
        experiment_name=experiment_name,
        experiment_state=experiment_state,
        experiment_start=experiment_start,
        experiment_end=experiment_end,
        experiment_coords=experiment_coords,
        experiment_data_vars=experiment_data_vars,
        calibration_data_vars_map=calibration_data_vars_map,
        calibration_coords_map=calibration_coords_map,
        quantify_dataset_version=quantify_dataset_version,
        software_versions=software_versions,
    )
    attrs.update(kwargs)

    return attrs
