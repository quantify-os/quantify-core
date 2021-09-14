# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities for dataset (python object) handling."""

from __future__ import annotations
from typing import List, Tuple

# TODO: convert to Dataclasses/Traitlets with method # pylint: disable=fixme


def mk_default_exp_coord_attrs(**kwargs) -> dict:

    units: str = ""
    long_name: str = ""
    # netCDF does not support `None`
    # as a workaround for attribute whose type is not str we use a custom str
    batched: bool = "__undefined_bool__"
    batch_size: int = "__undefined_int__"
    uniformly_spaced: bool = "__undefined_bool__"
    is_dataset_ref = False  # to flag if it is an array of tuids of other dataset

    attrs = dict(
        units=units,
        long_name=long_name,
        batched=batched,
        batch_size=batch_size,
        uniformly_spaced=uniformly_spaced,
        is_dataset_ref=is_dataset_ref,
    )
    attrs.update(kwargs)

    return attrs


def mk_default_exp_var_attrs(**kwargs) -> dict:

    units: str = ""
    long_name: str = ""
    batched: bool = "__undefined_bool__"
    batch_size: int = "__undefined_int__"
    # this attribute only makes sense to have for each exp. variable
    # in case we later make use of more dimensions this will be specially relevant
    grid: bool = "__undefined__"
    # included here because some vars can be exp. coords but a MultiIndex
    # is not supported yet
    uniformly_spaced: bool = "__undefined_bool__"
    is_dataset_ref: bool = False  # to flag if it is an array of tuids of other dataset

    attrs = dict(
        units=units,
        long_name=long_name,
        batched=batched,
        batch_size=batch_size,
        grid=grid,
        uniformly_spaced=uniformly_spaced,
        is_dataset_ref=is_dataset_ref,
    )
    attrs.update(kwargs)

    return attrs


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
