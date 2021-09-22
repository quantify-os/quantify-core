# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities used for creating examples for docs/tutorials/tests."""
# pylint: disable=too-many-arguments
from __future__ import annotations

from typing import List, Union, Any, Dict, Callable
from pathlib import Path
import xarray as xr
from quantify_core.data.types import TUID
from qcodes import Parameter
import quantify_core.data.handling as dh
import quantify_core.data.dataset_attrs as dd


def mk_dataset_attrs(
    tuid: Union[TUID, Callable[[], Any]] = dh.gen_tuid, **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for Quantify dataset.

    See :class:`~quantify_core.data.dataset_attrs.QDatasetAttrs` for details.

    Parameters
    ----------
    tuid
        If no tuid is provided a new one will be generated.
        See also :attr:`~quantify_core.data.dataset_attrs.QDatasetAttrs.tuid`.
    """
    attrs = dd.QDatasetAttrs(
        tuid=tuid() if callable(tuid) else tuid,
    ).to_dict()
    attrs.update(kwargs)

    return attrs


def mk_exp_coord_attrs(
    batched: bool = False,
    uniformly_spaced: bool = True,
    is_experiment_coord: bool = True,
    is_calibration_coord: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for calibration coordinates.

    See :class:`~quantify_core.data.dataset_attrs.QCoordAttrs` for details.

    Parameters
    ----------
    batched
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.batched`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.uniformly_spaced`.
    is_experiment_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_experiment_coord`.
    is_calibration_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_calibration_coord`.
    """
    attrs = dd.QCoordAttrs(
        batched=batched,
        uniformly_spaced=uniformly_spaced,
        is_experiment_coord=is_experiment_coord,
        is_calibration_coord=is_calibration_coord,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_cal_coord_attrs(
    batched: bool = False,
    uniformly_spaced: bool = True,
    is_experiment_coord: bool = False,
    is_calibration_coord: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for calibration coordinates.

    See :class:`~quantify_core.data.dataset_attrs.QCoordAttrs` for details.

    Parameters
    ----------
    batched
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.batched`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.uniformly_spaced`.
    is_experiment_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_experiment_coord`.
    is_calibration_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_calibration_coord`.
    """
    attrs = dd.QCoordAttrs(
        batched=batched,
        uniformly_spaced=uniformly_spaced,
        is_experiment_coord=is_experiment_coord,
        is_calibration_coord=is_calibration_coord,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_exp_var_attrs(
    experiment_coords: List[str],
    grid: bool = True,
    uniformly_spaced: bool = True,
    batched: bool = False,
    is_experiment_var: bool = True,
    is_calibration_var: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for experiment variables.

    See :class:`~quantify_core.data.dataset_attrs.QVarAttrs` for details.

    Parameters
    ----------
    experiment_coords
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.experiment_coords`.
    grid
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.grid`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.uniformly_spaced`.
    batched
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.batched`.
    is_experiment_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_experiment_var`.
    is_calibration_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_calibration_var`.
    """
    attrs = dd.QVarAttrs(
        grid=grid,
        uniformly_spaced=uniformly_spaced,
        batched=batched,
        is_experiment_var=is_experiment_var,
        is_calibration_var=is_calibration_var,
        experiment_coords=experiment_coords,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_cal_var_attrs(
    experiment_coords: List[str],
    grid: bool = True,
    uniformly_spaced: bool = True,
    batched: bool = False,
    is_experiment_var: bool = False,
    is_calibration_var: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for calibration variables.

    See :class:`~quantify_core.data.dataset_attrs.QVarAttrs` for details.

    Parameters
    ----------
    experiment_coords
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.experiment_coords`.
    grid
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.grid`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.uniformly_spaced`.
    batched
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.batched`.
    is_experiment_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_experiment_var`.
    is_calibration_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_calibration_var`.
    """
    attrs = dd.QVarAttrs(
        grid=grid,
        uniformly_spaced=uniformly_spaced,
        batched=batched,
        is_experiment_var=is_experiment_var,
        is_calibration_var=is_calibration_var,
        experiment_coords=experiment_coords,
    ).to_dict()

    attrs.update(kwargs)
    return attrs


def round_trip_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Writes a dataset to disk and loads it back returning it."""

    tuid = dataset.tuid
    assert tuid != ""
    dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, dataset)
    return dh.load_dataset(tuid)


def par_to_attrs(parameter: Union[Parameter, Any]) -> Dict[str, Any]:
    """
    Extracts unit and label from a parameter and returns a dictionary compatible with
    :class:`~quantify_core.data.dataset_attrs.QVarAttrs` and
    :class:`~quantify_core.data.dataset_attrs.QCoordAttrs`.

    Parameters
    ----------
    parameter
        An object with a `.unit` and `.label` attributes.

    Returns
    -------
    :
        The dictionary ``{"units": parameter.unit, "long_name": parameter.label}``.
    """
    return dict(units=parameter.unit, long_name=parameter.label)
