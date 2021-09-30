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


def mk_main_coord_attrs(
    uniformly_spaced: bool = True,
    is_main_coord: bool = True,
    is_secondary_coord: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for secondary coordinates.

    See :class:`~quantify_core.data.dataset_attrs.QCoordAttrs` for details.

    Parameters
    ----------
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.uniformly_spaced`.
    is_main_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_main_coord`.
    is_secondary_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_secondary_coord`.
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QCoordAttrs(
        uniformly_spaced=uniformly_spaced,
        is_main_coord=is_main_coord,
        is_secondary_coord=is_secondary_coord,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_secondary_coord_attrs(
    uniformly_spaced: bool = True,
    is_main_coord: bool = False,
    is_secondary_coord: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for secondary coordinates.

    See :class:`~quantify_core.data.dataset_attrs.QCoordAttrs` for details.

    Parameters
    ----------
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.uniformly_spaced`.
    is_main_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_main_coord`.
    is_secondary_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_secondary_coord`.
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QCoordAttrs(
        uniformly_spaced=uniformly_spaced,
        is_main_coord=is_main_coord,
        is_secondary_coord=is_secondary_coord,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_main_var_attrs(
    coords: List[str],
    grid: bool = True,
    uniformly_spaced: bool = True,
    is_main_var: bool = True,
    is_secondary_var: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for main variables.

    See :class:`~quantify_core.data.dataset_attrs.QVarAttrs` for details.

    Parameters
    ----------
    coords
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.coords`.
    grid
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.grid`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.uniformly_spaced`.
    is_main_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_main_var`.
    is_secondary_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_secondary_var`.
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QVarAttrs(
        grid=grid,
        uniformly_spaced=uniformly_spaced,
        is_main_var=is_main_var,
        is_secondary_var=is_secondary_var,
        coords=coords,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_secondary_var_attrs(
    coords: List[str],
    grid: bool = True,
    uniformly_spaced: bool = True,
    is_main_var: bool = False,
    is_secondary_var: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    A factory of attributes for secondary variables.

    See :class:`~quantify_core.data.dataset_attrs.QVarAttrs` for details.

    Parameters
    ----------
    coords
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.coords`.
    grid
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.grid`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.uniformly_spaced`.
    is_main_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_main_var`.
    is_secondary_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_secondary_var`.
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QVarAttrs(
        grid=grid,
        uniformly_spaced=uniformly_spaced,
        is_main_var=is_main_var,
        is_secondary_var=is_secondary_var,
        coords=coords,
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
