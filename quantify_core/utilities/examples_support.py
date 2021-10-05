# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities used for creating examples for docs/tutorials/tests."""
# pylint: disable=too-many-arguments
from __future__ import annotations

from typing import List, Union, Any, Dict, Callable
from pathlib import Path
import xarray as xr
import numpy as np
from quantify_core.data.types import TUID
import quantify_core.data.handling as dh
import quantify_core.data.dataset_attrs as dd

# ######################################################################################
# IQ-related data manipulation and plotting
# ######################################################################################


def mk_iq_shots(
    n_shots: int,
    sigmas: Union[tuple, list, np.ndarray],
    centers: Union[tuple, list, np.ndarray],
    probabilities: Union[tuple, list, np.ndarray],
    seed: Union[int, None] = 112233,
):
    """
    Generates clusters of (I + 1j*Q) points with a Gaussian distribution with the
    specified sigmas and centers according to the probabilities of each cluster

    .. admonition:: Examples
        :class: dropdown

        .. include:: ./examples/utilities.examples_support.mk_iq_shots.rst.txt

    Parameters
    ----------
    n_shots
        The number of shot to generate.
    sigma
        The sigma of the Gaussian distribution used for both real and imaginary parts.
    centers
        The center of each cluster on the imaginary plane.
    probabilities
        The probabilities of each cluster being randomly selected for each shot.
    seed
        Random number generator passed to ``numpy.random.default_rng``.
    """
    assert len(sigmas) == len(centers) == len(probabilities)

    rng = np.random.default_rng(seed=seed)

    cluster_indices = tuple(range(len(centers)))
    choices = rng.choice(a=cluster_indices, size=n_shots, p=probabilities)

    shots = []
    for idx in cluster_indices:
        num_shots_this_cluster = np.sum(choices == idx)
        i_data = rng.normal(
            loc=centers[idx].real,
            scale=sigmas[idx],
            size=num_shots_this_cluster,
        )
        q_data = rng.normal(
            loc=centers[idx].imag,
            scale=sigmas[idx],
            size=num_shots_this_cluster,
        )
        shots.append(i_data + 1j * q_data)
    return np.concatenate(shots)


def mk_trace_time(sampling_rate: float = 1e9, duration: float = 0.3e-6) -> np.ndarray:
    """
    Generates a :obj:`~numpy.arange` in which the entries correspond to time instants
    up to ``duration`` seconds sampled according to ``sampling_rate`` in Hz.

    See :func:`~.mk_trace_for_iq_shot` for an usage example.

    Parameters
    ----------
    sampling_rate
        The sampling rate in Hz.
    duration
        Total duration in seconds.

    Returns
    -------
    :
        An array with the time instants.
    """
    trace_length = sampling_rate * duration
    return np.arange(0, trace_length, 1) / sampling_rate


def mk_trace_for_iq_shot(
    iq_point: complex,
    time_values: np.ndarray = mk_trace_time(),
    intermediate_freq: float = 50e6,
) -> np.ndarray:
    """
    Generates mock "traces" that a physical instrument would digitize for the readout of
    a transmon qubit when using a down-converting IQ mixer.

    .. admonition:: Examples
        :class: dropdown

        .. include:: ./examples/utilities.examples_support.mk_trace_for_iq_shot.rst.txt

    Parameters
    ----------
    iq_point
        A complex number representing a point on the IQ-plane.
    time_values
        The time instants at which the mock intermediate-frequency signal is sampled.
    intermediate_freq
        The intermediate frequency used in the down-conversion scheme.

    Returns
    -------
    :
        An array of complex numbers.
    """

    return iq_point * np.exp(2.0j * np.pi * intermediate_freq * time_values)


def plot_centroids(
    ax,
    ground: complex,
    excited: complex,
    markersize: int = 10,
    legend: bool = True,
    **kwargs
):
    """Plots the centers of the ground and excited states on a 2D plot representing
    the IQ-plane.

    Parameters
    ----------
    ax
        A matplotlib axis to plot on.
    ground
        A complex number representing the ground state on the IQ-plane.
    excited
        A complex number representing the excited state on the IQ-plane.
    markersize
        The size of the markers used to plot
    legend
        Calls ``ax.legend`` if ``True``.
    **kwargs
        Keyword arguments passed to the ``ax.plot``.
    """
    ax.plot(
        [ground.real],
        [ground.imag],
        label="|0>",
        marker="o",
        color="C3",
        linestyle="",
        markersize=markersize,
        **kwargs
    )
    ax.plot(
        [excited.real],
        [excited.imag],
        label="|1>",
        marker="^",
        color="C4",
        linestyle="",
        markersize=markersize,
        **kwargs
    )
    if legend:
        ax.legend()


# ######################################################################################
# Dataset-related
# ######################################################################################


def mk_dataset_attrs(
    tuid: Union[TUID, Callable[[], TUID]] = dh.gen_tuid, **kwargs
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
