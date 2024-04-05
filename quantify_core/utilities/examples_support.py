# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Utilities used for creating examples for docs/tutorials/tests."""
# pylint: disable=too-many-arguments
from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
from qcodes.instrument import Instrument
from qcodes.parameters import ManualParameter

import quantify_core.data.dataset_attrs as dd
import quantify_core.data.handling as dh
from quantify_core.analysis.fitting_models import cos_func

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from quantify_core.data.types import TUID

# ######################################################################################
# Tutorial utilities
# ######################################################################################


def mk_cosine_instrument() -> Instrument:
    """A container of parameters (mock instrument) providing a cosine model."""

    instr = Instrument("ParameterHolder")

    # ManualParameter's is a handy class that preserves the QCoDeS' Parameter
    # structure without necessarily having a connection to the physical world
    instr.add_parameter(
        "amp",
        initial_value=0.5,
        unit="V",
        label="Amplitude",
        parameter_class=ManualParameter,
    )
    instr.add_parameter(
        "freq",
        initial_value=1,
        unit="Hz",
        label="Frequency",
        parameter_class=ManualParameter,
    )
    instr.add_parameter(
        "t", initial_value=1, unit="s", label="Time", parameter_class=ManualParameter
    )
    instr.add_parameter(
        "phi",
        initial_value=0,
        unit="Rad",
        label="Phase",
        parameter_class=ManualParameter,
    )
    instr.add_parameter(
        "noise_level",
        initial_value=0.05,
        unit="V",
        label="Noise level",
        parameter_class=ManualParameter,
    )
    instr.add_parameter(
        "acq_delay", initial_value=0.02, unit="s", parameter_class=ManualParameter
    )

    def cosine_model():
        sleep(instr.acq_delay())  # simulates the acquisition delay of an instrument
        return (
            cos_func(instr.t(), instr.freq(), instr.amp(), phase=instr.phi(), offset=0)
            + np.random.randn() * instr.noise_level()
        )

    # Wrap our function in a Parameter to be able to associate metadata to it, e.g. unit
    instr.add_parameter(
        name="sig", label="Signal level", unit="V", get_cmd=cosine_model
    )

    return instr


# ######################################################################################
# IQ-related data manipulation and plotting
# ######################################################################################


def mk_iq_shots(
    num_shots: int = 128,
    sigmas: Union[Tuple[float], NDArray[np.float64]] = (0.1, 0.1),
    centers: Union[Tuple[complex], NDArray[np.complex128]] = (-0.2 + 0.65j, 0.7 + 4j),
    probabilities: Union[Tuple[float], NDArray[np.float64]] = (0.4, 0.6),
    seed: Union[int, None] = 112233,
) -> NDArray:
    """
    Generate clusters of (I + 1j*Q) points with a Gaussian distribution.

    Utility to mock the data coming from qubit readout experiments.
    Clusters are centered around ``centers`` and data points are distributed between
    them according to ``probabilities``.

    .. seealso:: :ref:`howto-utilities-examples-ssro`

    Parameters
    ----------
    num_shots
        The number of shot to generate.
    sigma
        The sigma of the Gaussian distribution used for both real and imaginary parts.
    centers
        The center of each cluster on the imaginary plane.
    probabilities
        The probabilities of each cluster being randomly selected for each shot.
    seed
        Random number generator seed passed to ``numpy.random.default_rng``.
    """
    if not len(sigmas) == len(centers) == len(probabilities):
        raise ValueError(
            f"Incorrect input. sigmas={sigmas}, centers={centers} and "
            f"probabilities={probabilities} must have the same length."
        )

    rng = np.random.default_rng(seed=seed)

    cluster_indices = tuple(range(len(centers)))
    choices = rng.choice(a=cluster_indices, size=num_shots, p=probabilities)

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


def mk_trace_time(sampling_rate: float = 1e9, duration: float = 0.3e-6) -> NDArray:
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
    time_values: Optional[NDArray] = None,
    intermediate_freq: float = 50e6,
) -> NDArray:
    """
    Generates mock "traces" that a physical instrument would digitize for the readout of
    a transmon qubit when using a down-converting IQ mixer.

    .. seealso:: :ref:`howto-utilities-examples-trace`

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
    """  # pylint: disable=line-too-long
    if time_values is None:
        time_values = mk_trace_time()
    return iq_point * np.exp(2.0j * np.pi * intermediate_freq * time_values)


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
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QDatasetAttrs(
        tuid=tuid() if callable(tuid) else tuid,
    ).to_dict()
    attrs.update(kwargs)

    return attrs


def mk_main_coord_attrs(
    uniformly_spaced: bool = True,
    is_main_coord: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    A factory of attributes for main coordinates.

    See :class:`~quantify_core.data.dataset_attrs.QCoordAttrs` for details.

    Parameters
    ----------
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.uniformly_spaced`.
    is_main_coord
        See :attr:`quantify_core.data.dataset_attrs.QCoordAttrs.is_main_coord`.
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QCoordAttrs(
        uniformly_spaced=uniformly_spaced,
        is_main_coord=is_main_coord,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_secondary_coord_attrs(
    uniformly_spaced: bool = True,
    is_main_coord: bool = False,
    **kwargs,
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
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QCoordAttrs(
        uniformly_spaced=uniformly_spaced,
        is_main_coord=is_main_coord,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_main_var_attrs(
    grid: bool = True,
    uniformly_spaced: bool = True,
    is_main_var: bool = True,
    has_repetitions: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    A factory of attributes for main variables.

    See :class:`~quantify_core.data.dataset_attrs.QVarAttrs` for details.

    Parameters
    ----------
    grid
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.grid`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.uniformly_spaced`.
    is_main_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_main_var`.
    has_repetitions
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.has_repetitions`.
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QVarAttrs(
        grid=grid,
        uniformly_spaced=uniformly_spaced,
        is_main_var=is_main_var,
        has_repetitions=has_repetitions,
    ).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_secondary_var_attrs(
    grid: bool = True,
    uniformly_spaced: bool = True,
    is_main_var: bool = False,
    has_repetitions: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    A factory of attributes for secondary variables.

    See :class:`~quantify_core.data.dataset_attrs.QVarAttrs` for details.

    Parameters
    ----------
    grid
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.grid`.
    uniformly_spaced
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.uniformly_spaced`.
    is_main_var
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.is_main_var`.
    has_repetitions
        See :attr:`quantify_core.data.dataset_attrs.QVarAttrs.has_repetitions`.
    **kwargs
        Any other items used to update the output dictionary.
    """
    attrs = dd.QVarAttrs(
        grid=grid,
        uniformly_spaced=uniformly_spaced,
        is_main_var=is_main_var,
        has_repetitions=has_repetitions,
    ).to_dict()

    attrs.update(kwargs)
    return attrs


def round_trip_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Writes a dataset to disk and loads it back returning it."""

    tuid = dataset.tuid
    assert tuid
    dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, dataset)
    return dh.load_dataset(tuid)
