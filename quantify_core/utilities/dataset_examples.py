# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Factories of exemplary and mock datasets to be used for testing and documentation."""

from __future__ import annotations

from typing import Union
import numpy as np
import xarray as xr
import quantify_core.measurement.control as mc
from quantify_core.utilities.examples_support import (
    mk_dataset_attrs,
    mk_main_var_attrs,
    mk_main_coord_attrs,
)


def mk_two_qubit_chevron_data(rep_num: int = 5, seed: Union[int, None] = 112233):
    """
    Generates data that look similar to a two-qubit Chevron experiment.

    Parameters
    ----------
    rep_num
        The number of repetitions with noise to generate.
    seed
        Random number generator passed to ``numpy.random.default_rng``.

    Returns
    -------
    amp_values

    time_values

    population_q0

    population_q1
    """

    rng = np.random.default_rng(seed=seed)  # random number generator

    amp_values = np.linspace(0.45, 0.55, 30)
    time_values = np.linspace(0, 100e-9, 40)

    amp_values, time_values = mc.grid_setpoints([amp_values, time_values]).T
    amp_values_norm = np.abs(
        (amp_values - amp_values.mean()) / (amp_values - amp_values.mean()).max()
    )
    pop_q0 = (1 - amp_values_norm) * np.sin(
        2 * np.pi * time_values * 1 / 30e-9 * (amp_values_norm + 0.5)
    )
    pop_q1 = -pop_q0  # mock inverted population for q1

    # mock repetitions of the same experiment with noise
    pop_q0 = np.array(
        [pop_q0 + pop_q0 * rng.uniform(-0.5, 0.5, pop_q0.shape) for _ in range(rep_num)]
    )
    pop_q1 = np.array(
        [pop_q1 + pop_q1 * rng.uniform(-0.5, 0.5, pop_q1.shape) for _ in range(rep_num)]
    )

    pop_q0 = np.clip(pop_q0 / 2 + 0.5, 0, 1)  # shift to 0-1 range and clip for plotting
    pop_q1 = np.clip(pop_q1 / 2 + 0.5, 0, 1)  # shift to 0-1 range and clip for plotting

    return amp_values, time_values, pop_q0, pop_q1


# pylint: disable=too-many-locals
def mk_two_qubit_chevron_dataset(**kwargs) -> xr.Dataset:
    """
    Generates a dataset that look similar to a two-qubit Chevron experiment.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~.mk_two_qubit_chevron_data`.

    Returns
    -------
    :
        A mock Quantify dataset.
    """
    amp_values, time_values, pop_q0, pop_q1 = mk_two_qubit_chevron_data(**kwargs)

    dims_q0 = dims_q1 = ("repetitions", "main_dim")
    pop_q0_attrs = mk_main_var_attrs(
        long_name="Population Q0", unit="", coords=["amp", "time"]
    )
    pop_q1_attrs = mk_main_var_attrs(
        long_name="Population Q1", unit="", coords=["amp", "time"]
    )
    data_vars = dict(
        pop_q0=(dims_q0, pop_q0, pop_q0_attrs),
        pop_q1=(dims_q1, pop_q1, pop_q1_attrs),
    )

    dims_amp = dims_time = ("main_dim",)
    amp_attrs = mk_main_coord_attrs(long_name="Amplitude", unit="V")
    time_attrs = mk_main_coord_attrs(long_name="Time", unit="s")
    coords = dict(
        amp=(dims_amp, amp_values, amp_attrs),
        time=(dims_time, time_values, time_attrs),
    )

    dataset_attrs = mk_dataset_attrs(
        main_dims=["main_dim"], repetitions_dims=["repetitions"]
    )
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset_attrs)

    return dataset
