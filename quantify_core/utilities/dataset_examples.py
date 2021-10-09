# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Factories of exemplary and mock datasets to be used for testing and documentation."""

from __future__ import annotations

from typing import Union
import numpy as np
import xarray as xr
import quantify_core.data.handling as dh
import quantify_core.measurement.control as mc
from quantify_core.utilities.examples_support import (
    mk_dataset_attrs,
    mk_main_var_attrs,
    mk_main_coord_attrs,
    mk_iq_shots,
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


def mk_shots_from_probabilities(probabilities: Union[np.ndarray, list], **kwargs):
    """Generates multiple shots for a list of probabilities.

    Parameters
    ----------
    probabilities
        The list/array of the probabilities of one the states.

    Returns
    -------
    :
        Array containing the shots. Shape: (n_shots, len(probabilities)).
    """

    shots = np.array(
        tuple(
            mk_iq_shots(probabilities=[prob, 1 - prob], **kwargs)
            for prob in probabilities
        )
    ).T

    return shots


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
    pop_q0_attrs = mk_main_var_attrs(long_name="Population Q0", unit="")
    pop_q1_attrs = mk_main_var_attrs(long_name="Population Q1", unit="")
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


def mk_surface7_cyles_dataset(num_cycles: int = 3, **kwargs) -> xr.Dataset:
    """
    See also :func:`quantify_core.utilities.examples_support.mk_surface7_sched`.

    Parameters
    ----------
    num_cycles
        The number of repeating cycles before the final measurement.
    **kwargs
        Keyword arguments passed to :func:`~.mk_shots_from_probabilities`.
    """

    cycles = range(num_cycles)

    mock_data = mk_shots_from_probabilities(
        probabilities=[np.random.random() for _ in cycles], **kwargs
    )

    mock_data_final = mk_shots_from_probabilities(
        probabilities=[np.random.random()], **kwargs
    )

    # %%
    data_vars = {}

    # NB same random data is used for all qubits only for the simplicity of the mock!
    for qubit in (f"A{i}" for i in range(3)):
        data_vars[f"{qubit}_shots"] = (
            ("repetitions", "dim_cycle"),
            mock_data,
            mk_main_var_attrs(unit="V", long_name=f"IQ amplitude {qubit}"),
        )

    for qubit in (f"D{i}" for i in range(4)):
        data_vars[f"{qubit}_shots"] = (
            ("repetitions", "dim_final"),
            mock_data_final,
            mk_main_var_attrs(unit="V", long_name=f"IQ amplitude {qubit}"),
        )

    cycle_attrs = mk_main_coord_attrs(long_name="Surface code cycle number")
    final_msmt_attrs = mk_main_coord_attrs(long_name="Final measurement")
    coords = dict(
        cycle=("dim_cycle", cycles, cycle_attrs),
        final_msmt=("dim_final", [0], final_msmt_attrs),
    )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=mk_dataset_attrs(repetitions_dims=["repetitions"]),
    )

    return dataset


def mk_nested_mc_dataset(
    num_points: int = 12,
    flux_bias_min_max: tuple = (-0.04, 0.04),
    resonator_freqs_min_max: tuple = (7e9, 8.5e9),
    qubit_freqs_min_max: tuple = (4.5e9, 4.6e9),
    t1_values_min_max: tuple = (20e-6, 50e-6),
) -> xr.Dataset:
    """
    Generates a dataset with dataset references and several coordinates that serve to
    index the same variable(s).

    Parameters
    ----------
    num_points
        Number of datapoints to generate (used for all variables/coordinates).
    flux_bias_min_max
    resonator_freqs_min_max
    qubit_freqs_min_max
    t1_values_min_max
    """

    flux_bias_vals = np.linspace(*flux_bias_min_max, num_points)
    resonator_freqs = np.linspace(*resonator_freqs_min_max, num_points)
    qubit_freqs = np.linspace(*qubit_freqs_min_max, num_points)
    t1_values = np.linspace(*t1_values_min_max, num_points)

    resonator_freq_tuids = [dh.gen_tuid() for _ in range(num_points)]
    qubit_freq_tuids = [dh.gen_tuid() for _ in range(num_points)]
    t1_tuids = [dh.gen_tuid() for _ in range(num_points)]

    coords = dict(
        flux_bias=(
            "main_dim",
            flux_bias_vals,
            mk_main_coord_attrs(long_name="Flux bias", unit="A"),
        ),
        resonator_freq_tuids=(
            "main_dim",
            resonator_freq_tuids,
            mk_main_coord_attrs(long_name="Dataset TUID", is_dataset_ref=True),
        ),
        qubit_freq_tuids=(
            "main_dim",
            qubit_freq_tuids,
            mk_main_coord_attrs(long_name="Dataset TUID", is_dataset_ref=True),
        ),
        t1_tuids=(
            "main_dim",
            t1_tuids,
            mk_main_coord_attrs(long_name="Dataset TUID", is_dataset_ref=True),
        ),
    )

    # A tuple instead of a single str will indicate that these coordinates can used as
    # a Multindex
    vars_coords = tuple(coords.keys())

    data_vars = dict(
        resonator_freq=(
            "main_dim",
            resonator_freqs,
            mk_main_var_attrs(long_name="Resonator frequency", unit="Hz"),
        ),
        qubit_freq=(
            "main_dim",
            qubit_freqs,
            mk_main_var_attrs(long_name="Qubit frequency", unit="Hz"),
        ),
        t1=(
            "main_dim",
            t1_values,
            mk_main_var_attrs(long_name="T1", unit="s"),
        ),
    )
    dataset_attrs = mk_dataset_attrs()

    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset_attrs)

    return dataset
