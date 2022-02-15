# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""
Factories of exemplary and mock datasets to be used for testing and documentation.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import xarray as xr

import quantify_core.data.dataset_attrs as dattrs
import quantify_core.data.handling as dh
import quantify_core.measurement.control as mc
from quantify_core.analysis.calibration import rotate_to_calibrated_axis
from quantify_core.analysis.fitting_models import exp_decay_func
from quantify_core.measurement.control import grid_setpoints
from quantify_core.utilities.examples_support import (
    mk_dataset_attrs,
    mk_iq_shots,
    mk_main_coord_attrs,
    mk_main_var_attrs,
    mk_secondary_coord_attrs,
    mk_secondary_var_attrs,
    mk_trace_for_iq_shot,
    mk_trace_time,
)


def mk_2d_dataset_v1(num_amps: int = 10, num_times: int = 100):
    """Generates a 2D Quantify dataset (v1).

    Parameters
    ----------
    num_amps
        Number of x points.
    num_times
        Number of y points.
    """
    amps, times = np.linspace(-1, 1, num_amps), np.linspace(0, 10, num_times)
    amps, times = grid_setpoints([amps, times]).T
    sig = amps * np.cos(times)

    attrs = dict(
        tuid=dh.gen_tuid(),
        name="my experiment",
        grid_2d=True,
        grid_2d_uniformly_spaced=True,
        xlen=num_amps,
        ylen=num_times,
    )

    dataset = xr.Dataset(
        coords=dict(
            x0=("dim_0", amps, dict(name="amp", long_name="Amplitude", units="V")),
            x1=("dim_0", times, dict(name="t", long_name="Time", units="s")),
        ),
        data_vars=dict(
            y0=("dim_0", sig, dict(name="sig", long_name="Signal", units="V"))
        ),
        attrs=attrs,
    )
    return dataset


def mk_two_qubit_chevron_data(rep_num: int = 5, seed: Optional[int] = 112233):
    """
    Generates data that look similar to a two-qubit Chevron experiment.

    Parameters
    ----------
    rep_num
        The number of repetitions with noise to generate.
    seed
        Random number generator seed passed to ``numpy.random.default_rng``.

    Returns
    -------
    amp_values
        Amplitude values.
    time_values
        Time values.
    population_q0
        Q0 population values.
    population_q1
        Q1 population values.
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
    """Generates multiple shots for a list of probabilities assuming two states.

    Parameters
    ----------
    probabilities
        The list/array of the probabilities of one of the states.
    **kwargs
        Keyword arguments passed to
        :func:`~quantify_core.utilities.examples_support.mk_iq_shots`.

    Returns
    -------
    :
        Array containing the shots. Shape: (num_shots, len(probabilities)).
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
    pop_q0_attrs = mk_main_var_attrs(
        long_name="Population Q0", unit="", has_repetitions=True
    )
    pop_q1_attrs = mk_main_var_attrs(
        long_name="Population Q1", unit="", has_repetitions=True
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

    dataset_attrs = mk_dataset_attrs()
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset_attrs)

    return dataset


def mk_t1_av_dataset(
    t1_times: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Generates a dataset with mock data of a T1 experiment for a single qubit.

    Parameters
    ----------
    t1_times
        Array with the T1 times corresponding to each probability in ``probabilities``.
    probabilities
        The probabilities of finding the qubit in the excited state.
    **kwargs
        Keyword arguments passed to
        :func:`~quantify_core.utilities.examples_support.mk_iq_shots`.
    """
    if t1_times is None:
        t1_times = np.linspace(0, 120e-6, 30)

    if probabilities is None:
        probabilities = exp_decay_func(
            t=t1_times, tau=50e-6, offset=0, n_factor=1, amplitude=1
        )

    q0_iq_av = mk_shots_from_probabilities(probabilities, **kwargs).mean(axis=0)

    main_dims = ("main_dim",)
    q0_attrs = mk_main_var_attrs(unit="V", long_name="Q0 IQ amplitude")
    t1_time_attrs = mk_main_coord_attrs(unit="s", long_name="T1 Time")

    data_vars = dict(q0_iq_av=(main_dims, q0_iq_av, q0_attrs))
    coords = dict(t1_time=(main_dims, t1_times, t1_time_attrs))

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=mk_dataset_attrs(),
    )
    return dataset


def mk_t1_av_with_cal_dataset(
    t1_times: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Generates a dataset with mock data of a T1 experiment for a single qubit including
    calibration points for the ground and excited states.

    Parameters
    ----------
    t1_times
        Array with the T1 times corresponding to each probability in ``probabilities``.
    probabilities
        The probabilities of finding the qubit in the excited state.
    **kwargs
        Keyword arguments passed to
        :func:`~quantify_core.utilities.examples_support.mk_iq_shots`.
    """
    # reuse previous dataset
    dataset_av = mk_t1_av_dataset(t1_times, probabilities, **kwargs)

    # generate mock calibration data for the ground and excited states
    q0_iq_av_cal = mk_shots_from_probabilities([0, 1], **kwargs).mean(axis=0)

    secondary_dims = ("cal_dim",)
    q0_cal_attrs = mk_secondary_var_attrs(unit="V", long_name="Q0 IQ Calibration")
    cal_attrs = mk_secondary_coord_attrs(unit="", long_name="Q0 state")

    relationships = [
        dattrs.QDatasetIntraRelationship(
            item_name=dataset_av.q0_iq_av.name,  # name of a variable in the dataset
            relation_type="calibration",
            related_names=["q0_iq_av_cal"],  # the secondary variable in the dataset
        ).to_dict()
    ]

    data_vars = dict(
        q0_iq_av=dataset_av.q0_iq_av,  # reuse from the other dataset
        q0_iq_av_cal=(secondary_dims, q0_iq_av_cal, q0_cal_attrs),
    )
    coords = dict(
        t1_time=dataset_av.t1_time,  # reuse from the other dataset
        cal=(secondary_dims, ["|0>", "|1>"], cal_attrs),  # coords can be strings
    )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=mk_dataset_attrs(relationships=relationships),  # relationships added here
    )

    return dataset


def mk_t1_shots_dataset(
    t1_times: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Generates a dataset with mock data of a T1 experiment for a single qubit including
    calibration points for the ground and excited states, including all the individual
    shots (repeated qubit state measurement for the same exact experiment).

    Parameters
    ----------
    t1_times
        Array with the T1 times corresponding to each probability in ``probabilities``.
    probabilities
        The probabilities of finding the qubit in the excited state.
    **kwargs
        Keyword arguments passed to
        :func:`~quantify_core.utilities.examples_support.mk_iq_shots`.
    """
    # reuse previous dataset
    dataset_av_with_cal = mk_t1_av_with_cal_dataset(t1_times, probabilities, **kwargs)
    if probabilities is None:
        probabilities = dataset_av_with_cal.q0_iq_av.values
        probabilities = rotate_to_calibrated_axis(
            probabilities, *dataset_av_with_cal.q0_iq_av_cal.values
        ).real
    # generate mock data containing all the shots,
    # NB not the same data that was used for the average above, but this is just a mock
    q0_iq_shots = mk_shots_from_probabilities(probabilities, **kwargs)
    q0_iq_shots_cal = mk_shots_from_probabilities([0, 1], **kwargs)

    # the xarray dimensions will now require an outer repetitions dimension
    secondary_dims_rep = ("repetitions", "cal_dim")
    main_dims_rep = ("repetitions", "main_dim")

    relationships = [
        dattrs.QDatasetIntraRelationship(
            item_name=dataset_av_with_cal.q0_iq_av.name,
            relation_type="calibration",
            related_names=[dataset_av_with_cal.q0_iq_av_cal.name],
        ).to_dict(),
        dattrs.QDatasetIntraRelationship(
            item_name="q0_iq_shots",
            relation_type="calibration",
            related_names=["q0_iq_cal_shots"],
        ).to_dict(),
        # suggestion of a custom relationship
        dattrs.QDatasetIntraRelationship(
            item_name=dataset_av_with_cal.q0_iq_av.name,
            relation_type="individual_shots",
            related_names=["q0_iq_shots"],
        ).to_dict(),
    ]

    # Flag that these variables use a repetitions dimension
    q0_attrs_rep = dict(dataset_av_with_cal.q0_iq_av.attrs)
    q0_attrs_rep["has_repetitions"] = True
    q0_cal_attrs_rep = dict(dataset_av_with_cal.q0_iq_av_cal.attrs)
    q0_cal_attrs_rep["has_repetitions"] = True

    data_vars = dict(
        # variables that are the same as in the previous dataset, and are now redundant,
        # however, we include them to showcase the dataset flexibility
        q0_iq_av=dataset_av_with_cal.q0_iq_av,
        q0_iq_av_cal=dataset_av_with_cal.q0_iq_av_cal,
        # variables that contain all the individual shots
        q0_iq_shots=(main_dims_rep, q0_iq_shots, q0_attrs_rep),
        q0_iq_shots_cal=(secondary_dims_rep, q0_iq_shots_cal, q0_cal_attrs_rep),
    )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=dataset_av_with_cal.coords,  # same coordinates as in previous dataset
        attrs=mk_dataset_attrs(relationships=relationships),  # relationships added here
    )

    return dataset


def mk_t1_traces_dataset(
    t1_times: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Generates a dataset with mock data of a T1 experiment for a single qubit including
    calibration points for the ground and excited states, including all the individual
    shots (repeated qubit state measurement for the same exact experiment); and
    including all the signals that had to be digitized to obtain the rest of the data.

    Parameters
    ----------
    t1_times
        Array with the T1 times corresponding to each probability in ``probabilities``.
    probabilities
        The probabilities of finding the qubit in the excited state.
    **kwargs
        Keyword arguments passed to
        :func:`~quantify_core.utilities.examples_support.mk_iq_shots`.
    """
    dataset_shots = mk_t1_shots_dataset(t1_times, probabilities, **kwargs)
    shots = dataset_shots.q0_iq_shots.values
    shots_cal = dataset_shots.q0_iq_shots_cal.values

    # generate mock traces for all shots
    q0_traces = np.array(tuple(map(mk_trace_for_iq_shot, shots.flatten())))
    q0_traces = q0_traces.reshape(*shots.shape, q0_traces.shape[-1])
    # generate mock traces for calibration points shots
    q0_traces_cal = np.array(tuple(map(mk_trace_for_iq_shot, shots_cal.flatten())))
    q0_traces_cal = q0_traces_cal.reshape(*shots_cal.shape, q0_traces_cal.shape[-1])

    traces_dims = ("repetitions", "main_dim", "trace_dim")
    traces_cal_dims = ("repetitions", "cal_dim", "trace_dim")
    trace_times = mk_trace_time()
    trace_attrs = mk_main_coord_attrs(long_name="Trace time", unit="s")

    relationships_with_traces = dataset_shots.relationships + [
        dattrs.QDatasetIntraRelationship(
            item_name="q0_traces",
            related_names=["q0_traces_cal"],
            relation_type="calibration",
        ).to_dict(),
    ]

    data_vars = dict(
        q0_iq_av=dataset_shots.q0_iq_av,
        q0_iq_av_cal=dataset_shots.q0_iq_av_cal,
        q0_iq_shots=dataset_shots.q0_iq_shots,
        q0_iq_shots_cal=dataset_shots.q0_iq_shots_cal,
        q0_traces=(traces_dims, q0_traces, dataset_shots.q0_iq_shots.attrs),
        q0_traces_cal=(
            traces_cal_dims,
            q0_traces_cal,
            dataset_shots.q0_iq_shots_cal.attrs,
        ),
    )
    coords = dict(
        t1_time=dataset_shots.t1_time,
        cal=dataset_shots.cal,
        trace_time=(("trace_dim",), trace_times, trace_attrs),
    )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=mk_dataset_attrs(relationships=relationships_with_traces),
    )

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
            mk_main_var_attrs(
                unit="V", long_name=f"IQ amplitude {qubit}", has_repetitions=True
            ),
        )

    for qubit in (f"D{i}" for i in range(4)):
        data_vars[f"{qubit}_shots"] = (
            ("repetitions", "dim_final"),
            mock_data_final,
            mk_main_var_attrs(
                unit="V", long_name=f"IQ amplitude {qubit}", has_repetitions=True
            ),
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
        attrs=mk_dataset_attrs(),
    )

    return dataset


# pylint: disable=too-many-arguments
def mk_nested_mc_dataset(
    num_points: int = 12,
    flux_bias_min_max: tuple = (-0.04, 0.04),
    resonator_freqs_min_max: tuple = (7e9, 7.3e9),
    qubit_freqs_min_max: tuple = (4.5e9, 5.0e9),
    t1_values_min_max: tuple = (20e-6, 50e-6),
    seed: Optional[int] = 112233,
) -> xr.Dataset:
    """
    Generates a dataset with dataset references and several coordinates that serve to
    index the same variables.

    Note that the each value for ``resonator_freqs``, ``qubit_freqs`` and ``t1_values``
    would have been extracted from other dataset corresponding to individual experiments
    with their own dataset.

    Parameters
    ----------
    num_points
        Number of datapoints to generate (used for all variables/coordinates).
    flux_bias_min_max
        Range for mock values.
    resonator_freqs_min_max
        Range for mock values.
    qubit_freqs_min_max
        Range for mock values.
    t1_values_min_max
        Range for mock random values.
    seed
        Random number generator seed passed to ``numpy.random.default_rng``.
    """
    rng = np.random.default_rng(seed=seed)  # random number generator

    flux_bias_vals = np.linspace(*flux_bias_min_max, num_points)
    resonator_freqs = np.linspace(*resonator_freqs_min_max, num_points)
    qubit_freqs = np.linspace(*qubit_freqs_min_max, num_points)
    t1_values = rng.uniform(*t1_values_min_max, num_points)

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
            mk_main_coord_attrs(
                long_name="Dataset TUID resonator frequency", is_dataset_ref=True
            ),
        ),
        qubit_freq_tuids=(
            "main_dim",
            qubit_freq_tuids,
            mk_main_coord_attrs(
                long_name="Dataset TUID qubit frequency", is_dataset_ref=True
            ),
        ),
        t1_tuids=(
            "main_dim",
            t1_tuids,
            mk_main_coord_attrs(long_name="Dataset TUID T1", is_dataset_ref=True),
        ),
    )

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
