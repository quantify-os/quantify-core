# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 1
# %aimport quantify_core.data.dataset_attrs
# %aimport quantify_core.data.dataset_adapters
# %aimport quantify_core.utilities.examples_support

# %% [raw]
# .. admonition:: TODO
#
#     Write supporting text.

# %% [raw]
# Quantify dataset - advanced examples
# ====================================
#
# .. seealso::
#
#     The complete source code of this tutorial can be found in
#
#     :jupyter-download:notebook:`Quantify dataset - advanced examples`
#
#     :jupyter-download:script:`Quantify dataset - advanced examples`

# %% [raw]
# .. admonition:: Imports and auxiliary utilities
#     :class: dropdown

# %%
# rst-json-conf: {"indent": "    ", "jupyter_execute_options": [":hide-output:"]}

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from quantify_core.data import handling as dh
from quantify_core.measurement import grid_setpoints
from qcodes import ManualParameter
from rich import pretty
from pathlib import Path
from quantify_core.data.handling import get_datadir, set_datadir
import quantify_core.data.dataset_attrs as dd
from quantify_core.utilities.examples_support import (
    mk_dataset_attrs,
    mk_exp_coord_attrs,
    mk_exp_var_attrs,
    round_trip_dataset,
    par_to_attrs,
)

from typing import List, Tuple

pretty.install()

set_datadir(Path.home() / "quantify-data")  # change me!


def generate_mock_iq_data(
    n_shots, sigma=0.3, center0=(1, 1), center1=(1, -1), prob=0.5
):
    """
    Generates two clusters of I,Q points with a Gaussian distribution.
    """
    i_data = np.zeros(n_shots)
    q_data = np.zeros(n_shots)
    for i in range(n_shots):
        c = center0 if (np.random.rand() >= prob) else center1
        i_data[i] = np.random.normal(c[0], sigma)
        q_data[i] = np.random.normal(c[1], sigma)
    return i_data + 1j * q_data


def generate_exp_decay_probablity(time: np.ndarray, tau: float):
    return np.exp(-time / tau)


def generate_trace_time(sampling_rate: float = 1e9, trace_duratation: float = 0.3e-6):
    trace_length = sampling_rate * trace_duratation
    return np.arange(0, trace_length, 1) / sampling_rate


def generate_trace_for_iq_point(
    iq_amp: complex,
    tbase: np.ndarray = generate_trace_time(),
    intermediate_freq: float = 50e6,
) -> tuple:
    """
    Generates mock traces that a physical instrument would digitize for the readout of
    a transmon qubit.
    """

    return iq_amp * np.exp(2.0j * np.pi * intermediate_freq * tbase)


def plot_centroids(ax, ground, excited):
    ax.plot(
        [ground[0]],
        [ground[1]],
        label="|0>",
        marker="o",
        color="C3",
        markersize=10,
    )
    ax.plot(
        [excited[0]],
        [excited[1]],
        label="|1>",
        marker="^",
        color="C4",
        markersize=10,
    )


# %% [raw]
# An "unstructured" experiment and dataset example
# ------------------------------------------------

# %% [raw]
# Schdule reference: `one of the latest papers from DiCarlo Lab <https://arxiv.org/abs/2102.13071>`_, Fig. 4b.
#
# NB not exactly the same schedule, but what matter are the measurements.

# %%
from quantify_scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib
from quantify_scheduler import Schedule
from quantify_scheduler.gate_library import Reset, Measure, CZ, Rxy, X90, X, Y, Y90, X90

d1, d2, d3, d4 = [f"D{i}" for i in range(1, 5)]
a1, a2, a3 = [f"A{i}" for i in range(1, 4)]

all_qubits = d1, d2, d3, d4, a1, a2, a3

sched = Schedule(f"S7 dance")

sched.add(Reset(*all_qubits))

num_cycles = 4

for cycle in range(num_cycles):
    sched.add(Y90(d1))
    for q in [d2, d3, d4]:
        sched.add(Y90(q), ref_pt="start", rel_time=0)
    sched.add(Y90(a2), ref_pt="start", rel_time=0)

    for q in [d2, d1, d4, d3]:
        sched.add(CZ(qC=q, qT=a2))

    sched.add(Y90(d1))
    for q in [d2, d3, d4]:
        sched.add(Y90(q), ref_pt="start", rel_time=0)
    sched.add(Y90(a2), ref_pt="start", rel_time=0)

    sched.add(Y90(a1), ref_pt="end", rel_time=0)
    sched.add(Y90(a3), ref_pt="start", rel_time=0)

    sched.add(CZ(qC=d1, qT=a1))
    sched.add(CZ(qC=d2, qT=a3))
    sched.add(CZ(qC=d3, qT=a1))
    sched.add(CZ(qC=d4, qT=a3))

    sched.add(Y90(a1), ref_pt="end", rel_time=0)
    sched.add(Y90(a3), ref_pt="start", rel_time=0)

    sched.add(Measure(a2, acq_index=cycle))
    for q in (a1, a3):
        sched.add(Measure(q, acq_index=cycle), ref_pt="start", rel_time=0)

    for q in [d1, d2, d3, d4]:
        sched.add(X(q), ref_pt="start", rel_time=0)

# final measurements

sched.add(Measure(*all_qubits[:4], acq_index=0), ref_pt="end", rel_time=0)

f, ax = circuit_diagram_matplotlib(sched)
# f.set_figheight(10)
f.set_figwidth(30)

# %% [raw]
# How do we store all shots for this measurement? (we want it because, e.g., we know we have issue with leakage to the second excited state)

# %%
num_shots = 128
center_ground = (-0.2, 0.65)
center_excited = (0.7, -0, 4)
sigma = 0.1

cycles = range(num_cycles)

radom_data = np.array(
    tuple(
        generate_mock_iq_data(
            n_shots=num_shots,
            sigma=sigma,
            center0=center_ground,
            center1=center_excited,
            prob=prob,
        )
        for prob in [np.random.random() for _ in cycles]
    )
).T

radom_data_final = np.array(
    tuple(
        generate_mock_iq_data(
            n_shots=num_shots,
            sigma=sigma,
            center0=center_ground,
            center1=center_excited,
            prob=prob,
        )
        for prob in [np.random.random()]
    )
).T

# NB same random data is used for all qubits only for the simplicity of the mock!

data_vars = {}

for q in (a1, a2, a3):
    data_vars[f"{q}_shots"] = (
        ("repetitions", "dim_cycles"),
        radom_data,
        mk_exp_var_attrs(
            units="V", long_name=f"IQ amplitude {q}", experiment_coords=["cycles"]
        ),
    )

for q in (d1, d2, d3, d4):
    data_vars[f"{q}_shots"] = (
        ("repetitions", "dim_final"),
        radom_data_final,
        mk_exp_var_attrs(
            units="V", long_name=f"IQ amplitude {q}", experiment_coords=["final_msmt"]
        ),
    )

dataset = xr.Dataset(
    data_vars=data_vars,
    coords={
        "cycle": (
            "dim_cycles",
            cycles,
            mk_exp_coord_attrs(units="", long_name="Surface code cycle number"),
        ),
        "final_msmt": (
            "dim_final",
            [0],
            mk_exp_coord_attrs(units="", long_name="Final measurement"),
        ),
    },
    attrs=mk_dataset_attrs(repetitions_dims=["repetitions"]),
)


assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset.A1_shots.shape

# %%
dataset.D1_shots.shape

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="dim_cycle", coords_names=["cycle"]
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded, dimension="dim_final", coords_names=["final_msmt"]
)
dataset_gridded

# %% [raw]
# "Nested MeasurementControl" example
# -----------------------------------

# %%
flux_bias_values = np.linspace(-0.04, 0.04, 12)

resonator_frequencies = np.linspace(7e9, 8.5e9, len(flux_bias_values))
qubit_frequencies = np.linspace(4.5e9, 4.6e9, len(flux_bias_values))
t1_values = np.linspace(20e-6, 50e-6, len(flux_bias_values))

resonator_freq_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]
qubit_freq_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]
t1_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]

# %%
dataset = xr.Dataset(
    data_vars={
        "resonator_freq": (
            "dim_0",
            resonator_frequencies,
            mk_exp_var_attrs(
                long_name="Resonator frequency",
                units="Hz",
                experiment_coords=["flux_bias"],
            ),
        ),
        "qubit_freq": (
            "dim_0",
            qubit_frequencies,
            mk_exp_var_attrs(
                long_name="Qubit frequency", units="Hz", experiment_coords=["flux_bias"]
            ),
        ),
        "t1": (
            "dim_0",
            t1_values,
            mk_exp_var_attrs(
                long_name="T1", units="s", experiment_coords=["flux_bias"]
            ),
        ),
    },
    coords={
        "flux_bias": (
            "dim_0",
            flux_bias_values,
            mk_exp_coord_attrs(long_name="Flux bias", units="A"),
        ),
        "resonator_freq_tuids": (
            "dim_0",
            resonator_freq_tuids,
            mk_exp_coord_attrs(long_name="Dataset TUID", units="", is_dataset_ref=True),
        ),
        "qubit_freq_tuids": (
            "dim_0",
            qubit_freq_tuids,
            mk_exp_coord_attrs(long_name="Dataset TUID", units="", is_dataset_ref=True),
        ),
        "t1_tuids": (
            "dim_0",
            t1_tuids,
            mk_exp_coord_attrs(long_name="Dataset TUID", units="", is_dataset_ref=True),
        ),
    },
    attrs=mk_dataset_attrs(
        experiment_coords=[
            ("flux_bias", "resonator_freq_tuids", "qubit_freq_tuids", "t1_tuids")
        ],
        experiment_vars=[
            "resonator_freq",
            "qubit_freq",
            "t1",
        ],
    ),
)

assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
coords_for_multi_index = dd.get_experiment_coords(dataset)
coords_for_multi_index

# %% [raw]
# In this case the four experiment coordinates are not orthogonal coordinates, but instead just different label for the same datapoints, also known as a "multi-index". It is possible to work with an explicit MultiIndex within a (python) xarray object:

# %%
dataset_multi_indexed = dataset.set_index({"dim_0": coords_for_multi_index})

dataset_multi_indexed

# %% [raw]
# The MultiIndex is very handy for selecting data in different ways, e.g.:

# %%
dataset_multi_indexed.qubit_freq.sel(resonator_freq_tuids=resonator_freq_tuids[2])

# %%
dataset_multi_indexed.qubit_freq.sel(t1_tuids=t1_tuids[2])

# %% [raw]
# Known limiations
# ^^^^^^^^^^^^^^^^

# %% [raw]
# But at the moment has the problem of being incompatible with the NetCDF format used to write to disk:

# %%
try:
    assert dataset_multi_indexed == round_trip_dataset(
        dataset_multi_indexed
    )  # confirm read/write
except NotImplementedError as exp:
    print(exp)

# %% [raw]
# We could make our load/write utilities to take care of setting and resetting the index under the hood. Though there are some nuances there as well. If we would do that then some extra metadata needs to be stored in order to store/restore the multi-index. At the moment the MultiIndex is not supported yet when writing a Quantify dataset to disk. Below are a few examples of potential complications.

# %% [raw]
# Fortunetly, the MultiIndex can be reset back:

# %%
dataset_multi_indexed.reset_index(dims_or_levels="dim_0")

# %%
all(dataset_multi_indexed.reset_index("dim_0").t1_tuids == dataset.t1_tuids)

# %% [raw]
# But, for example, the ``dtype`` has been changed to ``object`` (from fixed-length string):

# %%
dataset.t1_tuids.dtype, dataset_multi_indexed.reset_index("dim_0").t1_tuids.dtype

# %%
dataset.t1_tuids.dtype == dataset_multi_indexed.reset_index("dim_0").t1_tuids.dtype
