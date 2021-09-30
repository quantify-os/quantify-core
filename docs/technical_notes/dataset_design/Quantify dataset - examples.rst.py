# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: py:percent
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
# rst-json-conf: {"jupyter_execute_options": [":hide-code:"]}
# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement


# %%
# %load_ext autoreload
# %autoreload 1
# %aimport quantify_core.data.dataset_attrs
# %aimport quantify_core.data.dataset_adapters
# %aimport quantify_core.utilities.examples_support

# %% [raw]
"""
.. admonition:: TODO

    Write supporting text.
"""

# %% [raw]
"""
.. _sec-quantify-dataset-examples:

Quantify dataset - examples
===========================

.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Quantify dataset - examples`

    :jupyter-download:script:`Quantify dataset - examples`
"""

# %% [raw]
"""
.. admonition:: Imports and auxiliary utilities
    :class: dropdown
"""

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
    mk_main_coord_attrs,
    mk_secondary_coord_attrs,
    mk_main_var_attrs,
    mk_secondary_var_attrs,
    round_trip_dataset,
    par_to_attrs,
)

from typing import List, Tuple

pretty.install()

set_datadir(Path.home() / "quantify-data")  # change me!

# %%
## rst-json-conf: {"indent": "    "}

x0s = np.linspace(0.45, 0.55, 30)
x1s = np.linspace(0, 100e-9, 40)
time_par = ManualParameter(name="time", label="Time", unit="s")
amp_par = ManualParameter(name="amp", label="Flux amplitude", unit="V")
pop_q0_par = ManualParameter(name="pop_q0", label="Population Q0", unit="arb. unit")
pop_q1_par = ManualParameter(name="pop_q1", label="Population Q1", unit="arb. unit")

x0s, x1s = grid_setpoints([x0s, x1s], [amp_par, time_par]).T
x0s_norm = np.abs((x0s - x0s.mean()) / (x0s - x0s.mean()).max())
y0s = (1 - x0s_norm) * np.sin(
    2 * np.pi * x1s * 1 / 30e-9 * (x0s_norm + 0.5)
)  # ~chevron
y1s = -y0s  # mock inverted population for q1

dataset = dataset_2d_example = xr.Dataset(
    data_vars={
        pop_q0_par.name: (
            ("repetitions", "dim_0"),
            [y0s + np.random.random(y0s.shape) / k for k in (100, 10, 5)],
            mk_main_var_attrs(
                **par_to_attrs(pop_q0_par),
                coords=[amp_par.name, time_par.name],
            ),
        ),
        pop_q1_par.name: (
            ("repetitions", "dim_0"),
            [y1s + np.random.random(y1s.shape) / k for k in (100, 10, 5)],
            mk_main_var_attrs(
                **par_to_attrs(pop_q1_par),
                coords=[amp_par.name, time_par.name],
            ),
        ),
    },
    coords={
        amp_par.name: ("dim_0", x0s, mk_main_coord_attrs(**par_to_attrs(amp_par))),
        time_par.name: ("dim_0", x1s, mk_main_coord_attrs(**par_to_attrs(time_par))),
    },
    attrs=mk_dataset_attrs(repetitions_dims=["repetitions"]),
)

assert dataset == round_trip_dataset(dataset)  # confirm read/write
dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset_2d_example,
    dimension=dd.get_main_dims(dataset_2d_example)[0],
    coords_names=dd.get_main_coords(dataset_2d_example),
)
dataset_gridded.pop_q0.plot.pcolormesh(x="amp", col=dataset_gridded.pop_q0.dims[0])
_ = dataset_gridded.pop_q1.plot.pcolormesh(x="amp", col=dataset_gridded.pop_q1.dims[0])

# %% [raw]
"""
In xarray, among other features, it is possible to average along a dimension which can
be very convenient:
"""

# %%
_ = dataset_gridded.pop_q0.mean(dim=dataset_gridded.pop_q0.dims[0]).plot(x="amp")

# %%
## rst-json-conf: {"indent": "    "}

dataset = xr.Dataset(
    data_vars={
        pop_q0_par.name: (
            ("repetitions", "dim_0"),
            [y0s + np.random.random(y0s.shape) / k for k in (100, 10, 5)],
            mk_main_var_attrs(
                **par_to_attrs(pop_q0_par),
                coords=[amp_par.name, time_par.name],
            ),
        ),
        pop_q1_par.name: (
            ("repetitions", "dim_0"),
            [y1s + np.random.random(y1s.shape) / k for k in (100, 10, 5)],
            mk_main_var_attrs(
                **par_to_attrs(pop_q1_par),
                coords=[amp_par.name, time_par.name],
            ),
        ),
    },
    coords={
        amp_par.name: ("dim_0", x0s, mk_main_coord_attrs(**par_to_attrs(amp_par))),
        time_par.name: ("dim_0", x1s, mk_main_coord_attrs(**par_to_attrs(time_par))),
        # here we choose to index the repetition dimension with an array of strings
        "repetitions": (
            "repetitions",
            ["noisy", "very noisy", "very very noisy"],
        ),
    },
    attrs=mk_dataset_attrs(repetitions_dims=["repetitions"]),
)

dataset

# %%
## rst-json-conf: {"indent": "    "}

dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="dim_0", coords_names=dd.get_main_coords(dataset)
)
dataset_gridded

# %%
## rst-json-conf: {"indent": "    "}

dataset_gridded.pop_q0.sel(repetitions="very noisy").plot(x="amp")
pass

# %% [raw]
"""
T1 dataset examples
-------------------
"""

# %% [raw]
"""
.. admonition:: Mock data utilities
    :class: dropdown
"""

# %%
# rst-json-conf: {"indent": "    "}


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


# %%
# rst-json-conf: {"indent": "    "}

center_ground = (-0.2, 0.65)
center_excited = (0.7, -0, 4)

shots = generate_mock_iq_data(
    n_shots=256, sigma=0.1, center0=center_ground, center1=center_excited, prob=0.4
)

# %%
# rst-json-conf: {"indent": "    "}

plt.hexbin(shots.real, shots.imag)
plt.xlabel("I")
plt.ylabel("Q")
plot_centroids(plt.gca(), center_ground, center_excited)

# %%
# rst-json-conf: {"indent": "    "}

time = generate_trace_time()
trace = generate_trace_for_iq_point(shots[0])

fig, ax = plt.subplots(1, 1, figsize=(30, 5))
ax.plot(time, trace.imag, ".-")
_ = ax.plot(time, trace.real, ".-")

# %% [raw]
"""
T1 experiment averaged
~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
# parameters of our qubit model
tau = 30e-6
center_ground = (-0.2, 0.65)
center_excited = (0.7, -0, 4)
sigma = 0.1

# mock of data acquisition configuration
num_shots = 256
x0s = np.linspace(0, 150e-6, 30)
time_par = ManualParameter(name="time", label="Time", unit="s")
q0_iq_par = ManualParameter(name="q0_iq", label="Q0 IQ amplitude", unit="V")

probabilities = generate_exp_decay_probablity(time=x0s, tau=tau)
plt.ylabel("|1> probability")
plt.suptitle("Typical T1 experiment processed data")
_ = plt.plot(x0s, probabilities, ".-")

# %%
y0s = np.fromiter(
    (
        np.average(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=sigma,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
        )
        for prob in probabilities
    ),
    dtype=complex,
)

dataset = xr.Dataset(
    data_vars={
        q0_iq_par.name: (
            "dim_0",
            y0s,
            mk_main_var_attrs(**par_to_attrs(q0_iq_par), coords=[time_par.name]),
        ),
    },
    coords={
        time_par.name: ("dim_0", x0s, mk_main_coord_attrs(**par_to_attrs(time_par))),
    },
    attrs=mk_dataset_attrs(),
)


assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension=dataset.q0_iq.dims[0],
    coords_names=dd.get_main_coords(dataset),
)
dataset_gridded


# %% [raw]
"""
.. admonition:: Plotting utilities
    :class: dropdown
"""

# %%
# rst-json-conf: {"indent": "    "}


def plot_decay_no_repetition(gridded_dataset, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    y0 = gridded_dataset[dd.get_main_vars(gridded_dataset)[0]]
    y0.real.plot(ax=ax, marker=".", label="I data")
    y0.imag.plot(ax=ax, marker=".", label="Q data")
    ax.set_title(f"{y0.long_name} shape = {y0.shape}")
    ax.legend()
    return ax.get_figure(), ax


def plot_iq_no_repetition(gridded_dataset, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    y0 = gridded_dataset[dd.get_main_vars(gridded_dataset)[0]]
    ax.plot(
        y0.real,
        y0.imag,
        ".-",
        label="Data on IQ plane",
        color="C2",
    )
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    plot_centroids(ax, center_ground, center_excited)
    ax.legend()

    return ax.get_figure(), ax


# %%
plot_decay_no_repetition(dataset_gridded)
_ = plot_iq_no_repetition(dataset_gridded)

# %% [raw]
"""
T1 experiment averaged with calibration points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
y0s = np.fromiter(
    (
        np.average(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=sigma,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
        )
        for prob in probabilities
    ),
    dtype=complex,
)

y0s_calib = np.fromiter(
    (
        np.average(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=sigma,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
        )
        for prob in [0, 1]
    ),
    dtype=complex,
)

dataset = xr.Dataset(
    data_vars={
        q0_iq_par.name: (
            "dim_0",
            y0s,
            mk_main_var_attrs(**par_to_attrs(q0_iq_par), coords=[time_par.name]),
        ),
        f"{q0_iq_par.name}_cal": (
            "dim_0_cal",
            y0s_calib,
            mk_secondary_var_attrs(
                **par_to_attrs(q0_iq_par),
                coords=["cal"],
            ),
        ),
    },
    coords={
        time_par.name: ("dim_0", x0s, mk_main_coord_attrs(**par_to_attrs(time_par))),
        "cal": (
            "dim_0_cal",
            ["|0>", "|1>"],
            mk_secondary_coord_attrs(long_name="Q0 State", unit=""),
        ),
    },
    attrs=mk_dataset_attrs(
        relationships=[
            dd.QDatasetIntraRelationship(
                item_name=q0_iq_par.name,
                relation_type="calibration",
                related_names=[f"{q0_iq_par.name}_cal"],
            ).to_dict()
        ]
    ),
)


assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension=dd.get_main_dims(dataset)[0],
    coords_names=dd.get_main_coords(dataset),
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension=dd.get_secondary_dims(dataset_gridded)[0],
    coords_names=dd.get_secondary_coords(dataset_gridded),
)
dataset_gridded

# %%
fig = plt.figure(figsize=(8, 5))

ax = plt.subplot2grid((1, 10), (0, 0), colspan=9, fig=fig)
plot_decay_no_repetition(dataset_gridded, ax=ax)

ax_calib = plt.subplot2grid((1, 10), (0, 9), colspan=1, fig=fig, sharey=ax)
dataset_gridded.q0_iq_cal.real.plot(marker="o", ax=ax_calib)
dataset_gridded.q0_iq_cal.imag.plot(marker="o", ax=ax_calib)
ax_calib.yaxis.set_label_position("right")
ax_calib.yaxis.tick_right()

_ = plot_iq_no_repetition(dataset_gridded)


# %% [raw]
"""
We can use the calibration points to normalize the data and obtain the typical T1 decay.
"""

# %% [raw]
"""
.. admonition:: Data rotation and normalization utilities
    :class: dropdown
"""

# %%
# rst-json-conf: {"indent": "    "}


def rotate_data(complex_data: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates data on the complex plane around `0 + 0j`.

    Parameters
    ----------
    complex_data
        Data to rotate.
    angle
        Angle to rotate it by (in degrees).

    Returns
    -------
    :
        Rotated data.
    """
    angle_r = np.deg2rad(angle)
    rotation = np.cos(angle_r) + 1j * np.sin(angle_r)
    return rotation * complex_data


def find_rotation_angle(z1: complex, z2: complex) -> float:
    """
    Finds the angle of the line between two complex numbers on the complex plane with
    respect to the real axis.

    Parameters
    ----------
    z1
        First complex number.
    z2
        Second complex number.

    Returns
    -------
    :
        The angle found (in degrees).
    """
    return np.rad2deg(np.angle(z1 - z2))


# %% [raw]
"""
The normalization to the calibration point could look like this:
"""

# %%
angle = find_rotation_angle(*dataset_gridded.q0_iq_cal.values)
y0_rotated = rotate_data(dataset_gridded.q0_iq, -angle)
y0_calib_rotated = rotate_data(dataset_gridded.q0_iq_cal, -angle)
calib_0, calib_1 = (
    y0_calib_rotated.sel(cal="|0>").values,
    y0_calib_rotated.sel(cal="|1>").values,
)
y0_norm = (y0_rotated - calib_0) / (calib_1 - calib_0)

y0_norm.attrs.update(dataset_gridded.q0_iq.attrs)  # retain the attributes
y0_norm.attrs["long_name"] = "|1> Population"
y0_norm.attrs["units"] = ""

dataset_tmp = y0_norm.to_dataset()
dataset_tmp.attrs.update(dataset_gridded.attrs)
_ = plot_decay_no_repetition(dataset_tmp)

# %% [raw]
"""
T1 experiment storing all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
y0s = np.array(
    tuple(
        generate_mock_iq_data(
            n_shots=num_shots,
            sigma=sigma,
            center0=center_ground,
            center1=center_excited,
            prob=prob,
        )
        for prob in probabilities
    )
).T

y0s_calib = np.array(
    tuple(
        generate_mock_iq_data(
            n_shots=num_shots,
            sigma=sigma,
            center0=center_ground,
            center1=center_excited,
            prob=prob,
        )
        for prob in [0, 1]
    )
).T

dataset = xr.Dataset(
    data_vars={
        q0_iq_par.name: (
            "dim_0",
            y0s.mean(axis=0),
            mk_main_var_attrs(**par_to_attrs(q0_iq_par), coords=[time_par.name]),
        ),
        f"{q0_iq_par.name}_cal": (
            "dim_0_cal",
            y0s_calib.mean(axis=0),
            mk_secondary_var_attrs(
                **par_to_attrs(q0_iq_par),
                coords=["cal"],
            ),
        ),
        f"{q0_iq_par.name}_shots": (
            ("repetitions", "dim_0"),
            y0s,
            mk_main_var_attrs(**par_to_attrs(q0_iq_par), coords=[time_par.name]),
        ),
        f"{q0_iq_par.name}_shots_cal": (
            ("repetitions", "dim_0_cal"),
            y0s_calib,
            mk_secondary_var_attrs(
                **par_to_attrs(q0_iq_par),
                coords=["cal"],
            ),
        ),
    },
    coords={
        time_par.name: ("dim_0", x0s, mk_main_coord_attrs(**par_to_attrs(time_par))),
        "cal": (
            "dim_0_cal",
            ["|0>", "|1>"],
            mk_secondary_coord_attrs(
                long_name="Q0 State",
                unit="",
            ),
        ),
    },
    attrs=mk_dataset_attrs(
        repetitions_dims=["repetitions"],
        relationships=[
            dd.QDatasetIntraRelationship(
                item_name=q0_iq_par.name,
                related_names=[f"{q0_iq_par.name}_cal"],
                relation_type="calibration",
            ).to_dict(),
            dd.QDatasetIntraRelationship(
                item_name=f"{q0_iq_par.name}_shots",
                related_names=[f"{q0_iq_par.name}_shots_cal"],
                relation_type="calibration",
            ).to_dict(),
        ],
    ),
)


assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension=dd.get_main_dims(dataset)[0],
    coords_names=dd.get_main_coords(dataset),
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension=dd.get_secondary_dims(dataset_gridded)[0],
    coords_names=dd.get_secondary_coords(dataset_gridded),
)
dataset_gridded

# %% [raw]
"""
In this dataset we have both the averaged values and all the shots. The averaged values
can be plotted in the same way as before.
"""

# %%
_ = plot_decay_no_repetition(dataset_gridded)
_ = plot_iq_no_repetition(dataset_gridded)

# %% [raw]
"""
Here we focus on inspecting how the individual shots are distributed on the IQ plane
for some particular `Time` values.

Note that we are plotting the calibration points as well.
"""

# %%
for t_example in [x0s[len(x0s) // 5], x0s[-5]]:
    shots_example = (
        dataset_gridded.q0_iq_shots.real.sel(time=t_example),
        dataset_gridded.q0_iq_shots.imag.sel(time=t_example),
    )
    plt.hexbin(*shots_example)
    plt.xlabel("I")
    plt.ylabel("Q")
    calib_0 = dataset_gridded.q0_iq_cal.sel(cal="|0>")
    calib_1 = dataset_gridded.q0_iq_cal.sel(cal="|1>")
    plot_centroids(
        plt.gca(), (calib_0.real, calib_0.imag), (calib_1.real, calib_1.imag)
    )
    plt.suptitle(f"Shots fot t = {t_example:.5f} s")
    plt.show()


# %% [raw]
"""
We can collapse (average along) the ``repetitions`` dimension:
"""

# %% [raw]
"""
.. admonition:: Plotting utility
    :class: dropdown
"""

# %%
# rst-json-conf: {"indent": "    "}


def plot_iq_decay_repetition(gridded_dataset):
    y0_shots = gridded_dataset.q0_iq_shots
    y0_shots.real.mean(dim="repetitions").plot(marker=".", label="I data")
    y0_shots.imag.mean(dim="repetitions").plot(marker=".", label="Q data")
    plt.ylabel(f"{y0_shots.long_name} [{y0_shots.units}]")
    plt.suptitle(f"{y0_shots.name} shape = {y0_shots.shape}")
    plt.legend()

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        y0_shots.real.mean(dim="repetitions"),  # "collapses" outer dimension
        y0_shots.imag.mean(dim="repetitions"),  # "collapses" outer dimension
        ".-",
        label="Data on IQ plane",
        color="C2",
    )
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    plot_centroids(ax, center_ground, center_excited)
    ax.legend()


# %%
plot_iq_decay_repetition(dataset_gridded)

# %% [raw]
"""
T1 experiment storing digitized signals for all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# %%
# NB this is not necessarily the most efficient way to generate this mock data
y0s = np.array(
    tuple(
        generate_mock_iq_data(
            n_shots=num_shots,
            sigma=sigma,
            center0=center_ground,
            center1=center_excited,
            prob=prob,
        )
        for prob in probabilities
    )
).T

_y0s_traces = np.array(tuple(map(generate_trace_for_iq_point, y0s.flatten())))
y0s_traces = _y0s_traces.reshape(*y0s.shape, _y0s_traces.shape[-1])

y0s_calib = np.array(
    tuple(
        generate_mock_iq_data(
            n_shots=num_shots,
            sigma=sigma,
            center0=center_ground,
            center1=center_excited,
            prob=prob,
        )
        for prob in [0, 1]
    )
).T

_y0s_traces_calib = np.array(
    tuple(map(generate_trace_for_iq_point, y0s_calib.flatten()))
)
y0s_traces_calib = _y0s_traces_calib.reshape(
    *y0s_calib.shape, _y0s_traces_calib.shape[-1]
)

dataset = xr.Dataset(
    data_vars={
        f"{q0_iq_par.name}": (
            "dim_0",
            y0s.mean(axis=0),
            mk_main_var_attrs(**par_to_attrs(q0_iq_par), coords=[time_par.name]),
        ),
        f"{q0_iq_par.name}_cal": (
            "dim_0_cal",
            y0s_calib.mean(axis=0),
            mk_secondary_var_attrs(
                **par_to_attrs(q0_iq_par),
                coords=["cal"],
            ),
        ),
        f"{q0_iq_par.name}_shots": (
            ("repetitions", "dim_0"),
            y0s,
            mk_main_var_attrs(**par_to_attrs(q0_iq_par), coords=[time_par.name]),
        ),
        f"{q0_iq_par.name}_shots_cal": (
            ("repetitions", "dim_0_cal"),
            y0s_calib,
            mk_secondary_var_attrs(
                **par_to_attrs(q0_iq_par),
                coords=["cal"],
            ),
        ),
        f"{q0_iq_par.name}_traces": (
            ("repetitions", "dim_0", "dim_trace"),
            y0s_traces,
            mk_main_var_attrs(
                **par_to_attrs(q0_iq_par),
                coords=[time_par.name, "trace_time"],
            ),
        ),
        f"{q0_iq_par.name}_traces_cal": (
            ("repetitions", "dim_0_cal", "dim_trace"),
            y0s_traces_calib,
            mk_secondary_var_attrs(
                **par_to_attrs(q0_iq_par),
                coords=["cal", "trace_time"],
            ),
        ),
    },
    coords={
        time_par.name: ("dim_0", x0s, mk_main_coord_attrs(**par_to_attrs(time_par))),
        "cal": (
            "dim_0_cal",
            ["|0>", "|1>"],
            mk_secondary_coord_attrs(long_name="Q0 State", unit=""),
        ),
        "trace_time": (
            "dim_trace",
            generate_trace_time(),
            mk_main_coord_attrs(long_name="Time", unit="V"),
        ),
    },
    attrs=mk_dataset_attrs(
        repetitions_dims=["repetitions"],
        relationships=[
            dd.QDatasetIntraRelationship(
                item_name=q0_iq_par.name,
                related_names=[f"{q0_iq_par.name}_cal"],
                relation_type="calibration",
            ).to_dict(),
            dd.QDatasetIntraRelationship(
                item_name=f"{q0_iq_par.name}_shots",
                related_names=[f"{q0_iq_par.name}_shots_cal"],
                relation_type="calibration",
            ).to_dict(),
            dd.QDatasetIntraRelationship(
                item_name=f"{q0_iq_par.name}_traces",
                related_names=[f"{q0_iq_par.name}_traces_cal"],
                relation_type="calibration",
            ).to_dict(),
        ],
    ),
)


assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension="dim_0",
    # returns ['time', 'trace_time'] which is not what we need here
    # coords_names=dd.get_main_coords(dataset)
    coords_names=["time"],
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension=dd.get_secondary_dims(dataset_gridded)[0],
    coords_names=dd.get_secondary_coords(dataset_gridded),
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded, dimension="dim_trace", coords_names=["trace_time"]
)
dataset_gridded

# %%
dataset_gridded.q0_iq_traces.shape, dataset_gridded.q0_iq_traces.dims

# %% [raw]
"""
All the previous data is also present, but in this dataset we can inspect the IQ signal
for each individual shot. Let's inspect the signal of the shot number 123 of the last
point of the T1 experiment:
"""

# %%
trace_example = dataset_gridded.q0_iq_traces.sel(
    repetitions=123, time=dataset_gridded.time[-1]
)
trace_example.shape, trace_example.dtype

# %% [raw]
"""
For clarity, we plot only part of this digitized signal:
"""

# %%
trace_example_plt = trace_example[:200]
trace_example_plt.real.plot(figsize=(15, 5), marker=".")
_ = trace_example_plt.imag.plot(marker=".")
