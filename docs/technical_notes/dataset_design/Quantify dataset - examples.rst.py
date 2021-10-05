# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
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
# pylint: disable=pointless-statement
# pylint: disable=invalid-name

# %%
# %load_ext autoreload
# %autoreload 1
# %aimport quantify_core.data.dataset_attrs
# %aimport quantify_core.data.dataset_adapters
# %aimport quantify_core.utilities.examples_support

# %% [raw]
"""
.. _sec-quantify-dataset-examples:

Quantify dataset - examples
===========================

.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Quantify dataset - examples`

    :jupyter-download:script:`Quantify dataset - examples`

.. admonition:: Imports and auxiliary utilities
    :class: dropdown
"""

# %% tags=[]
# rst-json-conf: {"indent": "    ", "jupyter_execute_options": [":hide-output:"]}

import inspect
from IPython.display import Code, display
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from quantify_core.data import handling as dh
from quantify_core.measurement import grid_setpoints
from qcodes import ManualParameter
from rich import pretty
from pathlib import Path
import quantify_core.data.dataset_attrs as dd
from quantify_core.utilities import dataset_examples
from quantify_core.analysis.fitting_models import exp_decay_func
from quantify_core.utilities.examples_support import (
    mk_iq_shots,
    mk_trace_time,
    mk_trace_for_iq_shot,
    plot_centroids,
    mk_dataset_attrs,
    mk_main_coord_attrs,
    mk_secondary_coord_attrs,
    mk_main_var_attrs,
    mk_secondary_var_attrs,
    round_trip_dataset,
)

pretty.install()

dh.set_datadir(Path.home() / "quantify-data")  # change me!

# %% [raw]
"""
In this page we explore a series of datasets that comply with the :ref:`Quantify dataset specification <dataset-spec>`.

2D dataset example
------------------

We use the :func:`~quantify_core.utilities.dataset_examples.mk_two_qubit_chevron_dataset`
to generate our examplery dataset. Its source code is conveniently displayed in the
drop down below.
"""

# %% [raw]
"""
.. admonition:: Generate a 2D dataset
    :class: dropdown
"""

# %% tags=[] jupyter={"outputs_hidden": true}
# rst-json-conf: {"indent": "    "}

Code(
    inspect.getsource(dataset_examples.mk_two_qubit_chevron_dataset), language="python"
)

# %% tags=[]
dataset = dataset_examples.mk_two_qubit_chevron_dataset()

assert dataset == round_trip_dataset(dataset)  # confirm read/write
dataset

# %% [raw]
"""
The data within this dataset can be easily visualized using xarray facilities,
however we first need to convert the Quantify dataset to a "gridded" version with as
shown below.

Since our dataset contains multiple repetitions of the same experiment, it is convenient
to visualize them on different plots.
"""

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension="main_dim",
    coords_names=dd.get_main_coords(dataset),
)
dataset_gridded.pop_q0.plot.pcolormesh(x="amp", col="repetitions")
_ = dataset_gridded.pop_q1.plot.pcolormesh(x="amp", col="repetitions")

# %% [raw]
"""
In xarray, among other features, it is possible to average along a dimension which can
be very convenient to average out some of the noise:
"""

# %%
_ = dataset_gridded.pop_q0.mean(dim="repetitions").plot(x="amp")

# %% [raw]
"""
A repetitions dimension can be indexed by a coordinate such that we can have some
specific label for each of our repetitions. To showcase this, we will modify the previous
dataset by merging it with a dataset containing the relevant extra information.
"""

# %%
coord_name = "repetitions"
coord_dims = ("repetitions",)
coord_values = ["A", "B", "C", "D", "E"]
dataset_indexed_rep = xr.Dataset(
    coords={coord_name: (coord_dims, coord_values)},
    attrs=dict(repetitions_dims=["repetitions"]),
)

dataset_indexed_rep

# %%
# merge with the previous dataset
dataset_rep = dataset_gridded.merge(dataset_indexed_rep, combine_attrs="drop_conflicts")

assert dataset_rep == round_trip_dataset(dataset_rep)  # confirm read/write

dataset_rep

# %% [raw]
"""
Now we can select a specific repetition by its coordinate, in this case a string label.
"""

# %%
_ = dataset_rep.pop_q0.sel(repetitions="E").plot(x="amp")

# %% [raw]
"""
T1 dataset examples
-------------------

The T1 experiment is one of the most common quantum computing experiments.
Here we explore how the datasets for such an experiment, for a transmon qubit, can be
stored using the Qunatify dataset with increasing levels of data detail.

We start with the most simple format that contains only processed (averaged) measurements
and finish with a dataset containing the full raw data of a T1 experiment.
"""

# %% [raw]
"""
.. admonition:: Mock data utilities
    :class: dropdown

    We use a few auxiliary functions to generate, manipulate and plot the data of the
    examples that follow:

    - :func:`quantify_core.utilities.examples_support.mk_iq_shots`
    - :func:`quantify_core.utilities.examples_support.mk_trace_time`
    - :func:`quantify_core.utilities.examples_support.mk_trace_for_iq_shot`
    - :func:`quantify_core.utilities.examples_support.plot_centroids`
    - :func:`quantify_core.analysis.fitting_models.exp_decay_func`
    
    Below you can find the source-code of the most important ones and a few usage 
    examples in order to gain some intuition for the mock data.
"""

# %% jupyter={"outputs_hidden": true} tags=[]
# rst-json-conf: {"indent": "    "}

for func in (mk_iq_shots, mk_trace_time, mk_trace_for_iq_shot):
    code = Code(inspect.getsource(func), language="python")
    display(code)

# %%
# rst-json-conf: {"indent": "    "}

centroid_ground = -0.2 + 0.65j
centroid_excited = 0.7 - 0.4j

shots = mk_iq_shots(
    n_shots=256,
    sigmas=[0.1] * 2,
    centers=[centroid_ground, centroid_excited],
    probabilities=[0.4, 1 - 0.4],
)

plt.hexbin(shots.real, shots.imag)
plt.xlabel("I")
plt.ylabel("Q")
plot_centroids(plt.gca(), centroid_ground, centroid_excited)

# %%
# rst-json-conf: {"indent": "    "}

time = mk_trace_time()
trace = mk_trace_for_iq_shot(shots[0])

fig, ax = plt.subplots(1, 1, figsize=(12, 12 / 1.61 / 2))
_ = ax.plot(time * 1e6, trace.imag, ".-", label="I-quadrature")
_ = ax.plot(time * 1e6, trace.real, ".-", label="Q-quadrature")
_ = ax.set_xlabel("Time [µs]")
_ = ax.set_ylabel("Amplitude [V]")
_ = ax.legend()

# %% [raw]
"""
First we define a few parameters of our mock qubit and mock data aquisition.
"""

# %%
# parameters of our qubit model
tau = 30e-6
ground = -0.2 + 0.65j  # ground state on the IQ-plane
excited = 0.7 + -0.4j  # excited state on the IQ-plane
sigma = 0.1  # centroids sigma, NB in general not the same for both state

# mock of data acquisition configuration
# NB usually at least 1000+ shots are taken, here we use less for faster code execution
num_shots = 256
# time delays between exciting the qubit and measuring its state
t1_times = np.linspace(0, 120e-6, 30)

# NB this are the ideal probabilities from repeating the measurement many times for a
# qubit with a lifetime given by tau
probabilities = exp_decay_func(t=t1_times, tau=tau, offset=0, n_factor=1, amplitude=1)

# Ideal experiment result
plt.ylabel("|1> probability")
plt.suptitle("Typical processed data of a T1 experiment")
_ = plt.plot(t1_times * 1e6, probabilities, ".-")
_ = plt.xlabel("Time [µs]")

# %% [raw]
"""
T1 experiment averaged
~~~~~~~~~~~~~~~~~~~~~~

In this first example we generate the pseudo-random and average it, similar to what
some instrument are capable to do directly in the hardware.
"""

# %%
q0_iq_av = np.fromiter(
    (
        np.average(
            mk_iq_shots(
                n_shots=num_shots,
                sigmas=[sigma] * 2,
                centers=[ground, excited],
                probabilities=[prob, 1 - prob],
            )
        )
        for prob in probabilities
    ),
    dtype=complex,
)
q0_iq_av

# %%
dims = ("main_dim",)
q0_attrs = mk_main_var_attrs(units="V", long_name="Q0 IQ amplitude", coords=["t1_time"])
t1_time_attrs = mk_main_coord_attrs(units="s", long_name="T1 Time")

data_vars = dict(q0_iq_av=(dims, q0_iq_av, q0_attrs))
coords = dict(t1_time=(dims, t1_times, t1_time_attrs))

dataset = xr.Dataset(
    data_vars=data_vars,
    coords=coords,
    attrs=mk_dataset_attrs(),
)


assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension="main_dim",
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
    y0.real.plot(ax=ax, marker="", label="I data")
    y0.imag.plot(ax=ax, marker="", label="Q data")
    for vals in (y0.real, y0.imag):
        ax.scatter(
            gridded_dataset[dd.get_main_coords(gridded_dataset)[0]].values,
            vals,
            marker="o",
            c=np.arange(0, len(y0)),
            cmap="viridis",
        )
    ax.set_title(f"{y0.long_name} [{y0.name}]; shape = {y0.shape}")
    ax.legend()
    return ax.get_figure(), ax


def plot_iq_no_repetition(gridded_dataset, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    y0 = gridded_dataset[dd.get_main_vars(gridded_dataset)[0]]
    ax.scatter(
        y0.real,
        y0.imag,
        marker="o",
        label=f"Data on IQ plane [{y0.name}]",
        c=np.arange(0, len(y0)),
        cmap="viridis",
    )
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.legend()

    return ax.get_figure(), ax


# %%
plot_decay_no_repetition(dataset_gridded)
fig, ax = plot_iq_no_repetition(dataset_gridded)
plot_centroids(ax, ground, excited)

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

# %%
