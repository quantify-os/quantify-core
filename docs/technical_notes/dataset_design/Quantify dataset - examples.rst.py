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
rst_json_conf = {"jupyter_execute_options": [":hide-code:"]}
# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=pointless-statement
# pylint: disable=invalid-name

# %% [raw]
"""
.. _sec-quantify-dataset-examples:

Quantify dataset - examples
===========================

.. seealso::

    The complete source code of this tutorial can be found in

    .. NB .py is from notebook_to_sphinx_extension

    :jupyter-download:notebook:`Quantify dataset - examples.py`

    :jupyter-download:script:`Quantify dataset - examples.py`

.. admonition:: Imports and auxiliary utilities
    :class: dropdown
"""

# %%
rst_json_conf = {"indent": "    "}

import inspect
from IPython.display import Code, display
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from quantify_core.data import handling as dh
from rich import pretty
from pathlib import Path
import quantify_core.data.dataset_attrs as dattrs
from quantify_core.utilities import dataset_examples
from quantify_core.analysis.fitting_models import exp_decay_func
from quantify_core.analysis.calibration import rotate_to_calibrated_axis
from quantify_core.utilities.inspect_utils import display_source_code
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

# %%
rst_json_conf = {"indent": "    "}

display_source_code(dataset_examples.mk_two_qubit_chevron_dataset)

# %%
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
    coords_names=dattrs.get_main_coords(dataset),
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
stored using the Quantify dataset with increasing levels of data detail.

We start with the most simple format that contains only processed (averaged) measurements
and finish with a dataset containing the raw digitized signals from the transmon readout
during a T1 experiment.
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

# %%
rst_json_conf = {"indent": "    "}

for func in (mk_iq_shots, mk_trace_time, mk_trace_for_iq_shot):
    display_source_code(func)

# %%
rst_json_conf = {"indent": "    "}

ground = -0.2 + 0.65j
excited = 0.7 - 0.4j
centroids = ground, excited
sigmas = [0.1] * 2

shots = mk_iq_shots(
    n_shots=256,
    sigmas=sigmas,
    centers=centroids,
    probabilities=[0.4, 1 - 0.4],
)

plt.hexbin(shots.real, shots.imag)
plt.xlabel("I")
plt.ylabel("Q")
plot_centroids(plt.gca(), *centroids)

# %%
rst_json_conf = {"indent": "    "}

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
First we define a few parameters of our mock qubit and mock data acquisition.
"""

# %%
# parameters of our qubit model
tau = 30e-6
ground = -0.2 + 0.65j  # ground state on the IQ-plane
excited = 0.7 - 0.4j  # excited state on the IQ-plane
centroids = ground, excited
sigmas = [0.1] * 2  # centroids sigma, NB in general not the same for both state

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

In this first example we generate the individual measurement shots and average it,
similar to what some instrument are capable of doing directly in the hardware.
"""

# %%
q0_iq_av = np.fromiter(
    (
        np.average(
            mk_iq_shots(
                n_shots=num_shots,
                sigmas=sigmas,
                centers=centroids,
                probabilities=[prob, 1 - prob],
            )
        )
        for prob in probabilities
    ),
    dtype=complex,
)
q0_iq_av

# %% [raw]
"""
And here is how we store this data in the dataset along with the coordinates of these
datapoints:
"""

# %%
main_dims = ("main_dim",)
q0_attrs = mk_main_var_attrs(units="V", long_name="Q0 IQ amplitude", coords=["t1_time"])
t1_time_attrs = mk_main_coord_attrs(units="s", long_name="T1 Time")

data_vars = dict(q0_iq_av=(main_dims, q0_iq_av, q0_attrs))
coords = dict(t1_time=(main_dims, t1_times, t1_time_attrs))

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
    coords_names=dattrs.get_main_coords(dataset),
)
dataset_gridded


# %% [raw]
"""
.. admonition:: Plotting utilities
    :class: dropdown
"""


# %%
rst_json_conf = {"indent": "    "}


def plot_decay_no_repetition(gridded_dataset, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    y0 = gridded_dataset[dattrs.get_main_vars(gridded_dataset)[0]]
    y0.real.plot(ax=ax, marker="", label="I data")
    y0.imag.plot(ax=ax, marker="", label="Q data")
    for vals in (y0.real, y0.imag):
        ax.scatter(
            gridded_dataset[dattrs.get_main_coords(gridded_dataset)[0]].values,
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
    y0 = gridded_dataset[dattrs.get_main_vars(gridded_dataset)[0]]
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
plot_centroids(ax, *centroids)

# %% [raw]
"""
T1 experiment averaged with calibration points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is common for many experiment to require calibration data in order to interpret the
results. Often, these calibration datapoints have different array shapes. E.g. it can be
just two simple datapoints corresponding to the ground and excited states of our
transmon.
"""

# %%
q0_iq_av_cal = np.fromiter(  # generate mock calibration data
    (
        np.average(
            mk_iq_shots(
                n_shots=num_shots,
                sigmas=sigmas,
                centers=centroids,
                probabilities=[prob, 1 - prob],
            )
        )
        for prob in [0, 1]
    ),
    dtype=complex,
)
q0_iq_av_cal

# %% [raw]
"""
To accommodate this data in the dataset we make use of a secondary dimensions along which
the variables and its coordinate will lie along.

Additionally, since the secondary variable and coordinate used for calibration can have
arbitrary names and relate to other variable in more complex ways, we specify this
relationship in the dataset attributes
(see :class:`~quantify_core.data.dataset_attrs.QDatasetIntraRelationship`).
This information can be used later, for example, to run an appropriate analysis on this
dataset.
"""

# %%
secondary_dims = ("cal_dim",)
q0_cal_attrs = mk_secondary_var_attrs(
    units="V", long_name="Q0 IQ Calibration", coords=["cal"]
)
cal_attrs = mk_secondary_coord_attrs(units="", long_name="Q0 state")

relationships = [
    dattrs.QDatasetIntraRelationship(
        item_name="q0_iq_av",  # name of a variable in the dataset
        relation_type="calibration",
        related_names=["q0_iq_av_cal"],  # the secondary variable in the dataset
    ).to_dict()
]

data_vars = dict(
    q0_iq_av=(main_dims, q0_iq_av, q0_attrs),
    q0_iq_av_cal=(secondary_dims, q0_iq_av_cal, q0_cal_attrs),
)
coords = dict(
    t1_time=(main_dims, t1_times, t1_time_attrs),
    cal=(secondary_dims, ["|0>", "|1>"], cal_attrs),
)

dataset = xr.Dataset(
    data_vars=data_vars,
    coords=coords,
    attrs=mk_dataset_attrs(relationships=relationships),  # relationships added here
)


assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dattrs.get_main_dims(dataset), dattrs.get_secondary_dims(dataset)

# %%
dataset.relationships

# %% [raw]
"""
As before the coordinates can be set to index the variables that lie along the same
dimensions:
"""

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension=dattrs.get_main_dims(dataset)[0],
    coords_names=dattrs.get_main_coords(dataset),
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension=dattrs.get_secondary_dims(dataset_gridded)[0],
    coords_names=dattrs.get_secondary_coords(dataset_gridded),
)
dataset_gridded

# %%
fig = plt.figure(figsize=(8, 5))

ax = plt.subplot2grid((1, 10), (0, 0), colspan=9, fig=fig)
plot_decay_no_repetition(dataset_gridded, ax=ax)

ax_calib = plt.subplot2grid((1, 10), (0, 9), colspan=1, fig=fig, sharey=ax)
dataset_gridded.q0_iq_av_cal.real.plot(marker="o", ax=ax_calib, linestyle="")
dataset_gridded.q0_iq_av_cal.imag.plot(marker="o", ax=ax_calib, linestyle="")
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

    The normalization to the calibration points can be achieved as follows.
    Several of the
    :mod:`single-qubit time-domain analyses <quantify_core.analysis.single_qubit_timedomain>`
    provided use this under the hood.
    The result is that most of the information will now be contained within the same
    quadrature.
"""


# %%
rst_json_conf = {"indent": "    "}

rotated_and_normalized = rotate_to_calibrated_axis(
    dataset_gridded.q0_iq_av.values, *dataset_gridded.q0_iq_av_cal.values
)
dataset_tmp = dataset_gridded.q0_iq_av.to_dataset()
dataset_tmp.attrs.update(dataset_gridded.attrs)
dataset_tmp.q0_iq_av.values = rotated_and_normalized
dataset_gridded.q0_iq_av.attrs["long_name"] = "|1> Population"
dataset_gridded.q0_iq_av.attrs["units"] = ""
_ = plot_decay_no_repetition(dataset_tmp)

# %% [raw]
"""
T1 experiment storing all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we will include in the dataset all the single qubit states (shot) for each
individual measurement.
"""

# %%
q0_iq_shots = np.array(
    tuple(
        mk_iq_shots(
            n_shots=num_shots,
            sigmas=sigmas,
            centers=centroids,
            probabilities=[prob, 1 - prob],
        )
        for prob in probabilities
    )
).T

q0_iq_shots_cal = np.array(
    tuple(
        mk_iq_shots(
            n_shots=num_shots,
            sigmas=sigmas,
            centers=centroids,
            probabilities=[prob, 1 - prob],
        )
        for prob in [0, 1]
    )
).T
q0_iq_shots.shape, q0_iq_shots_cal.shape

# %%
# the xarray dimensions will no require an outer repetitions dimension
secondary_dims_rep = ("repetitions", "cal_dim")
main_dims_rep = ("repetitions", "main_dim")

relationships = [
    dattrs.QDatasetIntraRelationship(
        item_name="q0_iq_av",
        relation_type="calibration",
        related_names=["q0_iq_av_cal"],
    ).to_dict(),
    dattrs.QDatasetIntraRelationship(
        item_name="q0_iq_av_shots",
        relation_type="calibration",
        related_names=["q0_iq_av_cal_shots"],
    ).to_dict(),
    # suggestion of a custom relationship
    dattrs.QDatasetIntraRelationship(
        item_name="q0_iq_av",
        relation_type="individual_shots",
        related_names=["q0_iq_av_shots"],
    ).to_dict(),
]

data_vars = dict(
    # these are the same as in the previous dataset, and are now redundant,
    # however, we include them to showcase the dataset flexibility
    q0_iq_av=(main_dims, q0_iq_shots.mean(axis=0), q0_attrs),
    q0_iq_av_cal=(secondary_dims, q0_iq_shots_cal.mean(axis=0), q0_cal_attrs),
    q0_iq_shots=(main_dims_rep, q0_iq_shots, q0_attrs),
    q0_iq_shots_cal=(secondary_dims_rep, q0_iq_shots_cal, q0_cal_attrs),
)
coords = dict(
    t1_time=(main_dims, t1_times, t1_time_attrs),
    cal=(secondary_dims, ["|0>", "|1>"], cal_attrs),
)

dataset = xr.Dataset(
    data_vars=data_vars,
    coords=coords,
    attrs=mk_dataset_attrs(
        relationships=relationships,  # relationships added here
        # the dimensions that correspond to repetitions need to be specified here
        repetitions_dims=["repetitions"],
    ),
)

assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %% [raw]
"""
.. note:

    Note that we have to specify ``repetitions_dims=["repetitions"]`` in the dataset
    attributes in order to correctly identify the main and secondary dimensions later.
"""

# %%
rst_json_conf = {"indent": "    "}

dattrs.get_main_dims(dataset), dattrs.get_secondary_dims(dataset)

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension=dattrs.get_main_dims(dataset)[0],
    coords_names=dattrs.get_main_coords(dataset),
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension=dattrs.get_secondary_dims(dataset_gridded)[0],
    coords_names=dattrs.get_secondary_coords(dataset_gridded),
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
chosen_time_values = [
    t1_times[1],  # second value selected otherwise we won't see both centroids
    t1_times[len(t1_times) // 5],  # a value close to the end of the experiment
]
for t_example in chosen_time_values:
    shots_example = (
        dataset_gridded.q0_iq_shots.real.sel(t1_time=t_example),
        dataset_gridded.q0_iq_shots.imag.sel(t1_time=t_example),
    )
    plt.hexbin(*shots_example)
    plt.xlabel("I")
    plt.ylabel("Q")
    calib_0 = dataset_gridded.q0_iq_av_cal.sel(cal="|0>")
    calib_1 = dataset_gridded.q0_iq_av_cal.sel(cal="|1>")
    plot_centroids(plt.gca(), calib_0, calib_1)
    plt.suptitle(f"Shots fot t = {t_example:.5f} [s]")
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
rst_json_conf = {"indent": "    "}


def plot_iq_decay_repetition(gridded_dataset):
    y0_shots = gridded_dataset.q0_iq_shots
    y0_shots_mean = y0_shots.mean(dim="repetitions")
    y0_shots_mean.real.plot(marker=".", label="I data")
    y0_shots_mean.imag.plot(marker=".", label="Q data")
    plt.ylabel(f"{y0_shots.long_name} [{y0_shots.units}]")
    plt.suptitle(f"{y0_shots.name} shape = {y0_shots.shape}")
    plt.legend()

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        y0_shots_mean.real,
        y0_shots_mean.imag,
        ".-",
        label="Data on IQ plane",
        color="C2",
    )
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.legend()

    return fig, ax


# %%
fig, ax = plot_iq_decay_repetition(dataset_gridded)
plot_centroids(ax, *centroids)

# %% [raw]
"""
T1 experiment storing digitized signals for all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, in addition to the individual shots we will store all the digitized readout
signals that are required to obtain the previous measurement results.
"""

# %%
_q0_traces = np.array(tuple(map(mk_trace_for_iq_shot, q0_iq_shots.flatten())))
q0_traces = _q0_traces.reshape(*q0_iq_shots.shape, _q0_traces.shape[-1])

_q0_traces_cal = np.array(tuple(map(mk_trace_for_iq_shot, q0_iq_shots_cal.flatten())))
q0_traces_cal = _q0_traces_cal.reshape(*q0_iq_shots_cal.shape, _q0_traces_cal.shape[-1])

q0_traces.shape, q0_traces_cal.shape

# %%
# NB this is not necessarily the most efficient way to generate this mock data
traces_dims = ("repetitions", "main_dim", "trace_dim")
traces_cal_dims = ("repetitions", "cal_dim", "trace_dim")
trace_times = mk_trace_time()
trace_attrs = mk_main_coord_attrs(long_name="Trace time", unit="s")

relationships_with_traces = relationships + [
    dattrs.QDatasetIntraRelationship(
        item_name="q0_traces",
        related_names=["q0_traces_cal"],
        relation_type="calibration",
    ).to_dict(),
]

q0_trace_attrs = dict(q0_attrs)
q0_trace_attrs.update(coords=["t1_time", "trace_time"])
q0_trace_cal_attrs = dict(q0_attrs)
q0_trace_cal_attrs.update(coords=["cal", "trace_time"])

data_vars = dict(
    q0_iq_av=(main_dims, q0_iq_av, q0_attrs),
    q0_iq_av_cal=(secondary_dims, q0_iq_av_cal, q0_cal_attrs),
    q0_iq_shots=(main_dims_rep, q0_iq_shots, q0_attrs),
    q0_iq_shots_cal=(secondary_dims_rep, q0_iq_shots_cal, q0_cal_attrs),
    q0_traces=(traces_dims, q0_traces, q0_trace_attrs),
    q0_traces_cal=(traces_cal_dims, q0_traces_cal, q0_trace_cal_attrs),
)
coords = dict(
    t1_time=(main_dims, t1_times, t1_time_attrs),
    cal=(secondary_dims, ["|0>", "|1>"], cal_attrs),
    trace_time=(("trace_dim",), trace_times, trace_attrs),
)

dataset = xr.Dataset(
    data_vars=data_vars,
    coords=coords,
    attrs=mk_dataset_attrs(
        repetitions_dims=["repetitions"], relationships=relationships_with_traces
    ),
)

assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension="main_dim",
    # returns ['time', 'trace_time'] which is not what we need here
    # coords_names=dattrs.get_main_coords(dataset)
    coords_names=["t1_time"],
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension="cal_dim",
    coords_names=["cal"],
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded, dimension="trace_dim", coords_names=["trace_time"]
)
dataset_gridded

# %%
dataset_gridded.q0_traces.shape, dataset_gridded.q0_traces.dims

# %% [raw]
"""
All the previous data is also present, but in this dataset we can inspect the IQ signal
for each individual shot. Let's inspect the signal of the shot number 123 of the last
"point" of the T1 experiment:
"""

# %%
trace_example = dataset_gridded.q0_traces.sel(
    repetitions=123, t1_time=dataset_gridded.t1_time[-1]
)
trace_example.shape, trace_example.dtype

# %% [raw]
"""
Now we can plot this digitized signals for each quadrature. For clarity we plot only
part of the signal.
"""

# %%
trace_example_plt = trace_example[:200]
trace_example_plt.real.plot(figsize=(15, 5), marker=".")
_ = trace_example_plt.imag.plot(marker=".")
