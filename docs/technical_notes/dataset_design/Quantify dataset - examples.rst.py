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
rst_conf = {"jupyter_execute_options": [":hide-code:"]}
# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=pointless-statement
# pylint: disable=invalid-name
# pylint: disable=expression-not-assigned
# pylint: disable=duplicate-code


# %% [raw]
"""
.. _sec-dataset-examples:

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
rst_conf = {"indent": "    "}

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from rich import pretty

import quantify_core.data.dataset_attrs as dattrs
from quantify_core.analysis.calibration import rotate_to_calibrated_axis
from quantify_core.analysis.fitting_models import exp_decay_func
from quantify_core.data import handling as dh
from quantify_core.utilities import dataset_examples
from quantify_core.utilities.examples_support import (
    mk_iq_shots,
    mk_trace_for_iq_shot,
    mk_trace_time,
    round_trip_dataset,
)
from quantify_core.utilities.inspect_utils import display_source_code
from quantify_core.visualization.mpl_plotting import (
    plot_complex_points,
    plot_xr_complex,
    plot_xr_complex_on_plane,
)

pretty.install()

dh.set_datadir(Path.home() / "quantify-data")  # change me!

# %% [raw]
"""
In this page we explore a series of datasets that comply with the :ref:`Quantify dataset specification <dataset-spec>`.

2D dataset example
------------------

We use the :func:`~quantify_core.utilities.dataset_examples.mk_two_qubit_chevron_dataset`
to generate our exemplary dataset. Its source code is conveniently displayed in the
drop down below.
"""

# %% [raw]
"""
.. admonition:: Generate a 2D dataset
    :class: dropdown
"""

# %%
rst_conf = {"indent": "    "}

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
coord_dims = ("repetitions",)
coord_values = ["A", "B", "C", "D", "E"]
dataset_indexed_rep = xr.Dataset(coords=dict(repetitions=(coord_dims, coord_values)))

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
    - :func:`quantify_core.analysis.fitting_models.exp_decay_func`

    Below you can find the source-code of the most important ones and a few usage
    examples in order to gain some intuition for the mock data.
"""

# %%
rst_conf = {"indent": "    "}

for func in (mk_iq_shots, mk_trace_time, mk_trace_for_iq_shot):
    display_source_code(func)

# %%
rst_conf = {"indent": "    "}

ground = -0.2 + 0.65j
excited = 0.7 - 0.4j
centers = ground, excited
sigmas = [0.1] * 2

shots = mk_iq_shots(
    num_shots=256,
    sigmas=sigmas,
    centers=centers,
    probabilities=[0.4, 1 - 0.4],
)

plt.hexbin(shots.real, shots.imag)
plt.xlabel("I")
plt.ylabel("Q")
_ = plot_complex_points(centers, ax=plt.gca())

# %%
rst_conf = {"indent": "    "}

time = mk_trace_time()
trace = mk_trace_for_iq_shot(shots[0])

fig, ax = plt.subplots(1, 1, figsize=(12, 12 / 1.61 / 2))
ax.plot(time * 1e6, trace.imag, ".-", label="I-quadrature")
ax.plot(time * 1e6, trace.real, ".-", label="Q-quadrature")
ax.set_xlabel("Time [µs]")
ax.set_ylabel("Amplitude [V]")
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
centers = ground, excited
sigmas = [0.1] * 2  # sigma, NB in general not the same for both state

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
plt.plot(t1_times * 1e6, probabilities, ".-")
_ = plt.xlabel("Time [µs]")

# %%
# convenience dict with the mock parameters
mock_conf = dict(
    num_shots=num_shots,
    centers=centers,
    sigmas=sigmas,
    t1_times=t1_times,
    probabilities=probabilities,
)

# %% [raw]
"""
T1 experiment averaged
~~~~~~~~~~~~~~~~~~~~~~

In this first example we generate the individual measurement shots and average it,
similar to what some instrument are capable of doing directly in the hardware.

Here is how we store this data in the dataset along with the coordinates of these
datapoints:
"""

# %% [raw]
"""
.. admonition:: Generate dataset
    :class: dropdown, toggle-shown
"""

# %%
rst_conf = {"indent": "    "}
display_source_code(dataset_examples.mk_t1_av_dataset)

# %%
dataset = dataset_examples.mk_t1_av_dataset(**mock_conf)
assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset.q0_iq_av.shape, dataset.q0_iq_av.dtype

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
rst_conf = {"indent": "    "}

display_source_code(plot_xr_complex)
display_source_code(plot_xr_complex_on_plane)

# %%
plot_xr_complex(dataset_gridded.q0_iq_av)
fig, ax = plot_xr_complex_on_plane(dataset_gridded.q0_iq_av)
_ = plot_complex_points(centers, ax=ax)

# %% [raw]
"""
T1 experiment averaged with calibration points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is common for many experiment to require calibration data in order to interpret the
results. Often, these calibration datapoints have different array shapes. E.g. it can be
just two simple datapoints corresponding to the ground and excited states of our
transmon.

To accommodate this data in the dataset we make use of a secondary dimensions along which
the variables and its coordinate will lie along.

Additionally, since the secondary variable and coordinate used for calibration can have
arbitrary names and relate to other variable in more complex ways, we specify this
relationship in the dataset attributes
(see :class:`~quantify_core.data.dataset_attrs.QDatasetIntraRelationship`).
This information can be used later, for example, to run an appropriate analysis on this
dataset.
"""

# %% [raw]
"""
.. admonition:: Generate dataset
    :class: dropdown, toggle-shown
"""

# %%
rst_conf = {"indent": "    "}
display_source_code(dataset_examples.mk_t1_av_with_cal_dataset)

# %%
dataset = dataset_examples.mk_t1_av_with_cal_dataset(**mock_conf)
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
    dimension="main_dim",
    coords_names=dattrs.get_main_coords(dataset),
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension="cal_dim",
    coords_names=dattrs.get_secondary_coords(dataset_gridded),
)
dataset_gridded

# %%
fig = plt.figure(figsize=(8, 5))

ax = plt.subplot2grid((1, 10), (0, 0), colspan=9, fig=fig)
plot_xr_complex(dataset_gridded.q0_iq_av, ax=ax)

ax_calib = plt.subplot2grid((1, 10), (0, 9), colspan=1, fig=fig, sharey=ax)
for i, color in zip(
    range(2), ["C0", "C1"]
):  # plot each calibration point with same color
    dataset_gridded.q0_iq_av_cal.real[i : i + 1].plot.line(
        marker="o", ax=ax_calib, linestyle="", color=color
    )
    dataset_gridded.q0_iq_av_cal.imag[i : i + 1].plot.line(
        marker="o", ax=ax_calib, linestyle="", color=color
    )
ax_calib.yaxis.set_label_position("right")
ax_calib.yaxis.tick_right()

fig, ax = plot_xr_complex_on_plane(dataset_gridded.q0_iq_av)
_ = plot_complex_points(dataset_gridded.q0_iq_av_cal.values, ax=ax)


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
rst_conf = {"indent": "    "}

rotated_and_normalized = rotate_to_calibrated_axis(
    dataset_gridded.q0_iq_av.values, *dataset_gridded.q0_iq_av_cal.values
)
rotated_and_normalized_da = xr.DataArray(dataset_gridded.q0_iq_av)
rotated_and_normalized_da.values = rotated_and_normalized
rotated_and_normalized_da.attrs["long_name"] = "|1> Population"
rotated_and_normalized_da.attrs["units"] = ""
_ = plot_xr_complex(rotated_and_normalized_da)

# %% [raw]
"""
T1 experiment storing all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we will include in the dataset all the single qubit states (shot) for each
individual measurement.
"""

# %% [raw]
"""
.. admonition:: Generate dataset
    :class: dropdown, toggle-shown
"""

# %%
rst_conf = {"indent": "    "}
display_source_code(dataset_examples.mk_t1_shots_dataset)

# %%
dataset = dataset_examples.mk_t1_shots_dataset(**mock_conf)
dataset

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension="main_dim",
    coords_names=dattrs.get_main_coords(dataset),
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded,
    dimension="cal_dim",
    coords_names=dattrs.get_secondary_coords(dataset_gridded),
)
dataset_gridded

# %% [raw]
"""
In this dataset we have both the averaged values and all the shots. The averaged values
can be plotted in the same way as before.
"""

# %%
_ = plot_xr_complex(dataset_gridded.q0_iq_av)
_, ax = plot_xr_complex_on_plane(dataset_gridded.q0_iq_av)
_ = plot_complex_points(dataset_gridded.q0_iq_av_cal.values, ax=ax)

# %% [raw]
"""
Here we focus on inspecting how the individual shots are distributed on the IQ plane
for some particular `Time` values.

Note that we are plotting the calibration points as well.
"""

# %%
chosen_time_values = [
    t1_times[1],  # second value selected otherwise we won't see both centers
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
    plot_complex_points([calib_0, calib_1], ax=plt.gca())
    plt.suptitle(f"Shots for t = {t_example:.5f} [s]")
    plt.show()


# %% [raw]
"""
We can collapse (average along) the ``repetitions`` dimension:
"""

# %%
q0_iq_shots_mean = dataset_gridded.q0_iq_shots.mean(dim="repetitions", keep_attrs=True)
plot_xr_complex(q0_iq_shots_mean)
_, ax = plot_xr_complex_on_plane(q0_iq_shots_mean)
_ = plot_complex_points(centers, ax=ax)

# %% [raw]
"""
.. _sec-dataset-t1-traces:

T1 experiment storing digitized signals for all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, in addition to the individual shots we will store all the digitized readout
signals that are required to obtain the previous measurement results.
"""


# %% [raw]
"""
.. admonition:: Generate dataset
    :class: dropdown, toggle-shown
"""

# %%
rst_conf = {"indent": "    "}
display_source_code(dataset_examples.mk_t1_traces_dataset)

# %%
dataset = dataset_examples.mk_t1_traces_dataset(**mock_conf)
assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset.q0_traces.shape, dataset.q0_traces_cal.shape

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension="main_dim",
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
trace_example_plt.real.plot(figsize=(15, 5), marker=".", label="I-quadrature")
trace_example_plt.imag.plot(marker=".", label="Q-quadrature")
plt.gca().legend()
plt.show()
