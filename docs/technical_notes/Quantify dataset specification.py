#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Quantify dataset proposal/exploration
#
#
# Author: Victor Negîrneac @ Qblox
# Date: 2021-07-30
#
# Here we explore what the quantify dataset should look like in order to support the advanced data types and shapes.
# We start from current "implementation" and add complexity incrementally:
#
# - First by adding repetitions (e.g. being able to store all the binary shots);
# - Second by making each "data point" a complex time series (a.k.a. "digitized trace", or simply "trace").
#
# For each dataset we plot **ALL** the data in the dataset. Appreciate how awesome is `xarray`!
#
# - Note 1: the datasets are created "manually" in order to have full control over its structure.
# - Note 2: to my own surprise the data writing/saving of quantify "just worked".
# - Note 3: same goes for the `dh.to_gridded_dataset()` function.

# %% [markdown]
# Concepts
# --------
#
# - Experiment coordinate(s)
#     - `xarray` coordinate variables following the naming convention `f"x{i}"` with `i >= 0` a integer.
#     - Often correspond to physical coordinates, e.g., a signal frequency or amplitude.
# - Exeperiment variable(s)
#     - `xarray` variables following the naming convention `f"y{i}"` with `i >= 0` a integer.
#     - Often correspond to a physical quantity being measured, e.g., the signal magnitude at a specific frequency measured on a metal contact of a quantum chip.

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from quantify_core.data import handling as dh


def dataset_round_trip(ds: xr.Dataset) -> xr.Dataset:
    tuid = dh.gen_tuid()
    ds.attrs["tuid"] = tuid
    dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, ds)
    return dh.load_dataset(tuid)


from pathlib import Path
from quantify_core.data.handling import get_datadir, set_datadir

set_datadir(Path.home() / "quantify-data")  # change me!

# %%
l_t, l_a = 4, 3
num_setpoints = l_t * l_a

x0s = np.linspace(1, 4, l_t)
x1s = np.linspace(-1, 0, l_a)

x1s = np.repeat(x1s, l_t)
x0s = np.tile(x0s, l_a)

assert len(x0s) == len(x1s) == num_setpoints

# %% [markdown]
# ### 2D Dataset
#
# NB: ``acq_set_0`` in the dataset is kind of equivalent to ``acq_index`` apart from technical details, and ``y0``, ``y1``, etc. are the ``acq_channel``\s.

# %%
dataset = xr.Dataset(
    data_vars={
        "y0": (("repetition", "acq_set_0"), [np.linspace(1, 4, num_setpoints)]),
        "y1": (("repetition", "acq_set_0"), [np.linspace(-4, 5, num_setpoints) + 0.2]),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded

# %%
dataset_gridded.y0.plot()
plt.show()
dataset_gridded.y1.plot()
plt.show()

# %% [markdown]
# ### 2D Dataset with explicit single repetition

# %%
dataset = xr.Dataset(
    data_vars={
        "y0": (("repetition", "acq_set_0"), [np.linspace(1, 4, num_setpoints)]),
        "y1": (("repetition", "acq_set_0"), [np.linspace(-4, 5, num_setpoints) + 0.2]),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded

# %%
# Plotting still works even though there is an extra dimension
dataset_gridded.y0.plot()
plt.show()
dataset_gridded.y1.plot()
plt.show()

# %% [markdown]
# ### 2D Dataset with multiple repetitions

# %%
rep_num = 5
dataset = xr.Dataset(
    data_vars={
        "y0": (
            ("repetition", "acq_set_0"),
            [np.linspace(1, 4, num_setpoints) + i for i in range(rep_num)],
        ),
        "y1": (
            ("repetition", "acq_set_0"),
            [np.linspace(-4, 5, num_setpoints) + 2 * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded

# %%
dataset_gridded.y0.plot(x="x0", y="x1", col="repetition")
dataset_gridded.y1.plot(x="x0", y="x1", col="repetition")

# %% [markdown]
# ### 2D Dataset with repetitions and (complex) time "traces"

# %%
rep_num = 4
time = np.arange(0, 400e-9, 10e-9)
# cos = np.cos(2 * np.pi * 3e6 * time)
# plt.plot(time, cos, ".-")

# NB: just some "random" frequency and amplitude change, not really dependent on x0 and x1
traces = np.array(
    [
        (1.2 - f / 7e6) * np.exp(-2j * np.pi * f * time)
        for f in np.linspace(3e6, 7e6, num_setpoints)
    ]
)

dataset = xr.Dataset(
    data_vars={
        "y0_tseries": (
            ("repetition", "acq_set_0", "time"),
            [traces + (0.4 - 0.8j) * i for i in range(rep_num)],
        ),
        "y1_tseries": (
            ("repetition", "acq_set_0", "time"),
            [traces + (-0.6 + 0.9j) * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
        # NB there is a dimension named `time` and also a coordinate with the same name!
        # NB2 xarray automatically understands that we want to index the `time` dimension
        # using the `time` coordinate (it will appear in bold below).
        "time": ("time", time, {"unit": "s"}),
    },
)
dataset = dataset.set_coords(["x0", "x1"])
# dataset = dataset.set_index({"time": "time"})

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded

# %%
darray = dataset_gridded.y0_tseries
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y0_tseries
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)

darray = dataset_gridded.y1_tseries
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y1_tseries
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)

# %% [markdown]
# ### `acq_channel`\s with different datashapes per measured "data point"
# This would be case when we want to save e.g. the qubit population **AND** the time traces.

# %%
rep_num = 4
time = np.arange(0, 400e-9, 10e-9)
# cos = np.cos(2 * np.pi * 3e6 * time)
# plt.plot(time, cos, ".-")

# NB: just some "random" frequency and amplitude change, not really dependent on x0 and x1
traces = np.array(
    [
        (1.2 - f / 7e6) * np.exp(-2j * np.pi * f * time)
        for f in np.linspace(3e6, 7e6, num_setpoints)
    ]
)

dataset = xr.Dataset(
    data_vars={
        "y0": (
            ("repetition", "acq_set_0"),
            [np.linspace(1, 4, num_setpoints) + i for i in range(rep_num)],
        ),
        "y0_time": (
            ("repetition", "acq_set_0", "time"),
            [traces + (0.4 - 0.8j) * i for i in range(rep_num)],
        ),
        "y1_time": (
            ("repetition", "acq_set_0", "time"),
            [traces + (-0.6 + 0.9j) * i for i in range(rep_num)],
        ),
        "y1": (
            ("repetition", "acq_set_0"),
            [np.linspace(-4, 5, num_setpoints) + 2 * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
        "time": ("time", time, {"unit": "s"}),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded

# %%
dataset.y0.shape, dataset.y1.shape, dataset.y0.dtype, dataset.y1.dtype

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded

# %%
dataset_gridded.y0.plot(x="x0", y="x1", col="repetition")
dataset_gridded.y1.plot(x="x0", y="x1", col="repetition")

darray = dataset_gridded.y0_time
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y0_time
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)

darray = dataset_gridded.y1_time
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y1_time
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)

# %% [markdown]
# ### 2D dataset with calibration points
#
# One possibility is to use a dedicated variable(s) with an "independent" dedicated `xarray` dimension and a naming convention.

# %%
rep_num = 5
dataset = xr.Dataset(
    data_vars={
        "y0": (
            ("repetition", "acq_set_0"),
            [np.linspace(1, 4, num_setpoints) + i for i in range(rep_num)],
        ),
        "y0_calib": (
            ("repetition", "acq_set_0_calib"),
            [[1, 4] for i in range(rep_num)],
        ),
        "y1": (
            ("repetition", "acq_set_0"),
            [np.linspace(-4, 5, num_setpoints) + 2 * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded

# %%

# %%

# %%
# from importlib import reload
# from quantify_core.utilities import _docs_helpers

# reload(_docs_helpers)

# file_name = "Quantify dataset specification"
# _docs_helpers.notebook_to_rst(f"{file_name}.ipynb", f"{file_name}.rst")

# %%
