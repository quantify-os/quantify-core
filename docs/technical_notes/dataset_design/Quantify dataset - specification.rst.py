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
# .. _dataset-spec:
#
# Quantify dataset specification
# ==============================

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
import quantify_core.data.dataset_adapters as da
from quantify_core.utilities.examples_support import (
    mk_dataset_attrs,
    mk_exp_coord_attrs,
    mk_exp_var_attrs,
    dataset_round_trip,
    par_to_attrs,
)

from typing import List, Tuple

from importlib import reload

pretty.install()

set_datadir(Path.home() / "quantify-data")  # change me!

# %% [raw]
# This document describes the Qauntify dataset specification.
# Here we focus on the concepts and terminology specific to the Quantify dataset.
# It is based on the Xarray dataset, hence, we assume basic familiarity with the :class:`xarray.Dataset`.
# If you are not familiar with it, we highly recomend to first have a look at the :ref:`xarray-intro` for a brief overview.

# %% [raw]
# .. _sec-experiment-coordinates-and-variables:
#
# Terminology
# -----------
#
# The Quantify dataset is an xarray dataset that follows certain conventions. We define the following terminology:
#
# - **Experiment coordinate(s)**
#
#     - Xarray **Coordinates** whose names are specified in a list inside the dataset attributes under the key :attr:`~quantify_core.data.dataset_attrs.QExpCoordAttrs.experiment_coords`.
#     - Often correspond to physical coordinates, e.g., a signal frequency or amplitude.
#     - Often correspond to quantities set through :class:`~quantify_core.measurement.types.Settable`\s.
#
# - **Experiment variable(s)**
#
#     - Xarray **Variables** whose names are specified in a list inside the dataset attributes under the key :attr:`~quantify_core.data.dataset_attrs.QExpCoordAttrs.experiment_vars`.
#     - Often correspond to a physical quantity being measured, e.g., the signal magnitude at a specific frequency measured on a metal contact of a quantum chip.
#     - Often correspond to quantities returned by :class:`~quantify_core.measurement.types.Gettable`\s.
#
# .. note::
#
#     In this document we show exemplary datasets to highlight the details of the Quantify dataset specification.
#     However, for completness, we always show a valid Quantify dataset with all the required properties.
#
# In order to follow the rest of this specification more easily have a look at the example below.
# It should give you a more concrete feeling of the details that are exposed afterwards.

# %% [raw]
# .. admonition:: Generate dataset
#     :class: dropdown

# %%
# rst-json-conf: {"indent": "    "}

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

y0s = y0s / 2 + 0.5  # shift to 0-1 range
y1s = y1s / 2 + 0.5

dataset = dataset_2d_example = xr.Dataset(
    data_vars={
        pop_q0_par.name: (
            ("repetitions", "dim_0"),
            [y0s + y0s * np.random.uniform(-1, 1, y0s.shape) / k for k in (100, 10, 5)],
            mk_exp_var_attrs(
                **par_to_attrs(pop_q0_par),
                experiment_coords=[amp_par.name, time_par.name]
            ),
        ),
        pop_q1_par.name: (
            ("repetitions", "dim_0"),
            [y1s + y1s * np.random.uniform(-1, 1, y1s.shape) / k for k in (100, 10, 5)],
            mk_exp_var_attrs(
                **par_to_attrs(pop_q1_par),
                experiment_coords=[amp_par.name, time_par.name]
            ),
        ),
    },
    coords={
        amp_par.name: ("dim_0", x0s, mk_exp_coord_attrs(**par_to_attrs(amp_par))),
        time_par.name: ("dim_0", x1s, mk_exp_coord_attrs(**par_to_attrs(time_par))),
    },
    attrs=mk_dataset_attrs(main_dims=["dim_0"], repetitions_dims=["repetitions"]),
)

assert dataset == dataset_round_trip(dataset)  # confirm read/write

# %% [raw]
# .. admonition:: Quantify dataset: 2D example
#     :class: dropdown, toggle-shown
#
#     In the dataset below we have two experiment coordinates ``amp`` and ``time``; and two experiment variables ``pop_q0`` and ``pop_q1``.
#     Both experiment coordinates "lie" along one xarray dimension, ``dim_0``.
#     Both experiment variables lie along two xarray dimensions ``dim_0`` and ``repetitions_dim_0``.

# %%
# rst-json-conf: {"indent": "    "}

dataset

# %% [raw]
#     As seen above, in the Quantify dataset the experiment coordinates do not index the experiment variables because not all use-cases fit within this paradigm.
#     However, when possible, the Quantify dataset can be reshaped to take advantage of the xarray built-in utilities. Note, however, that this reshaping will produce an invalid Quantify dataset.

# %%
# rst-json-conf: {"indent": "    "}

dataset_gridded = dh.to_gridded_dataset(
    dataset_2d_example,
    dimension=dd.get_main_dims(dataset_2d_example)[0],
    coords_names=dd.get_experiment_coords(dataset_2d_example),
)
dataset_gridded.pop_q0.plot.pcolormesh(x="amp", col=dataset_gridded.pop_q0.dims[0])
_ = dataset_gridded.pop_q1.plot.pcolormesh(x="amp", col=dataset_gridded.pop_q1.dims[0])

# %% [raw]
#     In xarray, among other features, it is possible to average along a dimension which can be very convenient:

# %%
# rst-json-conf: {"indent": "    "}

_ = dataset_gridded.pop_q0.mean(dim=dataset_gridded.pop_q0.dims[0]).plot(x="amp")

# %% [raw]
# Detailed specification
# ----------------------

# %% [raw]
# Xarray dimensions
# ~~~~~~~~~~~~~~~~~

# %% [raw]
# The Quantify dataset has has the following required and optional dimensions:
#
# .. warning::
#
#     The specification below is a work-in-progress mentioning dimensions whose name follow the patterns ``dim_{i}``, ``repetition_dim_{i}`` and ``dim_{i}_cal``. However, the Quantify framework does not support yet ``i > 0``.
#
# - ``dim_{i}`` with ``i >= 0`` an integer
#
#     - **[Required]** ``dim_0``: The dataset must have at least one dimension named ``dim_0``.
#     - **[Optional]** ``dim_{i}``: Additional dimensions with the same name pattern are allowed for advanced use-cases.
#
#     - **[Optional, Advanced]** nested ``dim_{i}`` xarray dimensions.
#
#         - Intuition: intended primarily for time series, also known as "time trace" or simply trace.
#         - Nested ``dim_{i}`` xarray dimensions is allowed. I.e., **each entry** in, e.g., a ``y3`` experiment variable can be 1D, or nD array where each "D" has a corresponding ``dim_{i}`` xarray dimension.
#
# .. warning::
#
#     The advanced nested ``dim_{i}`` xarray dimensions are not supported yet in the Qunatify framework.
#
# - **[Optional]** ``repetition_dim_{i}`` with ``i >= 0`` an integer
#
#     - **Notes**
#
#         - Intuition for these xarray dimension: the equivalent would be to have ``dataset_reptition_0.hdf5``, ``dataset_reptition_1.hdf5``, etc. where each dataset was obtained from repeating exactly the same experiment. Instead we define an outer dimension for this.
#         - Default behavior of plotting tools will be to average the experiment variables along these dimension.
#
#     - The single (and only one) outermost dimension that the :ref:`experiment (and calibration) variables <sec-experiment-coordinates-and-variables>` can have.
#     - Any ``repetition_dim_{i}`` is only allowed to exist in the dataset if there is a corresponding ``dim_{i}`` dimension with the same ``i``.
#     - The :ref:`experiment variables <sec-experiment-coordinates-and-variables>` must lie along one (and only one) of these dimensions when more than one repetition of the experiement was performed.
#     - **[Optional]** The ``repetition_dim_{i}`` dimensions can be indexed by an optional xarray coordinate variable.
#
#         - **[Required]** The variable must be named ``repetition_dim_{i}`` as well.
#
#     - **[Required]** No other outer xarray dimensions are allowed.
#
# - **[Optional]** ``dim_{i}_cal`` with ``i >= 0`` an integer
#
#     - **Notes**
#
#         - Intended for calibration coordinates and variables that correspond to calibration data.
#         - The intention is to behave very similarly to ``dim_0`` but allow for array of different length compared to those that lie along ``dim_0``.
#
#     - **[Required]** Can only exist in the dataset if there is a corresponding ``dim_{i}``.
#     - **[Required]** Only calibration coordinates and variables are allowed to lie along a ``dim_{i}_cal`` dimension.
#     - **[Required]** Calibration coordinates and variables must lie along one and only one ``dim_{i}_cal`` dimension.

# %% [raw]
# .. admonition:: Examples datasets with repetition
#     :class: dropdown
#
#     As shown in the :ref:`xarray-intro` an xarray dimension can be indexed by a ``coordinate`` variable. In this example the ``repetitions`` dimension is indexed by the ``repetitions`` xarray coordinate variable:

# %%
# rst-json-conf: {"indent": "    "}

dataset = xr.Dataset(
    data_vars={
        pop_q0_par.name: (
            ("repetitions", "dim_0"),
            [y0s + np.random.random(y0s.shape) / k for k in (100, 10, 5)],
            mk_exp_var_attrs(
                **par_to_attrs(pop_q0_par),
                experiment_coords=[amp_par.name, time_par.name]
            ),
        ),
        pop_q1_par.name: (
            ("repetitions", "dim_0"),
            [y1s + np.random.random(y1s.shape) / k for k in (100, 10, 5)],
            mk_exp_var_attrs(
                **par_to_attrs(pop_q1_par),
                experiment_coords=[amp_par.name, time_par.name]
            ),
        ),
    },
    coords={
        amp_par.name: ("dim_0", x0s, mk_exp_coord_attrs(**par_to_attrs(amp_par))),
        time_par.name: ("dim_0", x1s, mk_exp_coord_attrs(**par_to_attrs(time_par))),
        # here we choose to index the repetition dimension with an array of strings
        "repetitions": (
            "repetitions",
            ["noisy", "very noisy", "very very noisy"],
        ),
    },
    attrs=mk_dataset_attrs(repetitions_dims=["repetitions"]),
)

assert dataset == dataset_round_trip(dataset)  # confirm read/write

dataset

# %%
# rst-json-conf: {"indent": "    "}

dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension=dd.get_main_dims(dataset)[0],
    coords_names=dd.get_experiment_coords(dataset),
)
dataset_gridded

# %% [raw]
#     It is now possible to retrieve (select) a specific entry along the ``repetitions`` dimension:

# %%
# rst-json-conf: {"indent": "    "}

_ = dataset_gridded.pop_q0.sel(repetitions="very noisy").plot(x="amp")

# %% [raw]
# Experiment coordinates
# ~~~~~~~~~~~~~~~~~~~~~~

# %% [raw]
# All the :ref:`experiment coordinates <sec-experiment-coordinates-and-variables>` in the dataset comply with:
#
# - **[Required]** Lie on at least one ``dim_{i}`` dimension.
#
#     - Usually equivalent to a settable, usually a parameter that an experimentalist "sweeps" in order to observe the effect on some other property of the system being studied.
#     - For some experiments it might not be suitable to think of a parameter that is being varied. In such cases, a coordinate, say ``point_idx``, can be simply an array of integers, e.g. ``np.linspace(0, number_of_points)``.

# %% [raw]
# Experiment variables
# ~~~~~~~~~~~~~~~~~~~~

# %% [raw]
# All the :ref:`experiment variables <sec-experiment-coordinates-and-variables>` in the dataset comply with:
#
# - **[Required]** Lie along at least one ``dim_{i}`` dimension.
# - **[Required]** Each entry is a data-point in the broad sense, i.e. it can be numeric (``int``/``float``/``complex``), fixed-lenght-string **OR** a nested ``numpy.ndarray`` (of one of these ``dtypes``).

# %% [raw]
# Dataset attributes
# ~~~~~~~~~~~~~~~~~~
#
# Tha mandatory attributes of the Quantify dataset are defined be the following dataclass.
# It can be used to generate a default dictionary that is attached to a dataset.
#
# .. autoclass:: quantify_core.data.dataset_attrs.QDatasetAttrs
#     :members:
#     :noindex:
#     :show-inheritance:
#
# .. autoclass:: quantify_core.data.dataset_attrs.QDatasetIntraRelationship
#     :members:
#     :noindex:
#     :show-inheritance:
#

# %%
from quantify_core.data.dataset_attrs import QDatasetAttrs

# tip: to_json and from_dict, from_json  are also available
dataset_2d_example.attrs = QDatasetAttrs().to_dict()
dataset_2d_example.attrs

# %% [raw]
# .. tip::
#
#     Note that xarray automatically provides the attributes as python attributes:

# %%
# rst-json-conf: {"indent": "    "}

dataset_2d_example.quantify_dataset_version, dataset_2d_example.tuid

# %% [raw]
# Experiment coordinates and variables attributes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% [raw]
# .. autoclass:: quantify_core.data.dataset_attrs.QExpCoordAttrs
#     :members:
#     :noindex:
#     :show-inheritance:

# %%
dataset_2d_example.amp.attrs


# %% [raw]
# .. autoclass:: quantify_core.data.dataset_attrs.QExpVarAttrs
#     :members:
#     :noindex:
#     :show-inheritance:

# %%
dataset_2d_example.pop_q0.attrs

# %% [raw]
# Storage format
# --------------
#
# The Quantify dataset is written to disk and loaded back making use of xarray-supported facilities.
# Internally we write to disk using:

# %%
# rst-json-conf: {"jupyter_execute_options": [":hide-code:"]}

import inspect
from IPython.display import Code

Code(inspect.getsource(dh.write_dataset), language="python")

# %% [raw]
# Note that we use the ``h5netcdf``` engine that is more permissive than the default NetCDF engine to accommodate for arrays of complex numbers type.
#
# .. admonition:: TODO
#     :class: warning
#
#     Furthermore, in order to support a variety of attribute types and shapes, in a seemless workflow, some additional tooling is required to be integrated. See sourcecodes below.

# %%
Code(inspect.getsource(dataset_round_trip), language="python")

# %%
Code(inspect.getsource(da.AdapterH5NETCDF), language="python")
