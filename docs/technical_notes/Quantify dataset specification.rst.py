#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# Quantify dataset specification
# ==============================

# %% [raw]
# .. warning:: Developement notes
#
#     We do not know yet if ``acq_set_{j}`` with ``j>0`` will be part of this specification (we lack some clear examples).

# %% [raw]
# .. note::
#
#     Along this page we show exemplary datasets to highlight the details of this specification.
#     However, keep in mind that we always show a valid dataset with all the required properties (except when exemplifying a bad dataset).
#
# .. admonition:: imports and auxiliary utilities
#     :class: dropdown

# %%
# notebook-to-rst-conf: {"indent": " " * 4, "jupyter_execute_options": [":hide-output:"]}

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from quantify_core.data import handling as dh
from quantify_core.measurement import grid_setpoints
from qcodes import ManualParameter


def assign_dataset_attrs(ds: xr.Dataset) -> dict:
    tuid = dh.gen_tuid()
    ds.attrs.update(
        {
            "grid_2d": True,  # necessary for live plotting
            "grid_2d_uniformly_spaced": True,  # pyqt requires interpolation
            "tuid": tuid,
            "quantify_dataset_version": "v1.0",
        }
    )
    return ds.attrs


def dataset_round_trip(ds: xr.Dataset) -> xr.Dataset:
    assign_dataset_attrs(ds)
    tuid = ds.attrs["tuid"]
    dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, ds)
    return dh.load_dataset(tuid)


def par_to_attrs(par):
    return {"units": par.unit, "long_name": par.label, "standard_name": par.name}


from pathlib import Path
from quantify_core.data.handling import get_datadir, set_datadir

set_datadir(Path.home() / "quantify-data")  # change me!

# %% [markdown]
# Introduction
# ------------

# %% [raw]
# Xarray overview
# ~~~~~~~~~~~~~~~

# %% [markdown]
# This is a brief overview of some concepts and functionalities of ``xarray`` that are leveraged to define the Quantify dataset.
#
# The dataset has **Dimensions** and **Variables**. Variables "lie" along at least one dimension:

# %%
n = 5
name_dim_a = "position_x"
name_dim_b = "velocity_x"
dataset = xr.Dataset(
    data_vars={
        "position": (name_dim_a, np.linspace(-5, 5, n), {"units": "m"}),
        "velocity": (name_dim_b, np.linspace(0, 10, n), {"units": "m/s"}),
    },
    attrs={"key": "my metadata"},
)
dataset

# %% [markdown]
# A variable can be set as coordinate for its dimension(s):

# %%
position = np.linspace(-5, 5, n)
dataset = xr.Dataset(
    data_vars={
        "position": (name_dim_a, position, {"units": "m"}),
        "velocity": (name_dim_a, 1 + position ** 2, {"units": "m/s"}),
    },
    attrs={"key": "my metadata"},
)
dataset = dataset.set_coords(["position"])
dataset

# %% [markdown]
# Xarray coordinates can be set to **index** other variables. (:func:`~quantify_core.data.handling.to_gridded_dataset` does this under the hood.)

# %%
dataset = dataset.set_index({"position_x": "position"})
dataset.position_x.attrs["units"] = "m"
dataset

# %% [markdown]
# An example of how this can be usefull:

# %%
dataset.velocity.sel(position_x=2.5)

# %% [markdown]
# Automatic plotting:

# %%
dataset.velocity.plot()

# %% [raw]
# .. _sec-experiment-coordinates-and-variables:
#
# Key dataset conventions
# ~~~~~~~~~~~~~~~~~~~~~~~

# %% [markdown]
# We define the following naming conventions in the Quantify dataset:
#
# - **Experiment coordinate(s)**
#     - ``xarray`` **Coordinates** following the naming convention ``f"x{i}"`` with ``i >= 0`` an integer.
#     - Often correspond to physical coordinates, e.g., a signal frequency or amplitude.
# - **Exeperiment variable(s)**
#     - ``xarray`` **Variables** following the naming convention ``f"y{i}"`` with ``i >= 0`` an integer.
#     - Often correspond to a physical quantity being measured, e.g., the signal magnitude at a specific frequency measured on a metal contact of a quantum chip.
#

# %% [raw]
# 2D Dataset example
# ~~~~~~~~~~~~~~~~~~

# %% [markdown]
# In the dataset below we have two experiment coordinates ``x0`` and ``x1``; and two experiment variables ``y0`` and ``y0``. Both experiment coordinates lie along one dimension, ``acq_set_0``. Both experiment variables lie along two dimensions ``acq_set_0`` and ``repetitions``.

# %% [raw]
# .. admonition:: Generate data
#     :class: dropdown

# %%
# notebook-to-rst-conf: {"indent": " " * 4}

x0s = np.linspace(0.45, 0.55, 30)
x1s = np.linspace(0, 100e-9, 40)
time_par = ManualParameter(name="time", label="Time", unit="s")
amp_par = ManualParameter(name="amp", label="Flux amplitude", unit="V")
pop_q0_par = ManualParameter(name="pop_q0", label="Population Q0", unit="arb. un.")
pop_q1_par = ManualParameter(name="pop_q1", label="Population Q1", unit="arb. un.")

x0s, x1s = grid_setpoints([x0s, x1s], [amp_par, time_par]).T
x0s_norm = np.abs((x0s - x0s.mean()) / (x0s - x0s.mean()).max())
y0s = (1 - x0s_norm) * np.sin(
    2 * np.pi * x1s * 1 / 30e-9 * (x0s_norm + 0.5)
)  # ~chevron
y1s = -y0s + 0.1

dataset = dataset_2d_example = xr.Dataset(
    data_vars={
        "y0": (("repetition", "acq_set_0"), [y0s], par_to_attrs(pop_q0_par)),
        "y1": (("repetition", "acq_set_0"), [y1s], par_to_attrs(pop_q1_par)),
    },
    coords={
        "x0": ("acq_set_0", x0s, par_to_attrs(amp_par)),
        "x1": ("acq_set_0", x1s, par_to_attrs(time_par)),
    },
)

assert dataset == dataset_round_trip(dataset)  # confirm read/write

# %%
dataset

# %% [markdown]
# As seen above, in the Quantify dataset the experiment coordinates do not index the experiment variables because not all use cases fit within this paradigm. However, when possible the dataset can be converted to take advange of the ``xarray`` built-in utlities.

# %%
dataset_gridded = dh.to_gridded_dataset(dataset, dimension="acq_set_0")
dataset_gridded.y0.plot(x="x0")
plt.show()
dataset_gridded.y1.plot(x="x0")
plt.show()

# %% [markdown]
# Detailed specification
# ----------------------

# %% [raw]
# Xarray dimensions
# ~~~~~~~~~~~~~~~~~

# %% [markdown]
# The Quantify dataset has has the following required and optional dimensions:
#
# - **[Required]** ``repetition``
#
#     - The outermost dimension of the :ref:`experiment variables <sec-experiment-coordinates-and-variables>`.
#     - Intuition for this ``xarray`` dimension: the equivalent would be to have ``dataset_reptition_0.hdf5``, ``dataset_reptition_1.hdf5``, etc. where each dataset was obtained from repeating exactly the same experiment. Instead we define an outer dimension for this.
#     - Default behavior of plotting tools will be to average the dataset along this dimension.
#     - The :ref:`experiment variables <sec-experiment-coordinates-and-variables>` must lie along this dimension (even when only one repetition of the experiment was executed).
#     - **[Optional]** The ``repetition`` dimension can be indexed by an optional ``xarray`` coordinate variable.
#
#         - **[Required]** The variable must be named ``repetition`` as well.
#
#     - **[Required]** no other outer ``xarray`` dimensions allowed.
#
#

# %% [raw]
# .. admonition:: Examples good datasets (repetition)
#     :class: dropdown
#
#     To be added:
#
#     - More than one repetitions.
#     - ``repetition`` dimensions indexed by a ``coordinate`` variables.

# %% [raw]
# .. admonition:: Examples bad datasets (repetition)
#     :class: dropdown
#
#      To be added:
#
#     - No repetition dimension.
#     - An outer dimension.

# %% [markdown]
# - **[Required]** ``acq_set_0``
#
#     - The outermost dimension of the :ref:`experiment coordinates <sec-experiment-coordinates-and-variables>`.
#     - The first inner dimension of the :ref:`experiment variables <sec-experiment-coordinates-and-variables>` (the outermost is the ``repetition`` dimension).
#

# %% [raw]
# .. admonition:: Examples good datasets (acq_set_0)
#     :class: dropdown

# %%
# notebook-to-rst-conf: {"indent": " " * 4}

dataset_2d_example

# %% [raw]
# .. admonition:: Examples bad datasets (acq_set_0)
#     :class: dropdown
#
#     To be added:
#
#     - `x0` and `y0` with some other dimension then ``acq_set_0``.

# %% [markdown]
#
# - **[Optional, Advanced]** other nested ``xarray`` dimensions under each ``acq_set_{i}``
#
#     - Intuition: intended primarily for time series, also known as "time trace" or simply trace.
#     - Other, potentially arbitrarily nested, ``xarray`` dimensions under each ``acq_set_{i}`` is allowed. I.e., **each entry** in a, e.g., ``y3`` ``xarray`` variable can be a 1D, or nD array where each "D" has a corresponding ``xarray`` dimension.
#     - Such ``xarray`` dimensions can be named arbitrarily.
#     - Each of such ``xarray`` dimension can be *indexed* by an ``xarray`` coordinate variable. E.g. for a time trace we would have in the dataset:
#
#         - ``assert "time" in dataset.coords``
#         - ``assert "time" in dataset.dims``
#         - ``assert len(dataset.time) == len(dataset.y3.isel(repetition=0, acq_set_0=0))`` where ``y3`` is a measured variable storing traces.
#
#     - Note: When nesting data like this, it is required to have "hyper-cubic"-shaped data, meaning that e.g. ``dataset.y3.isel(repetition=0, acq_set_0=0) == [[2], [ 5, 6]]`` is not possible, but ``dataset.y3.isel(repetition=0, acq_set_0=0) == [[2, 3], [5, 6]]`` is. This is a direct consequence of numpy ``ndarray`` (with entries of type ``int``/``float``/``complex``).
#

# %% [raw]
# .. admonition:: Examples good datasets (other nested dimensions)
#     :class: dropdown
#
#     To be added:
#
#     - time series example
#     - time series example with complex data
#     - Fictitious examples, does not necessarily repretime series with a few distinct DACs, where the DACs names index an extra dimension.
#

# %% [raw]
# .. admonition:: Examples bad datasets (other nested dimensions)
#     :class: dropdown
#
#     To be added:
#
#     - ``time`` coordinate is not indexing the ``time`` dimension.
#

# %% [raw]
# .. admonition:: To be refined (acq_set_{i})
#     :class: dropdown, warning
#
#     For reference from earlier dsicussion, requires some good example to justify this:
#
#     - **[Optional, Advanced]** ``acq_set_{i}``, where ``i`` > 0 is an integer.
#
#     - Reserves the possibility to store data for experiments that we have not yet encountered ourselves. I a gut feeling that we need this, but might not have a good realistic example, some help here is welcome.
#
#         - (Example ?) Imagine measuring some qubits until all of them are in a desired state, returning the data of these measurements and then proceeding to doing the "real" experiment you are interested in. I think having these extra *independent* ``xarray`` dimensions
#     - **[Required]** all ``acq_set_{i}`` dimensions (including ``acq_set_0``) are mutually excluding. This means variables in the dataset cannot depend on more than one of these dimensions.
#
#         - **Bad** variable: ``y0(repetition, acq_set_0, acq_set_1)``, this should never happen in the dataset.
#         - **Good** variable: ``y0(repetition, acq_set_0)`` or ``y1(repetition, acq_set_1)``.
#

# %% [raw]
# Xarray coordinates (variables)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% [markdown]
# Only the following `xarray` coordinates are allowed in the dataset:
#
# - **[Required]** The ``x0`` :ref:`experiment coordinate <sec-experiment-coordinates-and-variables>`.
#     - Usually equivalent to a settable, usually a parameter that an experimentalist "sweeps" in order to observe the effect on some other property of the system being studied.
#     - For some experiments it might not be suitable to think of a parameter that is being varied. In such cases ``x0`` can be simply an array of integers, e.g. ``np.linspace(0, number_of_points)``.
# - **[Optional]** Other ``f"x{i}"`` :ref:`experiment coordinates <sec-experiment-coordinates-and-variables>`, with ``i`` a positive integer.
#
#     - These are the coordinates that index the :ref:`experiment variables <sec-experiment-coordinates-and-variables>`. This indexing can be made explicit in a (separate) :class:`xarray.Dataset` instance retuned by `quantify_core.data.handling.to_gridded_dataset()` (when the data corresponds to a multi-dimensional grid).
#     - **[Required]** Each ``x{i}`` must lie along one (and only one) ``acq_set_{j}`` ``xarray`` dimension.
# - **[Optional]** Other ``xarray`` coordinates (that are not :ref:`experiment coordinates <sec-experiment-coordinates-and-variables>`) used to index the nested dimensions.
#
#     - Allowed dimension names:
#         - ``repetition``, or
#         - ``acq_set_{i}``, or
#         - ``<arbitrary_name>`` but with the same name as one of the **nested** dimensions (see :ref:`Xarray dimensions` section above).
#     - **[Required]** These other ``xarray`` coordinates must "lie" along a single dimension (and have the same name).
#

# %% [raw]
# .. admonition:: Examples good datasets (coordinates)
#     :class: dropdown
#
#     To be added...

# %% [raw]
# Xarray data variables
# ~~~~~~~~~~~~~~~~~~~~~

# %% [markdown]
# The only ``xarray`` data variables allowed in the dataset are the :ref:`experiment variables <sec-experiment-coordinates-and-variables>`. Each entry in one of these experiment variables is a data-point in the broad sense, i.e. it can be ``int``/``float``/``complex`` **OR** a nested ``numpy.ndarray`` (of one of these ``dtypes``).
#
# All the ``xarray`` data variables in the dataset (that are not ``xarray`` coordinates) comply with:
# - Naming:
#     - ``y{i}`` where  is an integer; **OR**
#     - ``y{i}_<arbitrary>`` where ``i => 0`` is an integer such that matches an existing ``y{i}`` in the same dataset.
#         - This is intended to denote a meaningful connection between ``y{i}`` and ``y{i}_<arbitrary>``.
#         - **[Required]** The number of elements in``y{i}`` and ``y{i}_<arbitrary>`` must be the same along the ``acq_set_{j}`` dimension.
#         - E.g., the digitized time traces stored in ``y0_trace(repetition, acq_set_0, time)`` and the demodulated values ``y0(repetition, acq_set_0)`` represent the same measurement with different levels of detail.
#     - Rationale: facilitates inspecting and processing the dataset in an intuitive way.
# - **[Required]** Lie along at least the ``repetition`` and ``acq_set_{i}`` dimensions.
# - **[Optional]** Lie along additional nested ``xarray`` dimensions.
#

# %% [raw]
# .. admonition:: Examples good datasets (variables)
#     :class: dropdown
#
#     To be added...
#
#     - ``y0_trace(repetition, acq_set_0, time)`` and the demodulated values ``y0(repetition, acq_set_0)``
#

# %% [markdown]
# Dataset with two ``y{i}``:

# %%
# notebook-to-rst-conf: {"indent": " " * 4}

dataset_2d_example

# %% [raw]
# Dataset attributes
# ~~~~~~~~~~~~~~~~~~

# %% [markdown]
#

# %% [raw]
# Variables attributes
# ~~~~~~~~~~~~~~~~~~~~

# %% [markdown]
#

# %% [raw]
# Calibration variables and dimensions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% [markdown]
# Calibration points can be tricky to deal with. As an addtion to the specification above we describe here how and which kind of calibration points are supported within the Qunatify dataset.
#
# Calibration points are stored as ``xarray`` data variables. We shall refer to them as *calibration variables*. They are similar to the experiment variables with the following differences:
#
# - They are ``xarray`` data variables named as ``y{j}_calib``.
# - They must lie along the ``acq_set_{i}_calib``, i.e. ``y{j}_calib(repetition, acq_set_{i}_calib, <other nested dimension(s)>)``.
#     - Note that we would have ``y{j}(repetition, acq_set_{i}, <other nested dimension(s)>)``.
# - ``y{i}_<arbitrary>_calib`` must be also present if both ``y{i}_calib`` and ``y{i}_<arbitary>`` are present in the dataset.
#
# .. note::
#
#     The number of elements in ``y{j}`` and ``y{j}_calib`` are indepenent. Usually there are only a few calibration points.
#

# %% [raw]
# .. admonition:: Examples good datasets (variables)
#     :class: dropdown
#
#     To be added...
#
#     - T1 with calibration points.
#     - T1 with calibration points and raw traces inlcuded also for the calibration points.
#

# %%
