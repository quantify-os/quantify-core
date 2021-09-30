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
.. _dataset-spec:

Quantify dataset specification
==============================

.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Quantify dataset - specification`

    :jupyter-download:script:`Quantify dataset - specification`
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
from quantify_core.data import handling as dh
from quantify_core.measurement import grid_setpoints
from qcodes import ManualParameter
from rich import pretty
from pathlib import Path
from quantify_core.data.handling import set_datadir
import quantify_core.data.dataset_attrs as dd
import quantify_core.data.dataset_adapters as da
from quantify_core.utilities.examples_support import (
    mk_dataset_attrs,
    mk_main_coord_attrs,
    mk_main_var_attrs,
    round_trip_dataset,
    par_to_attrs,
)

pretty.install()

set_datadir(Path.home() / "quantify-data")  # change me!

# %% [raw]
"""
This document describes the Quantify dataset specification.
Here we focus on the concepts and terminology specific to the Quantify dataset.
It is based on the Xarray dataset, hence, we assume basic familiarity with the :class:`xarray.Dataset`.
If you are not familiar with it, we highly recommend to first have a look at our :ref:`xarray-intro` for a brief overview.
"""

# %% [raw]
"""
.. _sec-main-coordinates-and-variables:

Coordinates and Variables
-------------------------

The Quantify dataset is an xarray dataset that follows certain conventions. We define "subtypes" of xarray coordinates and variables:

.. _sec-main-coordinates:

Main coordinate(s)
^^^^^^^^^^^^^^^^^^

- Xarray **Coordinates** that have an attribute :attr:`~quantify_core.data.dataset_attrs.QCoordAttrs.is_main_coord` set to ``True``.
- Often correspond to physical coordinates, e.g., a signal frequency or amplitude.
- Often correspond to quantities set through :class:`~quantify_core.measurement.Settable`\s.
- See also the method :func:`~quantify_core.data.dataset_attrs.get_main_coords`.

.. _sec-secondary-coordinates:

Secondary coordinate(s)
^^^^^^^^^^^^^^^^^^^^^^^

- An ubiquitous example are the coordinates that are used by "calibration" points.
- Similar to `main coordinates <sec-main-coordinates>`_\, but intended to serve as the coordinates of `secondary variables <sec-secondary-variables>`_\.
- Xarray **Coordinates** that have an attribute :attr:`~quantify_core.data.dataset_attrs.QCoordAttrs.is_secondary_coord` set to ``True``.
- See also :func:`~quantify_core.data.dataset_attrs.get_secondary_coords`.

.. _sec-main-variables:

Main variable(s)
^^^^^^^^^^^^^^^^

- Xarray **Variables** that have an attribute :attr:`~quantify_core.data.dataset_attrs.QVarAttrs.is_main_var` set to ``True``.
- Must have an attribute :attr:`~quantify_core.data.dataset_attrs.QVarAttrs.coords` indicating the names of its coordinates (usually corresponding to 'physical' coordinates). This ensures that the main coordinates of main variables can be determined without ambiguity.

    - Example 1: If a signal ``y1`` was measured as a function of ``time`` and ``amplitude`` main coordinates, then we will have ``y1.attrs["main_coords"] = ["time", "amplitude"]``.
    - Example 2: In some cases, the idea of a coordinate does not apply, however a main coordinate in the dataset is required. A simple "index" coordinate should be used, e.g., an array of integers.

- Often correspond to a physical quantity being measured, e.g., the signal magnitude at a specific frequency measured on a metal contact of a quantum chip.
- Often correspond to quantities returned by :class:`~quantify_core.measurement.Gettable`\s.
- See also :func:`~quantify_core.data.dataset_attrs.get_main_vars`.

.. _sec-secondary-variables:

Secondary variables(s)
^^^^^^^^^^^^^^^^^^^^^^

- Again, the ubiquitous example are "calibration" datapoints.
- Similar to `main variables <sec-main-variables>`_, but intended to serve as reference data for other main variables (e.g., calibration data).
- Xarray **Variables** that have an attribute :attr:`~quantify_core.data.dataset_attrs.QVarAttrs.is_secondary_var` set to ``True``.
- The "assignment" of secondary variables to main variables should be done using :attr:`~quantify_core.data.dataset_attrs.QDatasetAttrs.relationships`.
- See also :func:`~quantify_core.data.dataset_attrs.get_secondary_vars`.


.. note::

    In this document we show exemplary datasets to highlight the details of the Quantify dataset specification.
    However, for completeness, we always show a valid Quantify dataset with all the required properties.

In order to follow the rest of this specification more easily have a look at the example below.
It should give you a more concrete feeling of the details that are exposed afterwards. See :ref:`sec-quantify-dataset-examples` for exemplary dataset.
"""

# %% [raw]
"""
.. admonition:: Generate dataset
    :class: dropdown
"""

# %%
# rst-json-conf: {"indent": "    "}

x0s = np.linspace(0.45, 0.55, 30)
x1s = np.linspace(0, 100e-9, 40)
time_par = ManualParameter(name="time", label="Time", unit="s")
amp_par = ManualParameter(name="amp", label="Flux amplitude", unit="V")
pop_q0_par = ManualParameter(name="pop_q0", label="Population Q0", unit="")
pop_q1_par = ManualParameter(name="pop_q1", label="Population Q1", unit="")

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
            mk_main_var_attrs(
                **par_to_attrs(pop_q0_par), coords=[amp_par.name, time_par.name]
            ),
        ),
        pop_q1_par.name: (
            ("repetitions", "dim_0"),
            [y1s + y1s * np.random.uniform(-1, 1, y1s.shape) / k for k in (100, 10, 5)],
            mk_main_var_attrs(
                **par_to_attrs(pop_q1_par), coords=[amp_par.name, time_par.name]
            ),
        ),
    },
    coords={
        amp_par.name: ("dim_0", x0s, mk_main_coord_attrs(**par_to_attrs(amp_par))),
        time_par.name: ("dim_0", x1s, mk_main_coord_attrs(**par_to_attrs(time_par))),
    },
    attrs=mk_dataset_attrs(main_dims=["dim_0"], repetitions_dims=["repetitions"]),
)

assert dataset == round_trip_dataset(dataset)  # confirm read/write

# %% [raw]
"""
.. admonition:: Quantify dataset: 2D example
    :class: dropdown, toggle-shown

    In the dataset below we have two main coordinates ``amp`` and ``time``; and two main variables ``pop_q0`` and ``pop_q1``.
    Both main coordinates "lie" along a single xarray dimension, ``dim_0``.
    Both main variables lie along two xarray dimensions ``dim_0`` and ``repetitions``.
"""

# %%
# rst-json-conf: {"indent": "    "}

dataset

# %% [raw]
"""
    As seen above, in the Quantify dataset the main coordinates do not index the main variables because not all use-cases fit within this paradigm.
    However, when possible, the Quantify dataset can be reshaped to take advantage of the xarray built-in utilities. Note, however, that this reshaping will produce an xarray dataset that does not comply with the Quantify dataset specification.
"""

# %%
# rst-json-conf: {"indent": "    "}

dataset_gridded = dh.to_gridded_dataset(
    dataset_2d_example,
    dimension=dd.get_main_dims(dataset_2d_example)[0],
    coords_names=dd.get_main_coords(dataset_2d_example),
)
dataset_gridded.pop_q0.plot.pcolormesh(x="amp", col=dataset_gridded.pop_q0.dims[0])
_ = dataset_gridded.pop_q1.plot.pcolormesh(x="amp", col=dataset_gridded.pop_q1.dims[0])

# %% [raw]
"""
Dimensions
----------
"""

# %% [raw]
"""
The main variables and coordinates present in a Quantify dataset have the following required and optional xarray dimensions:

.. _sec-repetitions-dimensions:

Repetitions dimension(s) [Optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Repetition dimensions comply with the following:

- Any dimensions present in the dataset that are listed in the :attr:`QDatasetAttrs.repetitions_dims <quantify_core.data.dataset_attrs.QDatasetAttrs.repetitions_dims>` dataset attribute.
- Intuition for these xarray dimension: the equivalent would be to have ``dataset_reptition_0.hdf5``, ``dataset_reptition_1.hdf5``, etc. where each dataset was obtained from repeating exactly the same experiment. Instead we define an outer dimension for this.
- Default behavior of (live) plotting and analysis tools can be to average the main variables along the repetitions dimension(s).
- Can be the outermost dimension of :ref:`experiment (and calibration) variables <sec-main-coordinates-and-variables>`.
- The :ref:`main variables <sec-main-coordinates-and-variables>` can lie along one (and only one) repetition dimension.

Main dimension(s) [Required]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main dimensions comply with the following:

- The outermost dimension of any main coordinate/variable, OR the second outermost dimension if the outermost one is a `repetitions dimension <sec-repetitions-dimensions>`_.
- Do not require to be explicitly specified in any metadata attributes, instead utilities for extracting them are provided. See :func:`~quantify_core.data.dataset_attrs.get_main_dims` which simply applies the rule above while inspecting all the main coordinates and variables present in the dataset.
- The dataset must have at least one main dimension.

.. admonition:: Note on nesting main dimensions

    Nesting main dimensions is allowed in principle and such examples are
    provided but it should be considered an experimental feature.

    - Intuition: intended primarily for time series, also known as "time trace" or simply trace. See :ref:`sec-quantify-dataset-examples` for an example.


Secondary dimension(s) [Optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Equivalent to the main dimensions but used by the secondary coordinates and variables.
The secondary dimensions comply with the following:

- The outermost dimension of any secondary coordinate/variable, OR the second outermost dimension if the outermost one is a `repetitions dimension <sec-repetitions-dimensions>`_.
- Do not require to be explicitly specified in any metadata attributes, instead utilities for extracting them are provided. See :func:`~quantify_core.data.dataset_attrs.get_secondary_dims` which simply applies the rule above while inspecting all the secondary coordinates and variables present in the dataset.
"""

# %% [raw]
"""
.. admonition:: Examples datasets with repetition
    :class: dropdown

    As shown in the :ref:`xarray-intro` an xarray dimension can be indexed by a ``coordinate`` variable. In this example the ``repetitions`` dimension is indexed by the ``repetitions`` xarray coordinate. Note that in an xarray dataset, a dimension and a data variables or a coordinate can share the same name. This might be confusing at first. It takes just a bit of dataset manipulation practice to gain the intuition for how it works.
"""

# %%
# rst-json-conf: {"indent": "    "}

dataset = xr.Dataset(
    data_vars={
        pop_q0_par.name: (
            ("repetitions", "dim_0"),
            [y0s + np.random.random(y0s.shape) / k for k in (100, 10, 5)],
            mk_main_var_attrs(
                **par_to_attrs(pop_q0_par), coords=[amp_par.name, time_par.name]
            ),
        ),
        pop_q1_par.name: (
            ("repetitions", "dim_0"),
            [y1s + np.random.random(y1s.shape) / k for k in (100, 10, 5)],
            mk_main_var_attrs(
                **par_to_attrs(pop_q1_par), coords=[amp_par.name, time_par.name]
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

assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %% [raw]
"""
    And as before, we can reshape the dataset to take advantage of the xarray builtin utilities.
"""

# %%
# rst-json-conf: {"indent": "    "}

dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension=dd.get_main_dims(dataset)[0],
    coords_names=dd.get_main_coords(dataset),
)
dataset_gridded

# %% [raw]
"""
    It is now possible to retrieve (select) a specific entry along the ``repetitions`` dimension:
"""

# %%
# rst-json-conf: {"indent": "    "}

_ = dataset_gridded.pop_q0.sel(repetitions="very noisy").plot(x="amp")
_ = dataset_gridded.pop_q0.sel(repetitions="very very noisy").plot(x="amp")

# %% [raw]
"""
Dataset attributes
------------------

The required attributes of the Quantify dataset are defined by the following dataclass.
It can be used to generate a default dictionary that is attached to a dataset under the :attr:`xarray.Dataset.attrs` attribute.

.. autoclass:: quantify_core.data.dataset_attrs.QDatasetAttrs
    :members:
    :noindex:
    :show-inheritance:

Additionally in order to express relationships between coordinates and/or variables the
the following template is provided:

.. autoclass:: quantify_core.data.dataset_attrs.QDatasetIntraRelationship
    :members:
    :noindex:
    :show-inheritance:
"""

# %%
from quantify_core.data.dataset_attrs import QDatasetAttrs

# tip: to_json and from_dict, from_json  are also available
dataset_2d_example.attrs = QDatasetAttrs().to_dict()
dataset_2d_example.attrs

# %% [raw]
"""
.. tip::

    Note that xarray automatically provides the entries of the dataset attributes as python attributes. And similarly for the xarray coordinates and data variables.
"""

# %%
# rst-json-conf: {"indent": "    "}

dataset_2d_example.quantify_dataset_version, dataset_2d_example.tuid

# %% [raw]
"""
Main coordinates and variables attributes
-----------------------------------------

Similar to the dataset attributes (:attr:`xarray.Dataset.attrs`), the main coordinates and variables have each their own required attributes attached to them as dictionary under the :attr:`xarray.DataArray.attrs` attribute.
"""

# %% [raw]
"""
.. autoclass:: quantify_core.data.dataset_attrs.QCoordAttrs
    :members:
    :noindex:
    :show-inheritance:
"""

# %%
dataset_2d_example.amp.attrs


# %% [raw]
"""
.. autoclass:: quantify_core.data.dataset_attrs.QVarAttrs
    :members:
    :noindex:
    :show-inheritance:
"""

# %%
dataset_2d_example.pop_q0.attrs

# %% [raw]
"""
Storage format
--------------

The Quantify dataset is written to disk and loaded back making use of xarray-supported facilities.
Internally we write and load to/from disk using:
"""

# %%
# rst-json-conf: {"jupyter_execute_options": [":hide-code:"]}

import inspect
from IPython.display import Code

Code(inspect.getsource(dh.write_dataset), language="python")

# %%
# rst-json-conf: {"jupyter_execute_options": [":hide-code:"]}

Code(inspect.getsource(dh.load_dataset), language="python")

# %% [raw]
"""
Note that we use the ``h5netcdf`` engine that is more permissive than the default NetCDF engine to accommodate for arrays of complex numbers.

.. note::

    Furthermore, in order to support a variety of attribute types (e.g. the `None` type) and shapes (e.g. nested dictionaries) in a seamless dataset round trip, some additional tooling is required. See source codes below that implements the two-way conversion adapter used by the functions shown above.
"""

# %%
Code(inspect.getsource(da.AdapterH5NetCDF), language="python")
