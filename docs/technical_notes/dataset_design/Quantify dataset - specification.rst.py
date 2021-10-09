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
.. _dataset-spec:

Quantify dataset specification
==============================

.. seealso::

    The complete source code of this tutorial can be found in

    .. NB .py is from notebook_to_sphinx_extension

    :jupyter-download:notebook:`Quantify dataset - specification.py`

    :jupyter-download:script:`Quantify dataset - specification.py`
"""

# %% [raw]
"""
.. admonition:: Imports and auxiliary utilities
    :class: dropdown
"""

# %%
rst_json_conf = {"indent": "    "}

import xarray as xr
import matplotlib.pyplot as plt
from quantify_core.data import handling as dh
from rich import pretty
from pathlib import Path
from quantify_core.utilities import dataset_examples
import quantify_core.data.dataset_attrs as dattrs
import quantify_core.data.dataset_adapters as dadapters
from quantify_core.utilities.examples_support import round_trip_dataset
from quantify_core.utilities.inspect_utils import display_source_code

pretty.install()

dh.set_datadir(Path.home() / "quantify-data")  # change me!

# %% [raw]
"""
This document describes the Quantify dataset specification.
Here we focus on the concepts and terminology specific to the Quantify dataset.
It is based on the Xarray dataset, hence, we assume basic familiarity with the :class:`xarray.Dataset`.
If you are not familiar with it, we highly recommend to first have a look at our :ref:`xarray-intro` for a brief overview.
"""

# %% [raw]
r"""
.. _sec-coordinates-and-variables:

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

    We use the :func:`~quantify_core.utilities.dataset_examples.mk_two_qubit_chevron_dataset` to generate our dataset.
"""

# %%
rst_json_conf = {"indent": "    "}

display_source_code(dataset_examples.mk_two_qubit_chevron_dataset)

# %%
rst_json_conf = {"indent": "    "}

dataset = dataset_examples.mk_two_qubit_chevron_dataset()

assert dataset == round_trip_dataset(dataset)  # confirm read/write

# %% [raw]
"""
.. admonition:: Quantify dataset: 2D example
    :class: dropdown, toggle-shown

    In the dataset below we have two main coordinates ``amp`` and ``time``; and two main variables ``pop_q0`` and ``pop_q1``.
    Both main coordinates "lie" along a single xarray dimension, ``main_dim``.
    Both main variables lie along two xarray dimensions ``main_dim`` and ``repetitions``.
"""

# %%
rst_json_conf = {"indent": "    "}

dataset

# %% [raw]
"""
    As seen above, in the Quantify dataset the main coordinates do not index the main variables because not all use-cases fit within this paradigm.
    However, when possible, the Quantify dataset can be reshaped to take advantage of the xarray built-in utilities. Note, however, that this reshaping will produce an xarray dataset that does not comply with the Quantify dataset specification.
"""

# %%
rst_json_conf = {"indent": "    "}

dataset_gridded = dh.to_gridded_dataset(
    dataset,
    dimension="main_dim",
    coords_names=dattrs.get_main_coords(dataset),
)
dataset_gridded.pop_q0.plot.pcolormesh(x="amp", col="repetitions")
_ = dataset_gridded.pop_q1.plot.pcolormesh(x="amp", col="repetitions")

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
- Can be the outermost dimension of the main (and secondary) variables.
- The main variables can lie along one (and only one) repetition dimension.

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
rst_json_conf = {"indent": "    "}

coord_name = "repetitions"
coord_dims = ("repetitions",)
coord_values = ["A", "B", "C", "D", "E"]
dataset_indexed_rep = xr.Dataset(
    coords={coord_name: (coord_dims, coord_values)},
    attrs=dict(repetitions_dims=["repetitions"]),
)

dataset_indexed_rep

# %%
rst_json_conf = {"indent": "    "}

# merge with the previous dataset
dataset_rep = dataset.merge(dataset_indexed_rep, combine_attrs="drop_conflicts")

assert dataset_rep == round_trip_dataset(dataset_rep)  # confirm read/write

dataset_rep

# %% [raw]
"""
    And as before, we can reshape the dataset to take advantage of the xarray builtin utilities.
"""

# %%
rst_json_conf = {"indent": "    "}

dataset_gridded = dh.to_gridded_dataset(
    dataset_rep,
    dimension="main_dim",
    coords_names=dattrs.get_main_coords(dataset),
)
dataset_gridded

# %% [raw]
"""
    It is now possible to retrieve (select) a specific entry along the ``repetitions`` dimension:
"""

# %%
rst_json_conf = {"indent": "    "}
_ = dataset_gridded.pop_q0.sel(repetitions="A").plot(x="amp")
plt.show()
_ = dataset_gridded.pop_q0.sel(repetitions="D").plot(x="amp")

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
dataset.attrs = QDatasetAttrs().to_dict()
dataset.attrs

# %% [raw]
"""
.. tip::

    Note that xarray automatically provides the entries of the dataset attributes as python attributes. And similarly for the xarray coordinates and data variables.
"""

# %%
rst_json_conf = {"indent": "    "}

dataset.quantify_dataset_version, dataset.tuid

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
dataset.amp.attrs


# %% [raw]
"""
.. autoclass:: quantify_core.data.dataset_attrs.QVarAttrs
    :members:
    :noindex:
    :show-inheritance:
"""

# %%
dataset.pop_q0.attrs

# %% [raw]
"""
Storage format
--------------

The Quantify dataset is written to disk and loaded back making use of xarray-supported facilities.
Internally we write and load to/from disk using:
"""

# %%
display_source_code(dh.write_dataset)
display_source_code(dh.load_dataset)


# %% [raw]
"""
Note that we use the ``h5netcdf`` engine that is more permissive than the default NetCDF engine to accommodate for arrays of complex numbers.

.. note::

    Furthermore, in order to support a variety of attribute types (e.g. the `None` type) and shapes (e.g. nested dictionaries) in a seamless dataset round trip, some additional tooling is required. See source codes below that implements the two-way conversion adapter used by the functions shown above.
"""

# %%
display_source_code(dadapters.AdapterH5NetCDF)
