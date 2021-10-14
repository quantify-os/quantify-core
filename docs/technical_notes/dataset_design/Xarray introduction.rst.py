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
# pylint: disable=duplicate-code

# %% [raw]
"""
.. _xarray-intro:

Xarray - brief introduction
===========================

.. seealso::

    The complete source code of this tutorial can be found in

    .. NB .py is from notebook_to_sphinx_extension

    :jupyter-download:notebook:`Xarray introduction.py`

    :jupyter-download:script:`Xarray introduction.py`

The Quantify dataset is based on :doc:`Xarray <xarray:index>`.
This subsection is a very brief overview of some concepts and functionalities of xarray.
Here we use only pure xarray concepts and terminology.

This is not intended as an extensive introduction to xarray.
Please consult the :doc:`xarray documentation <xarray:index>` if you never used it
before (it has very neat features!).
"""

# %% [raw]
"""
.. admonition:: Imports and auxiliary utilities
    :class: dropdown
"""

# %%
rst_conf = {"indent": "    "}

import numpy as np
import xarray as xr
from rich import pretty

pretty.install()

# %% [raw]
"""
There are different ways to create a new xarray dataset.
Below we exemplify a few of them to showcase specific functionalities.

An xarray dataset has **Dimensions** and **Variables**. Variables "lie" along at least
one dimension:
"""

# %%
n = 5

values_pos = np.linspace(-5, 5, n)
dimensions_pos = ("position_x",)
# the "unit" and "long_name" are a convention for automatic plotting
attrs_pos = dict(unit="m", long_name="Position")  # attributes of this data variable

values_vel = np.linspace(0, 10, n)
dimensions_vel = ("velocity_x",)
attrs_vel = dict(unit="m/s", long_name="Velocity")

data_vars = dict(
    position=(dimensions_pos, values_pos, attrs_pos),
    velocity=(dimensions_vel, values_vel, attrs_vel),
)

dataset_attrs = dict(my_attribute_name="some meta information")

dataset = xr.Dataset(
    data_vars=data_vars,
    attrs=dataset_attrs,  # dataset attributes
)
dataset

# %%
dataset.dims

# %%
dataset.variables

# %% [raw]
"""
A variable can be "promoted" to (or defined as) a **Coordinate** for its dimension(s):
"""

# %%
values_vel = 1 + values_pos ** 2
data_vars = dict(
    position=(dimensions_pos, values_pos, attrs_pos),
    # now the velocity array "lies" along the same dimension as the position array
    velocity=(dimensions_pos, values_vel, attrs_vel),
)
dataset = xr.Dataset(
    data_vars=data_vars,
    # NB We could set "position" as a coordinate directly when creating the dataset:
    # coords=dict(position=(dimensions_pos, values_pos, attrs_pos)),
    attrs=dataset_attrs,
)

# Promote the "position" variable to a coordinate:
# In general, most of the functions that modify the structure of the xarray dataset will
# return a new object, hence the assignment
dataset = dataset.set_coords(["position"])
dataset

# %%
dataset.coords["position"]

# %% [raw]
"""
Note that the xarray coordinates are available as variables as well:
"""

# %%
dataset.variables["position"]

# %% [raw]
"""
Which, on its own, might not be very useful yet, however, xarray coordinates can be set
to **index** other variables (:func:`~quantify_core.data.handling.to_gridded_dataset`
does this for the Quantify dataset), as shown below (note the bold font in the output!):
"""

# %%
dataset = dataset.set_index({"position_x": "position"})
dataset.position_x.attrs["unit"] = "m"
dataset.position_x.attrs["long_name"] = "Position x"
dataset

# %% [raw]
"""
At this point the reader might get very confused. In an attempt to clarify, we now have
a dimension, a coordinate and a variable with the same name `"position_x"`.
"""

# %%
(
    "position_x" in dataset.dims,
    "position_x" in dataset.coords,
    "position_x" in dataset.variables,
)

# %%
dataset.dims["position_x"]

# %%
dataset.coords["position_x"]

# %%
dataset.variables["position_x"]

# %% [raw]
"""
Here the intention is to make the reader aware of this peculiar behavior.
Please consult the :doc:`xarray documentation <xarray:index>` for more details.

An example of how this can be useful is to retrieve data from an xarray variable using
one of its coordinates to select the desired entries:
"""

# %%
dataset.velocity

# %%
retrieved_value = dataset.velocity.sel(position_x=2.5)
retrieved_value

# %% [raw]
"""
Note that without this feature we would have to keep track of numpy integer indexes to
retrieve the desired data:
"""

# %%
dataset.velocity.values[3], retrieved_value.values == dataset.velocity.values[3]

# %% [raw]
"""
One of the great features of xarray is automatic plotting (explore the xarray
documentation for more advanced capabilities!):
"""

# %%
_ = dataset.velocity.plot(marker="o")

# %% [raw]
"""
Note the automatic labels and unit.
"""
