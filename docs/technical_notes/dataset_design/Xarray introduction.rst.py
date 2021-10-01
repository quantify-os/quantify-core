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
# pylint: disable=pointless-statement
# pylint: disable=invalid-name

# %% [raw]
"""
.. _xarray-intro:

Xarray - brief introduction
===========================

.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Xarray introduction`

    :jupyter-download:script:`Xarray introduction`

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
# rst-json-conf: {"indent": "    ", "jupyter_execute_options": [":hide-output:"]}

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
name_dim_a = "position_x"
name_dim_b = "velocity_x"
dataset = xr.Dataset(
    data_vars={
        "position": (  # variable name
            name_dim_a,  # dimension's name
            np.linspace(-5, 5, n),  # values of this data variable
            # the "units" and "long_name" are a convention for automatic plotting
            {"units": "m", "long_name": "Position"},  # attributes of this data variable
        ),
        "velocity": (
            name_dim_b,
            np.linspace(0, 10, n),
            {"units": "m/s", "long_name": "Velocity"},
        ),
    },
    attrs={"key": "my metadata"},  # dataset attributes
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
position = np.linspace(-5, 5, n)
dataset = xr.Dataset(
    data_vars={
        "position": (name_dim_a, position, {"units": "m", "long_name": "Position"}),
        "velocity": (
            name_dim_a,
            1 + position ** 2,
            {"units": "m/s", "long_name": "Velocity"},
        ),
    },
    # We could add coordinates like this as well:
    # coords={
    #    "position": (name_dim_a, position, {"units": "m", "long_name": "Position"})
    # },
    attrs={"key": "my metadata"},
)

# Promote the position variable to a coordinate:
# In general, most of the functions that modify the structure of the xarray dataset will
# return a new object
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
dataset.position_x.attrs["units"] = "m"
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
Note the automatic labels and units.
"""
