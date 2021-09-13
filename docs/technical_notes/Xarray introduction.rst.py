# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [raw]
# Xarray introduction
# ===================

# %% [raw]
# .. admonition:: Imports and auxiliary utilities
#     :class: dropdown

# %%
# rst-json-conf: {"indent": "    ", "jupyter_execute_options": [":hide-output:"]}

import numpy as np
import xarray as xr
from rich import pretty

pretty.install()

# %% [raw]
# Introduction
# ------------

# %% [raw]
# Xarray overview
# ~~~~~~~~~~~~~~~

# %% [raw]
# This subsection is a very brief overview of some concepts and functionalities of xarray. Here we use only pure xarray concepts and terminology.
#
# This is not intended as an extensive introduction to xarray. Please consult the :doc:`xarray documentation <xarray:index>` if you never used it before (it has very neat features!).
#
# There are different ways to create a new xarray dataset. Below we exemplify a few of them to showcase specific functionalities.
#
# An xarray dataset has **Dimensions** and **Variables**. Variables "lie" along at least one dimension:

# %%
n = 5
name_dim_a = "position_x"
name_dim_b = "velocity_x"
dataset = xr.Dataset(
    data_vars={
        "position": (  # variable name
            name_dim_a,  # dimension(s)' name(s)
            np.linspace(-5, 5, n),  # variable values
            {"units": "m", "long_name": "Position"},  # variable attributes
        ),
        "velocity": (
            name_dim_b,
            np.linspace(0, 10, n),
            {"units": "m/s", "long_name": "Velocity"},
        ),
    },
    attrs={"key": "my metadata"},
)
dataset

# %%
dataset.dims

# %%
dataset.variables

# %% [raw]
# A variable can be "promoted" to a **Coordinate** for its dimension(s):

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
    # coords={"position": (name_dim_a, position, {"units": "m", "long_name": "Position"})},
    attrs={"key": "my metadata"},
)
dataset = dataset.set_coords(
    ["position"]
)  # promote the position variable to a coordinate
dataset

# %%
dataset.coords["position"]

# %% [raw]
# Note that xarray coordinates are available as variables as well:

# %%
dataset.variables["position"]

# %% [raw]
# That on its own might not be very useful yet, however, xarray coordinates can be set to **index** other variables (:func:`~quantify_core.data.handling.to_gridded_dataset` does this under the hood), as shown below (note the bold font!):

# %%
dataset = dataset.set_index({"position_x": "position"})
dataset.position_x.attrs["units"] = "m"
dataset.position_x.attrs["long_name"] = "Position x"
dataset

# %% [raw]
# At this point the reader might get confused. In an attempt to clarify, we now have a dimension, a coordinate and a variable with the same name `"position_x"`.

# %%
dataset.dims

# %%
dataset.coords

# %%
dataset.variables["position_x"]

# %% [raw]
# Here the intention is to make the reader aware of this. Please consult the :doc:`xarray documentation <xarray:index>` for more details.
#
# An example of how this can be useful is to retrieve data from an xarray variable using one of its coordinates to select the desired entries:

# %% [raw]
# It is now possible to retrieve (select) a specific entry along the repetition dimension:

# %%
retrieved_value = dataset.velocity.sel(position_x=2.5)
retrieved_value

# %% [raw]
# Note that without this feature we would have to "manually" keep track of numpy integer indexes to retrieve the desired data:

# %%
dataset.velocity.values[3], retrieved_value.values == dataset.velocity.values[3]

# %% [raw]
# One of the great features of xarray is automatic plotting (explore the xarray documentation for more advanced capabilities!):

# %%
_ = dataset.velocity.plot(marker="o")

# %%
