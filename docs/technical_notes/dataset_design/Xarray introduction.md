---
file_format: mystnb
kernelspec:
  name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---

(xarray-intro)=
# Xarray - brief introduction

```{seealso}
The complete source code of this tutorial can be found in

{nb-download}`Xarray introduction.ipynb`
```

The Quantify dataset is based on {doc}`Xarray <xarray:index>`.
This subsection is a very brief overview of some concepts and functionalities of xarray.
Here we use only pure xarray concepts and terminology.

This is not intended as an extensive introduction to xarray.
Please consult the {doc}`xarray documentation <xarray:index>` if you never used it
before (it has very neat features!).

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import xarray as xr
from rich import pretty

pretty.install()
```

There are different ways to create a new xarray dataset.
Below we exemplify a few of them to showcase specific functionalities.

An xarray dataset has **Dimensions** and **Variables**. Variables "lie" along at least
one dimension:

```{code-cell} ipython3
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
    attrs=dataset_attrs,
)  # dataset attributes
dataset
```

```{code-cell} ipython3
dataset.dims
```

```{code-cell} ipython3
dataset.variables
```

A variable can be "promoted" to (or defined as) a **Coordinate** for its dimension(s):

```{code-cell} ipython3
values_vel = 1 + values_pos**2
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
```

```{code-cell} ipython3
dataset.coords["position"]
```

Note that the xarray coordinates are available as variables as well:

```{code-cell} ipython3
dataset.variables["position"]
```

Which, on its own, might not be very useful yet, however, xarray coordinates can be set
to **index** other variables ({func}`~quantify_core.data.handling.to_gridded_dataset`
does this for the Quantify dataset), as shown below (note the bold font in the output!):

```{code-cell} ipython3
dataset = dataset.set_index({"position_x": "position"})
dataset.position_x.attrs["unit"] = "m"
dataset.position_x.attrs["long_name"] = "Position x"
dataset
```

At this point the reader might get very confused. In an attempt to clarify, we now have
a dimension, a coordinate and a variable with the same name `"position_x"`.

```{code-cell} ipython3
(
    "position_x" in dataset.dims,
    "position_x" in dataset.coords,
    "position_x" in dataset.variables,
)
```

```{code-cell} ipython3
dataset.dims["position_x"]
```

```{code-cell} ipython3
dataset.coords["position_x"]
```

```{code-cell} ipython3
dataset.variables["position_x"]
```

Here the intention is to make the reader aware of this peculiar behavior.
Please consult the {doc}`xarray documentation <xarray:index>` for more details.

An example of how this can be useful is to retrieve data from an xarray variable using
one of its coordinates to select the desired entries:

```{code-cell} ipython3
dataset.velocity
```

```{code-cell} ipython3
retrieved_value = dataset.velocity.sel(position_x=2.5)
retrieved_value
```

Note that without this feature we would have to keep track of numpy integer indexes to
retrieve the desired data:

```{code-cell} ipython3
dataset.velocity.values[3], retrieved_value.values == dataset.velocity.values[3]
```

One of the great features of xarray is automatic plotting (explore the xarray
documentation for more advanced capabilities!):

```{code-cell} ipython3
_ = dataset.velocity.plot(marker="o")
```

Note the automatic labels and unit.
