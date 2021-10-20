# ---
# jupyter:
#   jupytext:
#     cell_markers: \"\"\"
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
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
# pylint: disable=duplicate-code


# %% [raw]
"""
.. admonition:: Example
    :class: dropdown, tip
"""

# %%
rst_conf = {"indent": "    "}

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from quantify_core.visualization.mpl_plotting import set_cyclic_colormap

zvals = xr.DataArray(np.random.rand(6, 10) * 360)
zvals.attrs["units"] = "deg"
zvals.plot()

fig, ax = plt.subplots(1, 1)
color_plot = zvals.plot(ax=ax)
set_cyclic_colormap(color_plot)

zvals_shifted = zvals - 180

fig, ax = plt.subplots(1, 1)
color_plot = zvals_shifted.plot(ax=ax)
ax.set_title("Shifted cyclic colormap")
set_cyclic_colormap(color_plot, shifted=zvals_shifted.min() < 0)

fig, ax = plt.subplots(1, 1)
color_plot = (zvals / 2).plot(ax=ax)
ax.set_title("Overwrite clim")
set_cyclic_colormap(color_plot, clim=(0, 180), unit="deg")

fig, ax = plt.subplots(1, 1)
zvals_rad = zvals / 180 * np.pi
zvals_rad.attrs["units"] = "rad"
color_plot = zvals_rad.plot(ax=ax)
ax.set_title("Radians")
set_cyclic_colormap(color_plot, unit=zvals_rad.units)
