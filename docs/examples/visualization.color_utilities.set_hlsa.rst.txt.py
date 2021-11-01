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

    In this example we use this function to create a custom colormap using several base colors for which we adjust the saturation and transparency (alpha, only visible when exporting the image).
"""

# %%
rst_conf = {"indent": "    "}

import colorsys

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np

from quantify_core.visualization.color_utilities import set_hlsa

color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
all_colors = []
for col in color_cycle:
    hls = colorsys.rgb_to_hls(*mplc.to_rgb(mplc.to_rgb(col)))
    sat_vals = (np.linspace(0.0, 1.0, 20) ** 2) * hls[2]
    alpha_vals = np.linspace(0.4, 1.0, 20)

    colors = [list(set_hlsa(col, s=s)) for s, a in zip(sat_vals, alpha_vals)]
    all_colors += colors

cmap = mplc.ListedColormap(all_colors)

np.random.seed(19680801)
data = np.random.randn(30, 30)

fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)

psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
fig.colorbar(psm, ax=ax)
