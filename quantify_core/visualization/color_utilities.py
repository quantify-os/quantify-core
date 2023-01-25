# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing utilities for color manipulation"""
from __future__ import annotations

import colorsys

import matplotlib.colors as mplc
import numpy as np


def set_hlsa(
    color,
    h: float | None = None,
    l: float | None = None,
    s: float | None = None,
    a: float | None = None,
    to_hex: bool = False,
) -> tuple:
    """
    Accepts a `matplotlib` color specification and returns an RGB color
    with the specified HLS values plus an optional alpha

    .. admonition:: Example
        :class: dropdown, tip

        In this example we use this function to create a custom colormap using several
        base colors for which we adjust the saturation and transparency (alpha,
        only visible when exporting the image).


        .. jupyter-execute::

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

                colors = [
                    list(set_hlsa(col, s=s)) for s, a in zip(sat_vals, alpha_vals)
                ]
                all_colors += colors

            cmap = mplc.ListedColormap(all_colors)

            np.random.seed(19680801)
            data = np.random.randn(30, 30)

            fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)

            psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
            fig.colorbar(psm, ax=ax)
            plt.show()
    """
    clip = lambda x: np.clip(x, 0, 1)
    rgb = mplc.to_rgb(color)
    hls = colorsys.rgb_to_hls(*mplc.to_rgb(rgb))
    new_hls = (old if new is None else clip(new) for old, new in zip(hls, (h, l, s)))
    col = colorsys.hls_to_rgb(*new_hls)

    # append alpha to tuple
    col = col if a is None else col + (clip(a),)
    # convert to int 255 range
    col = col if not to_hex else tuple(round(255 * x) for x in col)

    return col


def make_fadded_colors(
    num=5, color="#1f77b4", min_alpha=0.3, sat_power=2, to_hex=False
):
    hls = colorsys.rgb_to_hls(*mplc.to_rgb(mplc.to_rgb(color)))
    sat_vals = (np.linspace(1.0, 0.0, num) ** sat_power) * hls[2]
    alpha_vals = np.linspace(1.0, min_alpha, num)
    colors = tuple(
        set_hlsa(color, s=s, a=a, to_hex=to_hex) for s, a in zip(sat_vals, alpha_vals)
    )
    return colors
