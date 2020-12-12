# -----------------------------------------------------------------------------
# Description:    Module containing utilities for color manipulation
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------

import matplotlib.colors as mplc
import numpy as np
import colorsys


def clip(x, vmin=0.0, vmax=1.0):
    return max(min(x, vmax), vmin)


def set_hlsa(
    color,
    h: float = None,
    l: float = None,
    s: float = None,
    a: float = None,
    to_hex: bool = False,
):
    """
    Accepts a `matplotlib` color specification and returns an RGB color
    with the specified HLS values plus an optional alpha

    Example:

        import seaborn as sns
        from matplotlib.colors import ListedColormap
        import numpy as np

        color_cycle = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf'
        ]
        all_colors = []
        for col in color_cycle[:2]:
            hls = colorsys.rgb_to_hls(*mplc.to_rgb(mplc.to_rgb(col)))
            sat_vals = (np.linspace(0., 1.0, 20) ** 2) * hls[2]
            alpha_vals = np.linspace(0.4, 1.0, 20)

            colors = [list(set_hlsa(col, s=s)) for s, a in zip(sat_vals, alpha_vals)]
            all_colors += colors

        def plot_colors(colors):
            cmap = ListedColormap(colors)
            sns.palplot(colors)

        plot_colors(all_colors)
    """
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
