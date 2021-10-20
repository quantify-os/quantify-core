# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Module containing utilities for color manipulation"""
import colorsys

import matplotlib.colors as mplc
import numpy as np


def set_hlsa(
    color,
    h: float = None,
    l: float = None,
    s: float = None,
    a: float = None,
    to_hex: bool = False,
) -> tuple:
    """
    Accepts a `matplotlib` color specification and returns an RGB color
    with the specified HLS values plus an optional alpha


    .. include:: examples/visualization.color_utilities.set_hlsa.py.rst.txt
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
