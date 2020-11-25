# -----------------------------------------------------------------------------
# Description:    Module containing utilities for color manipulation
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------

import matplotlib.colors as mplc
import colorsys
from qcodes.plots.colors import color_cycle


def set_lightness(color, lightness):
    """
    Accepts a `matplotlib` color specification and return an rgb color
    with the specified `lightness`

    Args:
        color: e.g. "C0", "red", (40, 40, 40)
        lightness: value between 0 and 1

    Example:
        import seaborn as sns
        from matplotlib.colors import ListedColormap
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
        for col in color_cycle[1:]:
            ls = np.linspace(0., 1., 20)
            colors = [list(set_lightness(col, l)) for l in ls]
            all_colors += colors

        cmap = ListedColormap(all_colors)
        sns.palplot(all_colors)
    """
    rgb = mplc.to_rgb(color)
    hls_col = list(colorsys.rgb_to_hls(*mplc.to_rgb(rgb)))
    hls_col[1] = lightness
    return colorsys.hls_to_rgb(*hls_col)


# Create a "faded" version of the color_cycle
# NB pyqtgraph requires colors in [0, 255] range
darker_color_cycle, faded_color_cycle = tuple(tuple(
    tuple(int(255 * val) for val in set_lightness(color=col, lightness=lum))
    for col in color_cycle
) for lum in (0.4, 0.8))
