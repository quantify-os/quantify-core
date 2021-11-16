# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import tempfile

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import quantify_core.data.handling as dh
from quantify_core.visualization.mpl_plotting import set_cyclic_colormap

dh.set_datadir(tempfile.TemporaryDirectory().name)


def test_set_cyclic_colormap() -> None:
    zvals = xr.DataArray(np.random.rand(6, 10) * 360.0)
    zvals.attrs["units"] = "deg"

    _, ax = plt.subplots(1, 1)
    color_plot = zvals.plot(ax=ax)
    set_cyclic_colormap(color_plot)
    assert color_plot.get_clim() == (0.0, 360.0)
    assert color_plot.get_cmap().name == "twilight"

    zvals_shifted = zvals - 180.0
    _, ax = plt.subplots(1, 1)
    color_plot = zvals_shifted.plot(ax=ax)
    ax.set_title("Shifted cyclic colormap")
    set_cyclic_colormap(color_plot, shifted=zvals_shifted.min() < 0)
    assert color_plot.get_clim() == (-180.0, 180.0)
    assert color_plot.get_cmap().name == "twilight_shifted"

    _, ax = plt.subplots(1, 1)
    color_plot = (zvals / 2).plot(ax=ax)
    ax.set_title("Overwrite clim")
    set_cyclic_colormap(color_plot, clim=(0.0, 180.0), unit="deg")
    assert color_plot.get_clim() == (0.0, 180.0)
    assert color_plot.get_cmap().name == "twilight"

    _, ax = plt.subplots(1, 1)
    zvals_rad = zvals / 180 * np.pi - np.pi
    zvals_rad.attrs["units"] = "rad"
    color_plot = zvals_rad.plot(ax=ax)
    ax.set_title("Radians")
    set_cyclic_colormap(
        color_plot, shifted=zvals_shifted.min() < 0, unit=zvals_rad.units
    )
    assert color_plot.get_clim() == (-np.pi, np.pi)
    assert color_plot.get_cmap().name == "twilight_shifted"
