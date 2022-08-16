# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import tempfile

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import quantify_core.data.handling as dh
from quantify_core.analysis.fitting_models import CosineModel
from quantify_core.visualization.mpl_plotting import set_cyclic_colormap, plot_fit

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


def test_plot_fit() -> None:

    # Create some sinusoid data arrays
    noise_ampl = 0.05
    x = np.linspace(0, 2 * np.pi, 25)
    cos_data = np.cos(x) + noise_ampl * np.random.uniform(-1, 1, len(x))
    sin_data = np.sin(x) + noise_ampl * np.random.uniform(-1, 1, len(x))

    # Do the cosine fits
    model = CosineModel()

    cos_guess = model.guess(cos_data, x=x)
    sin_guess = model.guess(sin_data, x=x)

    cos_result = model.fit(cos_data, x=x, params=cos_guess)
    sin_result = model.fit(sin_data, x=x, params=sin_guess)

    # Plot only cos data with cos result to test default behaviour
    fig_default, ax_default = plt.subplots()
    (cos_data_default,) = ax_default.plot(x, cos_data, ".")
    cos_fit_default, cos_guess_default = plot_fit(ax_default, cos_result)
    legend_default = ax_default.legend()

    assert isinstance(cos_fit_default, plt.Line2D)
    assert cos_fit_default.get_color() == "C3"
    assert len(cos_fit_default.get_xdata()) == 1000

    assert isinstance(cos_guess_default, plt.Line2D)
    assert cos_guess_default.get_color() == "grey"
    assert len(cos_guess_default.get_xdata()) == 1000

    handles, labels = ax_default.get_legend_handles_labels()
    assert handles == [cos_fit_default, cos_guess_default]
    assert labels == ["Fit", "Guess"]

    # Plot both cos and sin data and results to test customisable behaviour
    fig_custom, ax_custom = plt.subplots()
    (cos_data_custom,) = ax_custom.plot(x, cos_data, ".", c="C1")
    (sin_data_custom,) = ax_custom.plot(x, sin_data, ".", c="C2")

    # Short matplotlib keywords no init plot
    fit_kwargs_cos = {
        "c": cos_data_custom.get_color(),
        "ls": "--",
        "label": "cos result",
    }
    (cos_fit_custom,) = plot_fit(
        ax=ax_custom,
        fit_res=cos_result,
        plot_numpoints=1100,
        plot_init=False,
        fit_kwargs=fit_kwargs_cos,
    )
    # Long matplotlib keywords with init plot
    fit_kwargs_sin = {"color": sin_data_custom.get_color(), "linestyle": ":"}
    init_kwargs_sin = {"c": "black", "ls": "-", "label": "sin guess"}
    sin_fit_custom, sin_guess_custom = plot_fit(
        ax=ax_custom,
        fit_res=sin_result,
        plot_init=True,
        fit_kwargs=fit_kwargs_sin,
        init_kwargs=init_kwargs_sin,
    )
    legend_custom = ax_custom.legend()

    assert isinstance(cos_fit_custom, plt.Line2D)
    assert cos_fit_custom.get_color() == fit_kwargs_cos["c"]
    assert cos_fit_custom.get_linestyle() == fit_kwargs_cos["ls"]
    assert cos_fit_custom.get_label() == fit_kwargs_cos["label"]
    assert len(cos_fit_custom.get_xdata()) == 1100

    assert sin_fit_custom.get_color() == fit_kwargs_sin["color"]
    assert sin_fit_custom.get_linestyle() == fit_kwargs_sin["linestyle"]

    assert sin_guess_custom.get_color() == init_kwargs_sin["c"]
    assert sin_guess_custom.get_linestyle() == init_kwargs_sin["ls"]

    handles, labels = ax_custom.get_legend_handles_labels()
    assert handles == [cos_fit_custom, sin_guess_custom]
    assert labels == ["cos result", "sin guess"]
