# -----------------------------------------------------------------------------
# Description:    Module containing matplotlib plotting utilities.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from typing import Tuple, Union
from typing_extensions import Literal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.collections import QuadMesh, Collection
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from quantify.visualization.SI_utilities import set_xlabel, set_ylabel, set_cbarlabel


def set_cyclic_colormap(
    image_or_collection: Union[AxesImage, QuadMesh, Collection],
    shifted: bool = False,
    unit: Literal["deg", "rad"] = "deg",
    clim: tuple = None,
) -> None:
    """
    Sets a cyclic colormap on a matplolib 2D color plot if cyclic units are detected.

    Parameters
    ----------
    image_or_collection: Union[:class:`~matplotlib.image.AxesImage`, :class:`~matplotlib.collections.QuadMesh`, :class:`~matplotlib.collections.Collection`]
        a matplotlib object returned by either one of :func:`~matplotlib.pyplot.pcolor`,
        :func:`~matplotlib.pyplot.pcolormesh`, :func:`~matplotlib.pyplot.imshow` or
        :func:`~matplotlib.pyplot.matshow`.
    shifted
        Chooses between :code:`"twilight_shifted"`/:code:`"twilight"` colormap and the
        colormap range.
    unit
        Used to fix the colormap range.
    clim
        The colormap limit.


    .. include:: ./docstring_examples/quantify.visualization.mpl_plotting.set_cyclic_colormap.rst.txt
    """
    shifted = bool(shifted)  # in case xarray min() is used
    if unit in {"deg", "rad"}:
        clim_d = {
            True: {"deg": (-180.0, 180.0), "rad": (-np.pi, np.pi)},
            False: {"deg": (0.0, 360.0), "rad": (0, 2 * np.pi)},
        }
        cmap = {True: "twilight_shifted", False: "twilight"}
        image_or_collection.set_clim(clim_d[shifted][unit] if clim is None else clim)
        image_or_collection.set_cmap(cmap[shifted])


def plot_textbox(ax: Axes, text: str, **kw) -> Text:
    """
    Plot a textbox with sensible defaults using :obj:`~matplotlib.axes.Axes.text`.

    Parameters
    ----------
    ax: :obj:`~matplotlib.axes.Axes`
        the :obj:`~matplotlib.axes.Axes` on which to plot
    text
        the text of the textbox

    Return
    ------
    :
        the new text object
    """
    box_props = dict(boxstyle="round", pad=0.4, facecolor="white", alpha=0.5)
    t_obj = ax.text(
        x=kw.get("x", 1.05),
        y=kw.get("y", 0.95),
        s=text,
        transform=kw.get("transform", ax.transAxes),
        bbox=kw.get("bbox", box_props),
        verticalalignment=kw.get("verticalalignment", "top"),
        **kw
    )
    return t_obj


def plot_fit(
    ax,
    fit_res,
    plot_init: bool = True,
    plot_numpoints: int = 1000,
    range_casting: Literal["abs", "angle", "real", "imag"] = "abs",
) -> None:
    """
    Plot a fit of an lmfit model with a real domain.

    Parameters
    ----------
    ax: :obj:`~matplotlib.axes.Axes`
        axis on which to plot the fit.
    fit_res
        an lmfit fit results object.
    plot_init
        if True, plot the initial guess of the fit.
    plot_numpoints
        the number of points used on which to evaulate the fit.
    range_casting
        how to plot fit functions that have a complex range.
        Casting of values happens using :obj:`~numpy.absolute`, :obj:`~numpy.angle`, :obj:`~numpy.real` and :obj:`~numpy.imag`.
        angle is in degrees.
    """
    model = fit_res.model

    if len(model.independent_vars) == 1:
        independent_var = model.independent_vars[0]
    else:
        raise ValueError(
            "Fit can only be plotted if the model function"
            " has one independent variable."
        )

    x_arr = fit_res.userkws[independent_var]
    x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
    y = model.eval(fit_res.params, **{independent_var: x})
    if range_casting != "angle":
        range_cast_func = getattr(np, range_casting)
        y = range_cast_func(y)
    else:
        y = np.angle(y, deg=True)

    ax.plot(x, y, label="Fit", c="C3")

    if plot_init:
        x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
        y = model.eval(fit_res.init_params, **{independent_var: x})
        if range_casting != "angle":
            range_cast_func = getattr(np, range_casting)
            y = range_cast_func(y)
        else:
            y = np.angle(y, deg=True)
        ax.plot(x, y, ls="--", c="grey", label="Guess")


def plot_fit_complex_plane(
    ax, fit_res, plot_init: bool = True, plot_numpoints: int = 1000
) -> None:
    """
    Plot a fit of an lmfit model with a real domain in the complex plane.
    """
    model = fit_res.model

    if len(model.independent_vars) == 1:
        independent_var = model.independent_vars[0]
    else:
        raise ValueError(
            "Fit can only be plotted if the model function"
            " has one independent variable."
        )

    x_arr = fit_res.userkws[independent_var]
    x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
    y = model.eval(fit_res.params, **{independent_var: x})

    ax.plot(y.real, y.imag, label="Fit", c="C3")

    if plot_init:
        x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
        y = model.eval(fit_res.init_params, **{independent_var: x})
        ax.plot(y.real, y.imag, ls="--", c="grey", label="Guess")


def flex_colormesh_plot_vs_xy(
    xvals: np.ndarray,
    yvals: np.ndarray,
    zvals: np.ndarray,
    ax: Axes = None,
    normalize: bool = False,
    log: bool = False,
    cmap: str = "viridis",
    vlim: list = (None, None),
    transpose: bool = False,
) -> QuadMesh:
    """
    Add a rectangular block to a color plot using pcolormesh.

    Parameters
    -----------------
    xvals:
        length N array corresponding to settable x0.
    yvals:
        length M array corresponding to settable x1.
    zvals:
        M*N array corresponding to gettable yi.
    ax: :obj:`~matplotlib.axes.Axes`
        axis to which to add the colormesh
    normalize:
        if True, normalezes each row of data.
    log:
        if True, uses a logarithmic colorscale
    cmap:
        colormap to use. See `matplotlib docs <https://matplotlib.org/tutorials/colors/colormaps.html>`_
        for choosing an appropriate colormap.
    vlim:
        limits of the z-axis.
    transpose:
        if True transposes the figure.

    Returns
    ------------
    :
        the created matplotlib QuadMesh


    .. warning::

        The **grid orientation** for the zvals is the same as is used in
        ax.pcolormesh.
        Note that the column index corresponds to the x-coordinate,
        and the row index corresponds to y.
        This can be counterintuitive: zvals(y_idx, x_idx)
        and can be inconsistent with some arrays of zvals
        (such as a 2D histogram from numpy).

    """

    xvals = np.array(xvals)
    yvals = np.array(yvals)

    # First, we need to sort the data as otherwise we get odd plotting
    # artifacts. An example is e.g., plotting a Fourier transform
    sorted_x_arguments = xvals.argsort()
    xvals = xvals[sorted_x_arguments]
    sorted_y_arguments = yvals.argsort()
    yvals = yvals[sorted_y_arguments]
    zvals = zvals[:, sorted_x_arguments]
    zvals = zvals[sorted_y_arguments, :]

    # convert xvals and yvals to single dimension arrays
    xvals = np.squeeze(np.array(xvals))
    yvals = np.squeeze(np.array(yvals))

    # calculate coordinates for corners of color blocks
    # x coordinates
    xvertices = np.zeros(np.array(xvals.shape) + 1)
    xvertices[1:-1] = (xvals[:-1] + xvals[1:]) / 2.0
    xvertices[0] = xvals[0] - (xvals[1] - xvals[0]) / 2
    xvertices[-1] = xvals[-1] + (xvals[-1] - xvals[-2]) / 2
    # y coordinates
    yvertices = np.zeros(np.array(yvals.shape) + 1)
    yvertices[1:-1] = (yvals[:-1] + yvals[1:]) / 2.0
    yvertices[0] = yvals[0] - (yvals[1] - yvals[0]) / 2
    yvertices[-1] = yvals[-1] + (yvals[-1] - yvals[-2]) / 2

    xgrid, ygrid = np.meshgrid(xvertices, yvertices)

    # normalized plot
    if normalize:
        zvals /= np.mean(zvals, axis=0)
    # logarithmic plot
    if log:
        for xx in range(len(xvals)):
            zvals[xx] = np.log(zvals[xx]) / np.log(10)

    # add blocks to plot
    if transpose:
        colormap = ax.pcolormesh(
            ygrid.transpose(),
            xgrid.transpose(),
            zvals.transpose(),
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
        )
    else:
        colormap = ax.pcolormesh(
            xgrid, ygrid, zvals, cmap=cmap, vmin=vlim[0], vmax=vlim[1]
        )

    return colormap


def plot_2d_grid(
    x,
    y,
    z,
    xlabel: str,
    xunit: str,
    ylabel: str,
    yunit: str,
    zlabel: str,
    zunit: str,
    ax: Axes,
    cax: Axes = None,
    add_cbar: bool = True,
    title: str = None,
    normalize: bool = False,
    log: bool = False,
    cmap: str = "viridis",
    vlim: list = (None, None),
    transpose: bool = False,
) -> Tuple[QuadMesh, Colorbar]:
    """
    Creates a heatmap of x,y,z data that was acquired on a grid expects three "columns" of data of equal length.


    Parameters
    ------------
    x, y:
        length N array corresponding to settable x0 and x1.
    z:
        length N array corresponding to gettable yi.
    xlabel, ylabel :
        x/y label to add to the heatmap.
    xunit, yunit :
        x/y unit used in unit aware axis labels.
    zlabel:
        label used for the colorbar
    ax: :obj:`~matplotlib.axes.Axes`
        axis to which to add the colormesh
    cax: :obj:`~matplotlib.axes.Axes`
        axis on which to add the colorbar, if set to None, will create a new axis.
    add_cbar:
        if True, adds a colorbar.
    title:
        Text to add as title to the axis.
    normalize:
        if True, normalezes each row of data.
    log:
        if True, uses a logarithmic colorscale
    cmap:
        colormap to use. See `matplotlib docs <https://matplotlib.org/tutorials/colors/colormaps.html>`_
        for choosing an appropriate colormap.
    vlim:
        limits of the z-axis.
    transpose:
        if True transposes the figure.

    Returns
    ------------
    :
        The new matplotlib QuadMesh and Colorbar


    """

    if ax is None:
        _, ax = plt.subplots()

    # Reshape the lenth N columns of data into unique xvals (n), yvals (m) and an (m*n) grid of zvals.
    xi = np.unique(x)
    yi = np.unique(y)
    zarr = np.array(z)  # to make this work natively with an xarray
    # to account for matlab style column ordering of pcolormesh
    zi = np.reshape(zarr, newshape=(len(yi), len(xi)))

    quadmesh = flex_colormesh_plot_vs_xy(
        xi,
        yi,
        zi,
        ax=ax,
        normalize=normalize,
        log=log,
        cmap=cmap,
        vlim=vlim,
        transpose=transpose,
    )

    if title is not None:
        ax.set_title(title)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)

    if add_cbar:
        # colorbar is added here.
        if cax is None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="5%", pad="2%")
        cbar = plt.colorbar(quadmesh, cax=cax, orientation="vertical")
        set_cbarlabel(cbar, zlabel, zunit)

    return quadmesh, cbar
