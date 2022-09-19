# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing matplotlib and xarray plotting utilities.

Naming convention: plotting functions that require Xarray object(s) as inputs are named
``plot_xr_...``.
"""
# pylint: disable=too-many-arguments, too-many-locals
from typing import List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import Collection, QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from quantify_core.visualization.SI_utilities import (
    set_cbarlabel,
    set_xlabel,
    set_ylabel,
)


def plot_complex_points(
    points: Union[list, np.ndarray],
    colors: list = None,
    labels: list = None,
    markers: list = None,
    legend: bool = True,
    ax: Axes = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plots complex points with (by default) different colors and markers on
    the imaginary plane using :meth:`matplotlib.axes.Axes.plot`.

    Intended for a small number of points.

    .. admonition:: Example

        from quantify_core.utilities.examples_support import plot_centroids

        _ = plot_centroids([1 + 1j, -1.5 - 2j])

    Parameters
    ----------
    ax
        A matplotlib axis to plot on.
    points
        Array of complex numbers.
    colors
        Colors to use for each point.
    labels
        Labels to use for each point. Defaults to ``f"|{i}>"``
    markers
        Markers to use for each point.
    legend
        Calls :meth:`~matplotlib.axes.Axes.legend` if ``True``.
    **kwargs
        Keyword arguments passed to the :meth:`~matplotlib.axes.Axes.plot`.
    """
    if ax is None:
        _, ax = plt.subplots()

    if colors is None:
        colors = [f"C{i%9}" for i in range(len(points))]

    if labels is None:
        # expected usage: plot calibration points
        labels = [f"|{i}>" for i in range(len(points))]

    if markers is None:
        markers = Line2D.filled_markers
        markers = [markers[i % len(Line2D.filled_markers)] for i in range(len(points))]

    if "linestyle" not in kwargs:
        kwargs["linestyle"] = ""

    if "markersize" not in kwargs:
        kwargs["markersize"] = 10

    points = np.asarray(points)

    for real, imag, label, marker, color in zip(
        points.real, points.imag, labels, markers, colors
    ):
        ax.plot(
            [real],
            [imag],
            label=label,
            marker=marker,
            color=color,
            **kwargs,
        )

    if legend:
        ax.legend()

    return ax.get_figure(), ax


def get_unit_from_attrs(data_array: xr.DataArray, str_format: str = " [{}]") -> str:
    """Extracts and formats the unit/units from an :class:`xarray.DataArray` attribute.

    Parameters
    ----------
    data_array
        Xarray array (coordinate or variable).
    str_format
        String that will be formatted if a unit is found.

    Returns
    -------
    :
        ``str_format`` string formatted with the ``data_array.unit`` or
        ``data_array.units``, with that order of precedence. Empty string is returned
        if none of these arguments are present.
    """

    if data_array.attrs.get("unit"):
        str_format = str_format.format(data_array.attrs["unit"])
    elif data_array.attrs.get("units"):
        str_format = str_format.format(data_array.attrs["units"])
    else:
        str_format = ""

    return str_format


# pylint: disable=invalid-name
def plot_xr_complex(
    var: xr.DataArray,
    marker_scatter: str = "o",
    label_real: str = "Real",
    label_imag: str = "Imag",
    cmap: str = "viridis",
    c: np.ndarray = None,
    kwargs_line: dict = None,
    kwargs_scatter: dict = None,
    title: str = "{} [{}]; shape = {}",
    legend: bool = True,
    ax: object = None,
) -> Tuple[Figure, Axes]:
    """Plots the real and imaginary parts of complex data. Points are colored by default
    according to their order in the array.

    Parameters
    ----------
    var
        1D array of complex data.
    marker_scatter
        Marker used for the scatter plot.
    label_real
        Label for legend.
    label_imag
        Label for legend.
    cmap
        The colormap to use for coloring the points.
    c
        Color of the points. Defaults to an array of integers.
    kwargs_line
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.
    kwargs_scatter
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.scatter`.
    title
        Axes title. By default gets formatted with ``var.long_name``, ``var.name`` and
        var.shape``.
    legend
        Calls :meth:`~matplotlib.axes.Axes.legend` if ``True``.
    ax
        The matplotlib axes. If ``None`` a new axes (and figure) is created.
    """

    if ax is None:
        _, ax = plt.subplots()

    if c is None:
        c = np.arange(len(var))

    if kwargs_line is None:
        kwargs_line = {}

    if kwargs_scatter is None:
        kwargs_scatter = {}

    if "marker" not in kwargs_line:
        kwargs_line["marker"] = ""

    var.real.plot(ax=ax, label=label_real, **kwargs_line)
    var.imag.plot(ax=ax, label=label_imag, **kwargs_line)

    for vals in (var.real, var.imag):
        ax.scatter(
            next(iter(var.coords.values())).values,
            vals,
            marker=marker_scatter,
            c=c,
            cmap=cmap,
            **kwargs_scatter,
        )

    ax.set_title(title.format(var.long_name, var.name, var.shape))

    if legend:
        ax.legend()

    return ax.get_figure(), ax


# pylint: disable=invalid-name
def plot_xr_complex_on_plane(
    var: xr.DataArray,
    marker: str = "o",
    label: str = "Data on imaginary plane",
    cmap: str = "viridis",
    c: np.ndarray = None,
    xlabel: str = "Real{}{}{}",
    ylabel: str = "Imag{}{}{}",
    legend: bool = True,
    ax: object = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plots complex data on the imaginary plane. Points are colored by default
    according to their order in the array.


    Parameters
    ----------
    var
        1D array of complex data.
    marker
        Marker used for the scatter plot.
    label
        Data label for the legend.
    cmap
        The colormap to use for coloring the points.
    c
        Color of the points. Defaults to an array of integers.
    xlabel
        Label o x axes.
    ylabel
        Label o y axes.
    legend
        Calls :meth:`~matplotlib.axes.Axes.legend` if ``True``.
    ax
        The matplotlib axes. If ``None`` a new axes (and figure) is created.
    """

    if ax is None:
        _, ax = plt.subplots()

    if c is None:
        c = np.arange(0, len(var))

    ax.scatter(var.real, var.imag, marker=marker, label=label, c=c, cmap=cmap, **kwargs)

    unit_str = get_unit_from_attrs(var)
    ax.set_xlabel(xlabel.format(" ", var.name, unit_str))
    ax.set_ylabel(ylabel.format(" ", var.name, unit_str))

    if legend:
        ax.legend()

    return ax.get_figure(), ax


def set_suptitle_from_dataset(
    fig: Figure, dataset: xr.Dataset, prefix: str = ""
) -> None:
    """
    Sets the suptitle of a matplotlib figure based on

    - (optional) ``prefix``;
    - ``dataset.name``;
    - ``dataset.tuid``,

    Intended for tagging figures with unique ID of the original dataset.

    Parameters
    ----------
    prefix
        Optional string to pre-pend, e.g., ``x0-y0``.
    fig
        The matplotlib figure.
    dataset
        A dataset expected to have a ``.name`` and a ``.tuid"`` attributes.
    """
    fig.suptitle(f"{prefix} {dataset.name}\ntuid: {dataset.tuid}")


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
    image_or_collection
        A matplotlib object returned by either one of :func:`~matplotlib.pyplot.pcolor`,
        :func:`~matplotlib.pyplot.pcolormesh`, :func:`~matplotlib.pyplot.imshow` or
        :func:`~matplotlib.pyplot.matshow`.
    shifted
        Chooses between :code:`"twilight_shifted"`/:code:`"twilight"` colormap and the
        colormap range.
    unit
        Used to fix the colormap range.
    clim
        The colormap limit.


    .. include:: examples/visualization.mpl_plotting.set_cyclic_colormap.rst.txt
    """  # pylint: disable=line-too-long
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
    ax
        The :obj:`~matplotlib.axes.Axes` on which to plot.
    text
        The text of the textbox.

    Return
    ------
    :
        the new text object
    """
    box_props = dict(boxstyle="round", pad=0.4, facecolor="white", alpha=0.5)
    new_kw_with_defaults = dict(
        x=1.05,
        y=0.95,
        transform=ax.transAxes,
        bbox=box_props,
        verticalalignment="top",
        s=text,
    )

    new_kw_with_defaults.update(kw)

    t_obj = ax.text(**new_kw_with_defaults)
    return t_obj


def plot_fit(
    ax,
    fit_res,
    plot_init: bool = True,
    plot_numpoints: int = 1000,
    range_casting: Literal["abs", "angle", "real", "imag"] = "real",
    fit_kwargs: dict = None,
    init_kwargs: dict = None,
) -> List[plt.Line2D]:
    """
    Plot a fit of an lmfit model with a real domain.

    Parameters
    ----------
    ax
        axis on which to plot the fit.
    fit_res
        an lmfit fit results object.
    plot_init
        if True, plot the initial guess of the fit.
    plot_numpoints
        the number of points used on which to evaluate the fit.
    range_casting
        how to plot fit functions that have a complex range.
        Casting of values happens using :obj:`~numpy.absolute`, :obj:`~numpy.angle`,
        :obj:`~numpy.real` and :obj:`~numpy.imag`. Angle is in degrees.
    fit_kwargs, optional
        Matplotlib pyplot formatting and label keyword arguments for the fit plot.
        default value is {"color": "C3", "label": "Fit"}
    init_kwargs, optional
        Matplotlib pyplot formatting and label keyword arguments for the init plot.
        default value is {"color": "grey", "linestyle": "--", "label": "Guess"}

    Returns
    -------
    :
        list of matplotlib pyplot Line2D objects
    """

    # Handle default values
    if fit_kwargs is None:
        fit_kwargs = {"color": "C3", "label": "Fit"}
    if init_kwargs is None:
        init_kwargs = {"color": "grey", "linestyle": "--", "label": "Guess"}

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
    fit_y = model.eval(fit_res.params, **{independent_var: x})
    init_y = model.eval(fit_res.init_params, **{independent_var: x})

    if range_casting != "angle":
        range_cast_func = getattr(np, range_casting)
        fit_y = range_cast_func(fit_y)
    else:
        fit_y = np.angle(fit_y, deg=True)

    lines = ax.plot(x, fit_y, **fit_kwargs)

    if plot_init:
        if range_casting != "angle":
            range_cast_func = getattr(np, range_casting)
            init_y = range_cast_func(init_y)
        else:
            init_y = np.angle(init_y, deg=True)

        lines += ax.plot(x, init_y, **init_kwargs)

    return lines


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
    Add a rectangular block to a color plot using
    :meth:`~matplotlib.axes.Axes.pcolormesh`.

    Parameters
    ----------
    xvals
        Length N array corresponding to settable x0.
    yvals
        Length M array corresponding to settable x1.
    zvals
        M*N array corresponding to gettable yi.
    ax
        Axis to which to add the colormesh.
    normalize
        If ``True``, normalizes each row of data.
    log
        if ``True``, uses a logarithmic colorscale.
    cmap
        Colormap to use. See
        `matplotlib docs <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for
        choosing an appropriate colormap.
    vlim
        Limits of the z-axis.
    transpose
        If ``True`` transposes the figure.

    Returns
    ------------
    :
        The created matplotlib QuadMesh.


    .. warning::

        The **grid orientation** for the zvals is the same as is used in
        :meth:`~matplotlib.axes.Axes.pcolormesh`.
        Note that the column index corresponds to the x-coordinate,
        and the row index corresponds to y.
        This can be counter.intuitive: zvals(y_idx, x_idx)
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


# pylint: disable=invalid-name
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
    Creates a heatmap of x,y,z data that was acquired on a grid expects three "columns"
    of data of equal length.

    Parameters
    ----------
    x
        Length N array corresponding to x values.
    y
        Length N array corresponding to y values.
    z:
        Length N array corresponding to gettable z values.
    xlabel
        x label to add to the heatmap.
    ylabel
        y label to add to the heatmap.
    xunit
        x unit used in unit aware axis labels.
    yunit
        y unit used in unit aware axis labels.
    zlabel
        Label used for the colorbar.
    ax
        Axis to which to add the colormesh.
    cax
        Axis on which to add the colorbar, if set to ``None``, will create a new axis.
    add_cbar
        if ``True``, adds a colorbar.
    title
        Text to add as title to the axis.
    normalize
        if ``True``, normalizes each row of data.
    log
        if ``True``, uses a logarithmic colorscale
    cmap
        The colormap to use. See
        `matplotlib docs <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for
        choosing an appropriate colormap.
    vlim
        limits of the z-axis.
    transpose
        if ``True`` transposes the figure.

    Returns
    -------
    :
        The new matplotlib QuadMesh and Colorbar.


    """

    if ax is None:
        _, ax = plt.subplots()

    # Reshape the length N columns of data into unique xvals (n), yvals (m) and an (m*n)
    # grid of zvals.
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
    set_xlabel(xlabel, xunit, ax)
    set_ylabel(ylabel, yunit, ax)

    if add_cbar:
        if cax is None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="5%", pad="2%")
        cbar = plt.colorbar(quadmesh, cax=cax, orientation="vertical")
        set_cbarlabel(cbar, zlabel, zunit)

    return quadmesh, cbar
