import matplotlib.pyplot as plt
import numpy as np


def flex_colormesh_plot_vs_xy(xvals: np.ndarray, yvals: np.ndarray,
                              zvals: np.ndarray, ax: plt.Axes = None,
                              normalize: bool = False, log: bool = False,
                              cmap: str = 'viridis', clim: list = (None, None),
                              transpose: bool = False):
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
    ax:
        axis to which to add the colormesh




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
    # artefacts. An example is e.g., plotting a fourier transform
    sorted_x_arguments = xvals.argsort()
    xvals = xvals[sorted_x_arguments]
    sorted_y_arguments = yvals.argsort()
    yvals = yvals[sorted_y_arguments]
    zvals = zvals[:,  sorted_x_arguments]
    zvals = zvals[sorted_y_arguments, :]

    # convert xvals and yvals to single dimension arrays
    xvals = np.squeeze(np.array(xvals))
    yvals = np.squeeze(np.array(yvals))

    # calculate coordinates for corners of color blocks
    # x coordinates
    xvertices = np.zeros(np.array(xvals.shape)+1)
    xvertices[1:-1] = (xvals[:-1]+xvals[1:])/2.
    xvertices[0] = xvals[0] - (xvals[1]-xvals[0])/2
    xvertices[-1] = xvals[-1] + (xvals[-1]-xvals[-2])/2
    # y coordinates
    yvertices = np.zeros(np.array(yvals.shape)+1)
    yvertices[1:-1] = (yvals[:-1]+yvals[1:])/2.
    yvertices[0] = yvals[0] - (yvals[1]-yvals[0])/2
    yvertices[-1] = yvals[-1] + (yvals[-1]-yvals[-2])/2

    xgrid, ygrid = np.meshgrid(xvertices, yvertices)

    # normalized plot
    if normalize:
        zvals /= np.mean(zvals, axis=0)
    # logarithmic plot
    if log:
        for xx in range(len(xvals)):
            zvals[xx] = np.log(zvals[xx])/np.log(10)

    # add blocks to plot
    if transpose:
        colormap = ax.pcolormesh(ygrid.transpose(),
                                 xgrid.transpose(),
                                 zvals.transpose(),
                                 cmap=cmap, vmin=clim[0], vmax=clim[1])
    else:
        colormap = ax.pcolormesh(xgrid, ygrid, zvals, cmap=cmap,
                                 vmin=clim[0], vmax=clim[1])

    return {'fig': ax.figure, 'ax': ax, 'cmap': colormap}
