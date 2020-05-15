"""
Module containing the pyqtgraph based plotting monitor.
"""
import numpy as np

from .plot_interpolation import interpolate_heatmap

from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals
from qcodes.plots.colors import color_cycle
from qcodes.instrument.base import Instrument


from quantify.data.handling import load_dataset, get_latest_tuid

from qcodes.plots.pyqtgraph import QtPlot, TransformState


class PlotMonitor_pyqt(Instrument):
    """
    Pyqtgraph based plot monitor.

    A plot monitor is intended to provide a real-time visualization of a
    dataset.
    A plotmon should implement two methods

    The interface of a plot monitor is based on the tuid.

    """

    def __init__(
            self,
            name: str):  # verbose: bool = True
        """
        Creates an instance of the Measurement Control.

        Args:
            name (str): name
        """
        super().__init__(name=name)

        # Paramaters are attributes that we include in logging
        # and intend the user to change.

        self.add_parameter(
            "tuid",
            docstring="The tuid of the dataset to monitor",
            parameter_class=ManualParameter,
            vals=vals.Strings(),
            initial_value='latest',)

        # used to track if tuid changed
        self._last_tuid = None

        self.create_plot_monitor()

        # convenient access to curve variable for updating
        self._curves = []
        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

    def create_plot_monitor(self):
        """
        Creates the PyQtGraph plotting monitors.
        Can also be used to recreate these when plotting has crashed.
        """
        if hasattr(self, "main_QtPlot"):
            del self.main_QtPlot
        if hasattr(self, "secondary_QtPlot"):
            del self.secondary_QtPlot

        self.main_QtPlot = QtPlot(
            window_title="Main plotmon of {}".format(self.name), figsize=(600, 400)
        )
        self.secondary_QtPlot = QtPlot(
            window_title="Secondary plotmon of {}".format(self.name), figsize=(600, 400)
        )

    def _initialize_plot_monitor(self, tuid):
        """
        Clears data in plot monitors and sets it up with the data from the
        dataset
        """

        # Clear the plot monitors if required.
        if self.main_QtPlot.traces != []:
            self.main_QtPlot.clear()

        if self.secondary_QtPlot.traces != []:
            self.secondary_QtPlot.clear()
        self.curves = []
        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

        # TODO add persistence based on previous dataset
        dset = load_dataset(tuid=tuid)

        set_parnames = list(filter(lambda k: 'x' in k, dset.keys()))
        get_parnames = list(filter(lambda k: 'y' in k, dset.keys()))

        #############################################################
        plot_idx = 1
        for yi in get_parnames:
            for xi in set_parnames:
                # TODO: add persist data here.
                self.main_QtPlot.add(
                    x=dset[xi].values, y=dset[yi].values,
                    subplot=plot_idx,
                    xlabel=dset[xi].attrs['long_name'],
                    xunit=dset[xi].attrs['unit'],
                    ylabel=dset[yi].attrs['long_name'],
                    yunit=dset[yi].attrs['unit'],
                    symbol='o', symbolSize=5,
                )
                # Manual counter is used because we may want to add more
                # than one quantity per panel
                plot_idx += 1
                self.curves.append(self.main_QtPlot.traces[-1])
            self.main_QtPlot.win.nextRow()

        #############################################################
        # Add a square heatmap
        if dset.attrs['2D-grid']:
            plot_idx = 1
            for yi in get_parnames:

                cmap = 'viridis'
                zrange = None

                x = dset['x0'].values[:dset.attrs['xlen']]
                y = dset['x1'].values[::dset.attrs['xlen']]
                Z = np.reshape(dset[yi].values,
                               (len(x), len(y)), order='F').T
                config_dict = {
                    "x": x,
                    "y": y,
                    "z": Z,
                    "xlabel": dset['x0'].attrs['long_name'],
                    "xunit": dset['x0'].attrs['unit'],
                    "ylabel": dset['x1'].attrs['long_name'],
                    "yunit": dset['x1'].attrs['unit'],
                    "zlabel": dset[yi].attrs['long_name'],
                    "zunit": dset[yi].attrs['unit'],
                    "subplot": plot_idx,
                    "cmap": cmap,
                }
                if zrange is not None:
                    config_dict["zrange"] = zrange
                self.secondary_QtPlot.add(**config_dict)
                plot_idx += 1

        #############################################################
        # if data is not on a grid but is 2D it makes sense to interpolate

        elif len(set_parnames) == 2:
            plot_idx = 1
            for yi in get_parnames:

                cmap = 'viridis'
                zrange = None

                config_dict = {
                    "x": [0, 1],
                    "y": [0, 1],
                    "z": np.zeros([2, 2]),
                    "xlabel": dset['x0'].attrs['long_name'],
                    "xunit": dset['x0'].attrs['unit'],
                    "ylabel": dset['x1'].attrs['long_name'],
                    "yunit": dset['x1'].attrs['unit'],
                    "zlabel": dset[yi].attrs['long_name'],
                    "zunit": dset[yi].attrs['unit'],
                    "subplot": plot_idx,
                    "cmap": cmap,
                }
                if zrange is not None:
                    config_dict["zrange"] = zrange
                self.secondary_QtPlot.add(**config_dict)
                self._im_curves.append(self.secondary_QtPlot.traces[-1])

                # used to mark the interpolation points
                self.secondary_QtPlot.add(
                    x=[], y=[],
                    pen=None,
                    color=1.0, width=0, symbol="o", symbolSize=4,
                    subplot=plot_idx,
                    xlabel=dset['x0'].attrs['long_name'],
                    xunit=dset['x0'].attrs['unit'],
                    ylabel=dset['x1'].attrs['long_name'],
                    yunit=dset['x1'].attrs['unit'],)
                self._im_scatters.append(self.secondary_QtPlot.traces[-1])

                # used to mark the last N-interpolation points
                self.secondary_QtPlot.add(
                    x=[], y=[],
                    color=color_cycle[3],  # marks the point red
                    width=0, symbol="o", symbolSize=7,
                    subplot=plot_idx,
                    xlabel=dset['x0'].attrs['long_name'],
                    xunit=dset['x0'].attrs['unit'],
                    ylabel=dset['x1'].attrs['long_name'],
                    yunit=dset['x1'].attrs['unit'],)
                self._im_scatters_last.append(self.secondary_QtPlot.traces[-1])

                plot_idx += 1

    def update(self):

        if self.tuid() == 'latest':
            # this should automatically set tuid to the most recent tuid.
            tuid = get_latest_tuid()
        else:
            tuid = self.tuid()

        # If tuid has changed, we need to initialize the figure
        if tuid != self._last_tuid:
            # need to initialize the plot monitor
            self._initialize_plot_monitor(tuid)
            self._last_tuid = tuid

        # otherwise we simply update
        else:
            dset = load_dataset(tuid=tuid)
            set_parnames = list(filter(lambda k: 'x' in k, dset.keys()))
            get_parnames = list(filter(lambda k: 'y' in k, dset.keys()))
            # Only updates the main monitor currently.

            #############################################################
            i = 0
            for yi in get_parnames:
                for xi in set_parnames:
                    self.curves[i]["config"]["x"] = dset[xi].values
                    self.curves[i]["config"]["y"] = dset[yi].values
                    i += 1
            self.main_QtPlot.update_plot()

            #############################################################
            # Add a square heatmap
            if dset.attrs['2D-grid']:
                for yidx, yi in enumerate(get_parnames):
                    Z = np.reshape(dset[yi].values,
                                   (dset.attrs['xlen'], dset.attrs['ylen']),
                                   order='F').T
                    self.secondary_QtPlot.traces[yidx]['config']['z'] = Z
                self.secondary_QtPlot.update_plot()

            #############################################################
            # if data is not on a grid but is 2D it makes sense to interpolate
            elif len(set_parnames) == 2:
                for yidx, yi in enumerate(get_parnames):
                    # exists to force reset the x- and y-axis scale
                    new_sc = TransformState(0, 1, True)

                    x = dset['x0'].values[~np.isnan(dset)['y0']]
                    y = dset['x1'].values[~np.isnan(dset)['y0']]
                    z = dset[yi].values[~np.isnan(dset)['y0']]
                    # interpolation needs to be meaningful
                    if len(z) < 8:
                        break
                    x_grid, y_grid, z_grid = interpolate_heatmap(
                        x=x, y=y, z=z,
                        interp_method='linear')

                    trace = self._im_curves[yidx]
                    trace['config']['x'] = x_grid
                    trace['config']['y'] = y_grid
                    trace['config']['z'] = z_grid
                    # force rescale axis so marking datapoints works
                    trace["plot_object"]["scales"]["x"] = new_sc
                    trace["plot_object"]["scales"]["y"] = new_sc

                    # Mark all measured points on which the interpolation
                    # is based
                    trace = self._im_scatters[yidx]
                    trace['config']['x'] = x
                    trace['config']['y'] = y

                    trace = self._im_scatters_last[yidx]
                    trace['config']['x'] = x[-5:]
                    trace['config']['y'] = y[-5:]

                self.secondary_QtPlot.update_plot()
