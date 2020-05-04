"""
Module containing the pyqtgraph based plotting monitor.
"""
import numpy as np

from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals
from qcodes.plots.colors import color_cycle
from qcodes.instrument.base import Instrument


from quantify.measurement.data_handling import load_dataset, get_latest_tuid

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

        # TODO add persistence based on previous dataset
        dset = load_dataset(tuid=tuid)

        set_parnames = list(filter(lambda k: 'x' in k, dset.keys()))
        get_parnames = list(filter(lambda k: 'y' in k, dset.keys()))

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
            i = 0
            for yi in get_parnames:
                for xi in set_parnames:
                    self.curves[i]["config"]["x"] = dset[xi].values
                    self.curves[i]["config"]["y"] = dset[yi].values
                    i += 1
            self.main_QtPlot.update_plot()

            # Add a square heatmap
            if dset.attrs['2D-grid']:
                for yidx, yi in enumerate(get_parnames):
                    Z = np.reshape(dset[yi].values,
                                   (dset.attrs['xlen'], dset.attrs['ylen']),
                                   order='F').T
                    self.secondary_QtPlot.traces[yidx]['config']['z'] = Z
                self.secondary_QtPlot.update_plot()
