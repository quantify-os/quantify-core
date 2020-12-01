# -----------------------------------------------------------------------------
# Description:    Module containing the pyqtgraph based plotting monitor.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import numpy as np
from collections import deque
from itertools import chain

from qcodes import validators as vals
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.plots.colors import color_cycle
from .color_utilities import faded_color_cycle, darker_color_cycle
from qcodes.plots.pyqtgraph import QtPlot, TransformState

from quantify.utilities.general import get_keys_containing
from quantify.data.handling import load_dataset, _xi_and_yi_match
from quantify.visualization.plot_interpolation import interpolate_heatmap


class PlotMonitor_pyqt(Instrument):
    """
    Pyqtgraph based plot monitor.

    A plot monitor is intended to provide a real-time visualization of a dataset.
    """

    def __init__(self, name: str):
        """
        Creates an instance of the Measurement Control.

        Parameters
        ----------
        name : str
            name
        """
        super().__init__(name=name)

        # used to track if tuid changed
        self._last_tuid = None

        # used to track last tuid added to the previous dsets
        self._last_tuid_prev = None

        # used to be able to check if dataset are compatible to plot together
        self._dset = None

        # used to store user-specified persistent datasets
        self._persistent_dsets = []

        # used to track the tuids of previous datasets
        # deque([<oldest>, ..., <one before latest>])
        self._previous_dsets = deque()
        self._previous_tuids = deque()

        symbols = ["o", "t", "t1", "t2", "t3", "s", "p", "h", "star", "+", "d"]
        # keeps track of the symbol assigned to each dataset
        # we reserve "o" symbol for the latest dataset
        self._previous_symbols = deque(symbols[1:])
        # Invert the order for less likely duplicates (in the beginning)
        self._persistent_symbols = symbols[1:][::-1]

        # We reserve the first (blue color to the latest dataset)
        # When it is not the latest anymore it will change color and maintain it
        self._previous_colors = deque(faded_color_cycle[1:])
        self._persistent_colors = darker_color_cycle[1:][::-1]

        self.create_plot_monitor()

        # convenient access to curve variable for updating
        self.curves = []
        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

        # Keep at the end of __init__ due to some dependencies
        # Parameters are attributes that we include in logging
        # and intend the user to change.

        self.add_parameter(
            name="tuid",
            docstring="The tuid of the dataset to monitor",
            parameter_class=Parameter,
            vals=vals.MultiType(vals.Strings(), vals.Enum(None)),
            # avoid set_cmd being called at __init__
            initial_cache_value=None,
            set_cmd=self._set_tuid,
        )
        self.add_parameter(
            name="max_num_previous_dsets",
            docstring="The maximum number of auto-accumulated persistent datasets",
            parameter_class=Parameter,
            vals=vals.Ints(min_value=0, max_value=100),
            # avoid set_cmd being called at __init__
            initial_cache_value=2,
            set_cmd=self._set_max_num_previous_dsets,
        )
        self.add_parameter(
            name="previous_tuids",
            docstring=(
                "The tuids of the auto-accumulated previous datasets.\n"
                "Read-only. NB: does not contain the latest tuid "
                "[which is stored in `.tuid()`].\n"
                "See also `persistent_tuids`."
            ),
            parameter_class=Parameter,
            get_cmd=self._get_previous_tuids,
            set_cmd=self._set_previous_tuids,
        )

        self.add_parameter(
            name="persistent_tuids",
            docstring=(
                "The tuids of the user-specified persistent datasets.\n"
                "As opposed to the `previous_tuids`, these never vanish.\n"
                "Can be reset by setting to `[]`"
            ),
            parameter_class=Parameter,
            vals=vals.Lists(),
            set_cmd=self._set_persistent_tuids,
            # avoid set_cmd being called at __init__
            initial_cache_value=[]
        )

    def _set_max_num_previous_dsets(self, val):
        """
        used only to update relevant variables
        """
        self._pop_old_prev_dsets(val)
        self._initialize_plot_monitor()

    def _set_tuid(self, tuid):
        """
        To be called only on `.tuid()`
        """
        if self._last_tuid_prev is not None:
            self._previous_tuids.append(self._last_tuid_prev)
            self._previous_dsets.append(load_dataset(self._last_tuid_prev))

        self._last_tuid_prev = tuid

        if tuid is not None:
            # Load the dataset to be monitored
            self._dset = load_dataset(tuid)
        else:
            # Discard the dataset
            self._dset = None

        # Now we ensure all datasets are compatible to be plotted together
        dset_it = [self._dset] if self._dset else []
        if not _xi_and_yi_match(chain(dset_it, self._previous_dsets, self._persistent_dsets)):
            # Last dataset under self.tuid() has priority, reset the others
            if not _xi_and_yi_match(chain(dset_it, self._previous_dsets)):
                # Reset the previous datasets
                self._pop_old_prev_dsets(val=0)
            if not _xi_and_yi_match(chain(dset_it, self._persistent_dsets)):
                # Reset the user-defined persistent datasets
                self.persistent_tuids([])

        self._pop_old_prev_dsets(val=self.max_num_previous_dsets())

    def _pop_old_prev_dsets(self, val):
        while len(self._previous_dsets) > val:
            self._previous_dsets.popleft()
            self._previous_tuids.popleft()
            # This ensures each datasets preserves it symbol
            self._previous_symbols.rotate(-1)
            self._previous_colors.rotate(-1)

    def _get_previous_tuids(self):
        """
        Get cmd for previous_tuids
        """
        return self._previous_tuids

    def _set_previous_tuids(self, value):
        """
        Set cmd for previous_tuids
        """
        print(
            "`{}` is read-only. Use `{}`.".format(
                self.previous_tuids.name, self.persistent_tuids.name
            )
        )
        return None

    def _set_persistent_tuids(self, tuids):
        """
        Set cmd for persistent_tuids
        Loads all the user datasets so that they can be plotted afterwards
        """
        self._persistent_dsets = [load_dataset(tuid) for tuid in tuids]

        # Now we ensure all datasets are compatible to be plotted together
        dset_it = [self._dset] if self._dset else []
        if not _xi_and_yi_match(chain(dset_it, self._previous_dsets, self._persistent_dsets)):
            # persistent dataset specified by user have priority, reset the others

            # This check need to be the first
            if not _xi_and_yi_match(chain(self._persistent_dsets, dset_it)):
                # Reset the tuid because the user has specified persistent dsets
                self.tuid(None)

            if not _xi_and_yi_match(chain(self._persistent_dsets, self._previous_dsets)):
                # Reset the previous datasets
                self._pop_old_prev_dsets(val=0)

        self._initialize_plot_monitor()
        return True

    def create_plot_monitor(self):
        """
        Creates the PyQtGraph plotting monitors.
        Can also be used to recreate these when plotting has crashed.
        """
        if hasattr(self, "main_QtPlot"):
            del self.main_QtPlot
        if hasattr(self, "secondary_QtPlot"):
            del self.secondary_QtPlot

        self.secondary_QtPlot = QtPlot(
            window_title="Secondary plotmon of {}".format(self.name), figsize=(600, 400)
        )
        # Create last to appear on top
        self.main_QtPlot = QtPlot(
            window_title="Main plotmon of {}".format(self.name), figsize=(600, 400)
        )

    def _mk_legend(self, dset):
        return dset.attrs["tuid"].split("-")[-1] + " " + dset.attrs["name"]

    def _initialize_plot_monitor(self):
        """
        Clears data in plot monitors and sets it up with the data from the dataset
        """

        # Clear the plot monitors if required.
        if self.main_QtPlot.traces:
            self.main_QtPlot.clear()

        if self.secondary_QtPlot.traces:
            self.secondary_QtPlot.clear()

        self.curves = []
        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

        all_dsets = (
            self._persistent_dsets
            + list(self._previous_dsets)
            + ([self._dset] if self._dset else [])
        )

        if not len(all_dsets):
            # Nothing to be done
            return None

        set_parnames = sorted(_get_parnames(all_dsets[0], par_type="x"))
        get_parnames = sorted(_get_parnames(all_dsets[0], par_type="y"))

        dset = self._dset

        #############################################################

        plot_idx = 1
        for yi in get_parnames:
            for xi in set_parnames:
                # Persistent datasets auto and user
                p_dsets = (self._persistent_dsets, self._previous_dsets)
                p_colors = (self._persistent_colors, self._previous_colors)
                p_symbols = (self._persistent_symbols, self._previous_symbols)
                p_symbolSizes = (10, 8)
                p_symbolBrushes = (False, (230, 230, 230))  # gray for previous dsets
                for colors, symbols, dsets, symbolSize, symBrush in zip(
                    p_colors, p_symbols, p_dsets, p_symbolSizes, p_symbolBrushes
                ):
                    for i_p_dset, p_dset in enumerate(dsets):
                        has_settable = xi in _get_parnames(p_dset, "x")
                        has_gettable = yi in _get_parnames(p_dset, "y")
                        if has_settable and has_gettable:
                            self.main_QtPlot.add(
                                x=p_dset[xi].values,
                                y=p_dset[yi].values,
                                subplot=plot_idx,
                                xlabel=p_dset[xi].attrs["long_name"],
                                xunit=p_dset[xi].attrs["unit"],
                                ylabel=p_dset[yi].attrs["long_name"],
                                yunit=p_dset[yi].attrs["unit"],
                                symbol=symbols[i_p_dset % len(symbols)],
                                symbolSize=symbolSize,
                                # Oldest datasets fade more
                                color=colors[i_p_dset % len(colors)],
                                name=self._mk_legend(p_dset),
                                # to avoid passing the argument
                                **({"symbolBrush": symBrush} if symBrush else {})
                            )

                if dset:
                    has_settable = xi in _get_parnames(dset, "x")
                    has_gettable = yi in _get_parnames(dset, "y")
                    if has_settable and has_gettable:
                        # Real-time dataset
                        self.main_QtPlot.add(
                            x=dset[xi].values,
                            y=dset[yi].values,
                            subplot=plot_idx,
                            xlabel=dset[xi].attrs["long_name"],
                            xunit=dset[xi].attrs["unit"],
                            ylabel=dset[yi].attrs["long_name"],
                            yunit=dset[yi].attrs["unit"],
                            symbol="o",
                            symbolSize=5,
                            color=darker_color_cycle[0],
                            name=self._mk_legend(dset),
                        )
                        # We keep track only of the curves that need to be
                        # updated in real-time
                        self.curves.append(self.main_QtPlot.traces[-1])
                        # Manual counter is used because we may want to add more
                        # than one quantity per panel
                plot_idx += 1
            self.main_QtPlot.win.nextRow()

        # We loop again so that the first self.curves are reserved for
        # the live dataset

        #############################################################
        # Add a square heatmap

        if dset and dset.attrs["2D-grid"]:
            plot_idx = 1
            for yi in get_parnames:

                cmap = "viridis"
                zrange = None

                x = dset["x0"].values[: dset.attrs["xlen"]]
                y = dset["x1"].values[:: dset.attrs["xlen"]]
                z = np.reshape(dset[yi].values, (len(x), len(y)), order="F").T
                config_dict = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "xlabel": dset["x0"].attrs["long_name"],
                    "xunit": dset["x0"].attrs["unit"],
                    "ylabel": dset["x1"].attrs["long_name"],
                    "yunit": dset["x1"].attrs["unit"],
                    "zlabel": dset[yi].attrs["long_name"],
                    "zunit": dset[yi].attrs["unit"],
                    "subplot": plot_idx,
                    "cmap": cmap,
                }
                if zrange is not None:
                    config_dict["zrange"] = zrange
                self.secondary_QtPlot.add(**config_dict)
                plot_idx += 1

        #############################################################
        # if data is not on a grid but is 2D it makes sense to interpolate

        elif dset and len(set_parnames) == 2:
            plot_idx = 1
            for yi in get_parnames:

                cmap = "viridis"
                zrange = None

                config_dict = {
                    "x": [0, 1],
                    "y": [0, 1],
                    "z": np.zeros([2, 2]),
                    "xlabel": dset["x0"].attrs["long_name"],
                    "xunit": dset["x0"].attrs["unit"],
                    "ylabel": dset["x1"].attrs["long_name"],
                    "yunit": dset["x1"].attrs["unit"],
                    "zlabel": dset[yi].attrs["long_name"],
                    "zunit": dset[yi].attrs["unit"],
                    "subplot": plot_idx,
                    "cmap": cmap,
                }
                if zrange is not None:
                    config_dict["zrange"] = zrange
                self.secondary_QtPlot.add(**config_dict)
                self._im_curves.append(self.secondary_QtPlot.traces[-1])

                # used to mark the interpolation points
                self.secondary_QtPlot.add(
                    x=[],
                    y=[],
                    pen=None,
                    color=1.0,
                    width=0,
                    symbol="o",
                    symbolSize=4,
                    subplot=plot_idx,
                    xlabel=dset["x0"].attrs["long_name"],
                    xunit=dset["x0"].attrs["unit"],
                    ylabel=dset["x1"].attrs["long_name"],
                    yunit=dset["x1"].attrs["unit"],
                )
                self._im_scatters.append(self.secondary_QtPlot.traces[-1])

                # used to mark the last N-interpolation points
                self.secondary_QtPlot.add(
                    x=[],
                    y=[],
                    color=color_cycle[3],  # marks the point red
                    width=0,
                    symbol="o",
                    symbolSize=7,
                    subplot=plot_idx,
                    xlabel=dset["x0"].attrs["long_name"],
                    xunit=dset["x0"].attrs["unit"],
                    ylabel=dset["x1"].attrs["long_name"],
                    yunit=dset["x1"].attrs["unit"],
                )
                self._im_scatters_last.append(self.secondary_QtPlot.traces[-1])

                plot_idx += 1

        # Not running this can lead to nasty memory issues due to
        # accumulation of traces
        self.main_QtPlot.update_plot()
        self.secondary_QtPlot.update_plot()

    def update(self):
        # If tuid has changed, we need to initialize the figure
        if self.tuid() != self._last_tuid:
            # need to initialize the plot monitor
            self._initialize_plot_monitor()
            self._last_tuid = self.tuid()

        # otherwise we simply update
        elif self.tuid() is not None:
            dset = load_dataset(tuid=self.tuid())
            # This is necessary to be able to check for compatibility of
            # plotting together with persistent/previous datasets
            self._dset = dset
            set_parnames = _get_parnames(dset, "x")
            get_parnames = _get_parnames(dset, "y")
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
            if dset.attrs["2D-grid"]:
                for yidx, yi in enumerate(get_parnames):
                    Z = np.reshape(
                        dset[yi].values,
                        (dset.attrs["xlen"], dset.attrs["ylen"]),
                        order="F",
                    ).T
                    self.secondary_QtPlot.traces[yidx]["config"]["z"] = Z
                self.secondary_QtPlot.update_plot()

            #############################################################
            # if data is not on a grid but is 2D it makes sense to interpolate
            elif len(set_parnames) == 2:
                for yidx, yi in enumerate(get_parnames):
                    # exists to force reset the x- and y-axis scale
                    new_sc = TransformState(0, 1, True)

                    x = dset["x0"].values[~np.isnan(dset)["y0"]]
                    y = dset["x1"].values[~np.isnan(dset)["y0"]]
                    z = dset[yi].values[~np.isnan(dset)["y0"]]
                    # interpolation needs to be meaningful
                    if len(z) < 8:
                        break
                    x_grid, y_grid, z_grid = interpolate_heatmap(
                        x=x, y=y, z=z, interp_method="linear"
                    )

                    trace = self._im_curves[yidx]
                    trace["config"]["x"] = x_grid
                    trace["config"]["y"] = y_grid
                    trace["config"]["z"] = z_grid
                    # force rescale axis so marking datapoints works
                    trace["plot_object"]["scales"]["x"] = new_sc
                    trace["plot_object"]["scales"]["y"] = new_sc

                    # Mark all measured points on which the interpolation
                    # is based
                    trace = self._im_scatters[yidx]
                    trace["config"]["x"] = x
                    trace["config"]["y"] = y

                    trace = self._im_scatters_last[yidx]
                    trace["config"]["x"] = x[-5:]
                    trace["config"]["y"] = y[-5:]

                self.secondary_QtPlot.update_plot()

    def _ensure_matching_settbles(self):
        """
        Ensures plotting datasets are settables-compatible and discards
        datasets if necessary according to some priorities
        """
        pass


def _get_parnames(dset, par_type):
    return sorted(get_keys_containing(dset, par_type))
