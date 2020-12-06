# -----------------------------------------------------------------------------
# Description:    Module containing the pyqtgraph based plotting monitor.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import numpy as np
from collections import deque
import itertools

from qcodes import validators as vals
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.plots.colors import color_cycle
from qcodes.plots.pyqtgraph import QtPlot, TransformState

from quantify.utilities.general import get_keys_containing
from quantify.data.handling import load_dataset, _xi_and_yi_match
from quantify.visualization.plot_interpolation import interpolate_heatmap
from quantify.data.types import TUID
from .color_utilities import make_fadded_colors


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

        # used to track the tuids of previous datasets
        self._tuids = deque()
        # keep all datasets in one place and update/remove as needed
        self._dsets = dict()

        # extra permanent datasets that the user can plot at the same time
        self._tuids_extra = []

        # make sure each dataset gets a new symbol, see _get_next_symb
        self.symbols = ["t", "s", "t1", "p", "t2", "h", "star", "t3", "d"]

        # We reserve the first (fading blues color to the latest datasets)
        self.colors = color_cycle[1:]

        # convenient access to curve variable for updating
        self.curves = []
        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

        # Keep the rest at the end of __init__ due to some dependencies

        # Parameters are attributes that we include in logging
        # and intend the user to change.

        self.add_parameter(
            name="tuids_max_num",
            docstring="The maximum number of auto-accumulated datasets in `tuids`",
            parameter_class=Parameter,
            vals=vals.Ints(min_value=0, max_value=100),
            # avoid set_cmd being called at __init__
            initial_cache_value=2,
            set_cmd=self._set_tuids_max_num,
        )
        self.add_parameter(
            name="tuids",
            docstring=(
                "The tuids of the auto-accumulated previous datasets when "
                "specified through `tuids_append`.\n"
                "Can also be set to any list `['tuid_one', 'tuid_two', ...]`\n"
                "Can be reset by setting to `[]`\n"
                "See also `tuids_extra`."
            ),
            parameter_class=Parameter,
            get_cmd=self._get_tuids,
            set_cmd=self._set_tuids,
        )

        self.add_parameter(
            name="tuids_extra",
            docstring=(
                "Extra tuids whose datasets are never affected by `tuids_append` or `tuids_max_num`.\n"
                "As opposed to the `tuids`, these never vanish.\n"
                "Can be reset by setting to `[]`"
            ),
            parameter_class=Parameter,
            vals=vals.Lists(),
            set_cmd=self._set_tuids_extra,
            get_cmd=lambda: self._tuids_extra,
            # avoid set_cmd being called at __init__
            initial_cache_value=[],
        )

        self.create_plot_monitor()

    def _set_tuids_max_num(self, val):
        """
        used only to update relevant variables
        """
        init = len(self._tuids) > val
        self._pop_old_dsets(val)
        if init:
            # Only need to update if datasets were discarded
            self._initialize_plot_monitor()

    def _pop_old_dsets(self, max_tuids):
        while len(self._tuids) > max_tuids:
            discard_tuid = self._tuids.pop()
            self._dsets.pop(discard_tuid)

    def tuids_append(self, tuid):
        """
        FIXME TBW
        """

        # verify tuid
        TUID(tuid)

        self._tuids.appendleft(tuid)
        self._dsets[tuid] = load_dataset(tuid)

        # Now we ensure all datasets are compatible to be plotted together

        # Last dataset has priority, reset the others
        if not _xi_and_yi_match(self._dsets[t] for t in self._tuids):
            # Reset the previous datasets
            self._pop_old_dsets(max_tuids=1)
        if not _xi_and_yi_match(self._dsets.values()):
            # Force reset the user-defined extra datasets
            # Needs to be manual otherwise we go in circles checking for _xi_and_yi_match
            [self._dsets.pop(t, None) for t in self._tuids_extra]  # discard dsets
            self._tuids_extra = []

        # discard older datasets when max_num overflows
        self._pop_old_dsets(self.tuids_max_num())

        self._initialize_plot_monitor()

    def _get_tuids(self):
        return list(self._tuids)

    def _set_tuids(self, tuids):
        """
        Set cmd for tuids
        """

        dsets = {tuid: load_dataset(tuid) for tuid in tuids}

        # Now we ensure all datasets are compatible to be plotted together
        if dsets and not _xi_and_yi_match(dsets.values()):
            raise NotImplementedError(
                "Datasets with different x and/or y variables not supported"
            )

        # it is enough to compare one dataset from each dict
        if dsets and not _xi_and_yi_match(
            (next(iter(dsets.values())), next(iter(self._dsets.values())))
        ):
            # Reset the extra tuids
            [self._dsets.pop(t, None) for t in self._tuids_extra]

        # Discard old dsets
        [self._dsets.pop(t, None) for t in self._tuids]

        self._tuids = deque(tuids)
        self._dsets.update(dsets)

        self._initialize_plot_monitor()
        return True

    def _set_tuids_extra(self, tuids):
        """
        Set cmd for tuids_extra
        """
        extra_dsets = {tuid: load_dataset(tuid) for tuid in tuids}

        # Now we ensure all datasets are compatible to be plotted together

        if extra_dsets and not _xi_and_yi_match(extra_dsets.values()):
            raise NotImplementedError(
                "Datasets with different x and/or y variables not supported"
            )

        # it is enough to compare one dataset from each dict
        if extra_dsets and not _xi_and_yi_match(
            dset
            for dset in (
                next(iter(extra_dsets.values()), False),
                next(iter(self._dsets.values()), False),
            )
            if dset
        ):
            # Reset the tuids because the user has specified persistent dsets
            self._pop_old_dsets(max_tuids=0)

        # Discard old dsets
        [self._dsets.pop(t, None) for t in self._tuids_extra]

        self._dsets.update(extra_dsets)
        self._tuids_extra = tuids

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

    def _initialize_plot_monitor(self):
        """
        Clears data in plot monitors and sets it up with the data from the dataset
        """

        # Clear the plot monitors if required.
        if self.main_QtPlot.traces:
            self.main_QtPlot.clear()

        if self.secondary_QtPlot.traces:
            self.secondary_QtPlot.clear()

        self.curves = dict()

        if not len(self._dsets):
            # Nothing to be done
            return None

        # we have forced all xi and yi to match so any dset will do here
        a_dset = next(iter(self._dsets.values()))
        set_parnames = sorted(_get_parnames(a_dset, par_type="x"))
        get_parnames = sorted(_get_parnames(a_dset, par_type="y"))

        #############################################################

        fadded_colors = make_fadded_colors(
            num=len(self._tuids), color=color_cycle[0], to_hex=True
        )
        extra_colors = tuple(
            self.colors[i % len(self.colors)] for i in range(len(self._tuids_extra))
        )
        all_colors = fadded_colors + extra_colors
        # We reserve circle for the latest dataset
        symbols = ("o",) + tuple(
            self.symbols[i % len(self.symbols)] for i in range(len(all_colors) - 1)
        )

        plot_idx = 1
        for yi in get_parnames:
            for xi in set_parnames:
                for i_tuid, tuid in enumerate(
                    itertools.chain(
                        (t for t in self._tuids), (t for t in self._tuids_extra)
                    )
                ):
                    dset = self._dsets[tuid]
                    self.main_QtPlot.add(
                        x=dset[xi].values,
                        y=dset[yi].values,
                        subplot=plot_idx,
                        xlabel=dset[xi].attrs["long_name"],
                        xunit=dset[xi].attrs["unit"],
                        ylabel=dset[yi].attrs["long_name"],
                        yunit=dset[yi].attrs["unit"],
                        symbol=symbols[i_tuid],
                        symbolSize=9,
                        color=all_colors[i_tuid],
                        name=_mk_legend(dset),
                    )

                    # Keep track of all traces so that any curves can be updated
                    if tuid not in self.curves.keys():
                        self.curves[tuid] = dict()
                    self.curves[tuid][xi + yi] = self.main_QtPlot.traces[-1]

                # Manual counter is used because we may want to add more
                # than one quantity per panel
                plot_idx += 1
            self.main_QtPlot.win.nextRow()

        self.main_QtPlot.update_plot()

        #############################################################

        # On the secondary window we plot the first dset that is 2D
        # Below are some "extra" checks that are not currently strictly required

        all_tuids_it = itertools.chain(self._tuids, self._tuids_extra)
        self._tuids_2D = tuple(
            tuid
            for tuid in all_tuids_it
            if (len(_get_parnames(self._dsets[tuid], "x")) == 2)
        )

        dset = self._dsets.get(self._tuids_2D[0] if self._tuids_2D else "", None)

        # Add a square heatmap

        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

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

        self.secondary_QtPlot.update_plot()

    def update(self, tuid):
        dset = load_dataset(tuid)
        self._dsets[tuid] = dset

        set_parnames = _get_parnames(dset, "x")
        get_parnames = _get_parnames(dset, "y")

        update_2D = tuid == self._tuids_2D

        #############################################################

        for yi in get_parnames:
            for xi in set_parnames:
                key = xi + yi
                self.curves[tuid][key]["config"]["x"] = dset[xi].values
                self.curves[tuid][key]["config"]["y"] = dset[yi].values
        self.main_QtPlot.update_plot()

        #############################################################
        # Add a square heatmap
        if update_2D and dset.attrs["2D-grid"]:
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
        elif update_2D and len(set_parnames) == 2:
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


def _get_parnames(dset, par_type):
    return sorted(get_keys_containing(dset, par_type))


def _mk_legend(dset):
    return dset.attrs["tuid"].split("-")[-1] + " " + dset.attrs["name"]
