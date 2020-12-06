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
        # deque([<oldest>, ..., <one before latest>])
        self._tuids = deque()
        # keep all datasets in one place and update/remove as needed
        self._dsets = dict()

        # extra permanent datasets that the user can plot at the same time
        self._tuids_extra = []

        # make sure each dataset gets a new symbol, see _get_next_symb
        self.symbols = deque(["o", "t", "s", "t1", "p", "t2", "h", "star", "t3", "d"])

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
                "Can also be set to any list `['tuid1', 'tuid2', ...]`\n"
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
            initial_cache_value=[]
        )

        self.create_plot_monitor()

    def tuids_append(self, tuid):
        """
        FIXME TBW
        """

        # verify tuid
        TUID(tuid)

        self._tuids.append(tuid)
        self._dsets[tuid] = load_dataset(self._last_tuid_prev)

        # Now we ensure all datasets are compatible to be plotted together

        # Last dataset has priority, reset the others
        if not _xi_and_yi_match(self._dsets[t] for t in self._tuids):
            # Reset the previous datasets
            self._pop_old_dsets(max_tuids=1)
        if not _xi_and_yi_match(self._dsets):
            # Force reset the user-defined extra datasets
            # Needs to be manual otherwise we go in circles checking for _xi_and_yi_match
            [self.dests[t].pop() for t in self._tuids_extra]  # discard dsets
            self._tuids_extra = []

        # discard older datasets if the max_num overflows
        self._pop_old_dsets(self.tuids_max_num())

    def _set_tuids_max_num(self, val):
        """
        used only to update relevant variables
        """
        init = len(self._tuids) > val
        self._pop_old_dsets(val)
        if init:
            # Only need to update if datasets were discarded
            self._initialize_plot_monitor()

    def _get_next_symb(self):
        symb = self.symbols[0]
        self.symbols.rotate(-1)
        return symb

    def _pop_old_dsets(self, max_tuids):
        while len(self._tuids) > max_tuids:
            discard_tuid = self._tuids.popleft()
            self._dsets.pop(discard_tuid)

    def _get_tuids(self):
        return list(self._tuids)

    def _set_tuids(self, tuids):
        """
        Set cmd for tuids
        """

        dsets = {tuid: load_dataset(tuid) for tuid in tuids}

        # Now we ensure all datasets are compatible to be plotted together
        if not _xi_and_yi_match(dsets):
            raise NotImplementedError("Datasets with different x and/or y variables not supported")

        # it is enough to compare one dataset from each dict
        if not _xi_and_yi_match([next(iter(dsets.values())), next(iter(self._dsets.values()))]):
            # Reset the extra tuids
            discard_tuid = self._tuids_extra.popleft()
            self._dsets.pop(discard_tuid)

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

        if not _xi_and_yi_match(extra_dsets):
            raise NotImplementedError("Datasets with different x and/or y variables not supported")

        # it is enough to compare one dataset from each dict
        if not _xi_and_yi_match([next(iter(extra_dsets.values())), next(iter(self._dsets.values()))]):
            # Reset the tuids because the user has specified persistent dsets
            self._pop_old_dsets(val=0)

        self._tuids_extra = tuids
        self._dsets.update(extra_dsets)

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

        self.curves = []
        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

        all_dsets = (
            self._dsets_extra
            + list(self._dsets)
            + ([self._dset] if self._dset else [])
        )

        if not len(all_dsets):
            # Nothing to be done
            return None

        # Any can do, we have forced all xi and yi yo match
        a_dset = next(iter(self._dsets.values()))
        set_parnames = sorted(_get_parnames(a_dset, par_type="x"))
        get_parnames = sorted(_get_parnames(a_dset, par_type="y"))

        #############################################################

        fadded_colors = make_fadded_colors(num=len(self._tuids), color=color_cycle[0], to_hex=True)
        extra_colors = (self.colors[i % len(self.colors)] for i in range(len(self._tuids_extra)))
        it_colors = itertools.chain(fadded_colors, extra_colors)

        plot_idx = 1
        for yi in get_parnames:
            for xi in set_parnames:
                for tuid in itertools.chain((t for t in self._tuids), (t for t in self._tuids_extra)):
                    dset = self._dsets[tuid]
                    self.main_QtPlot.add(
                        x=dset[xi].values,
                        y=dset[yi].values,
                        subplot=plot_idx,
                        xlabel=dset[xi].attrs["long_name"],
                        xunit=dset[xi].attrs["unit"],
                        ylabel=dset[yi].attrs["long_name"],
                        yunit=dset[yi].attrs["unit"],
                        symbol=self._get_next_symb(),
                        symbolSize=8,
                        color=next(it_colors),
                        name=self._mk_legend(dset),
                    )

                    # Keep track of all traces so that any curves can be updated
                    if tuid not in self.curves.keys():
                        self.curves[tuid] = dict()
                    self.curves[tuid][xi + yi](self.main_QtPlot.traces[-1])

                # Manual counter is used because we may want to add more
                # than one quantity per panel
                plot_idx += 1
            self.main_QtPlot.win.nextRow()

        #############################################################
        # Add a square heatmap

        # if dset is None and len(self._dsets_extra):
        #     # If there is not an "active" tuid we make the secundary plot
        #     # do the 2D plots of the first persistent dataset of the user
        #     dset = self._dsets_extra[0]

        # if dset and dset.attrs["2D-grid"]:
        #     plot_idx = 1
        #     for yi in get_parnames:

        #         cmap = "viridis"
        #         zrange = None

        #         x = dset["x0"].values[: dset.attrs["xlen"]]
        #         y = dset["x1"].values[:: dset.attrs["xlen"]]
        #         z = np.reshape(dset[yi].values, (len(x), len(y)), order="F").T
        #         config_dict = {
        #             "x": x,
        #             "y": y,
        #             "z": z,
        #             "xlabel": dset["x0"].attrs["long_name"],
        #             "xunit": dset["x0"].attrs["unit"],
        #             "ylabel": dset["x1"].attrs["long_name"],
        #             "yunit": dset["x1"].attrs["unit"],
        #             "zlabel": dset[yi].attrs["long_name"],
        #             "zunit": dset[yi].attrs["unit"],
        #             "subplot": plot_idx,
        #             "cmap": cmap,
        #         }
        #         if zrange is not None:
        #             config_dict["zrange"] = zrange
        #         self.secondary_QtPlot.add(**config_dict)
        #         plot_idx += 1

        #############################################################
        # if data is not on a grid but is 2D it makes sense to interpolate

        # elif dset and len(set_parnames) == 2:
        #     plot_idx = 1
        #     for yi in get_parnames:

        #         cmap = "viridis"
        #         zrange = None

        #         config_dict = {
        #             "x": [0, 1],
        #             "y": [0, 1],
        #             "z": np.zeros([2, 2]),
        #             "xlabel": dset["x0"].attrs["long_name"],
        #             "xunit": dset["x0"].attrs["unit"],
        #             "ylabel": dset["x1"].attrs["long_name"],
        #             "yunit": dset["x1"].attrs["unit"],
        #             "zlabel": dset[yi].attrs["long_name"],
        #             "zunit": dset[yi].attrs["unit"],
        #             "subplot": plot_idx,
        #             "cmap": cmap,
        #         }
        #         if zrange is not None:
        #             config_dict["zrange"] = zrange
        #         self.secondary_QtPlot.add(**config_dict)
        #         self._im_curves.append(self.secondary_QtPlot.traces[-1])

        #         # used to mark the interpolation points
        #         self.secondary_QtPlot.add(
        #             x=[],
        #             y=[],
        #             pen=None,
        #             color=1.0,
        #             width=0,
        #             symbol="o",
        #             symbolSize=4,
        #             subplot=plot_idx,
        #             xlabel=dset["x0"].attrs["long_name"],
        #             xunit=dset["x0"].attrs["unit"],
        #             ylabel=dset["x1"].attrs["long_name"],
        #             yunit=dset["x1"].attrs["unit"],
        #         )
        #         self._im_scatters.append(self.secondary_QtPlot.traces[-1])

        #         # used to mark the last N-interpolation points
        #         self.secondary_QtPlot.add(
        #             x=[],
        #             y=[],
        #             color=color_cycle[3],  # marks the point red
        #             width=0,
        #             symbol="o",
        #             symbolSize=7,
        #             subplot=plot_idx,
        #             xlabel=dset["x0"].attrs["long_name"],
        #             xunit=dset["x0"].attrs["unit"],
        #             ylabel=dset["x1"].attrs["long_name"],
        #             yunit=dset["x1"].attrs["unit"],
        #         )
        #         self._im_scatters_last.append(self.secondary_QtPlot.traces[-1])

        #         plot_idx += 1

        # # Not running this can lead to nasty memory issues due to
        # # accumulation of traces
        # self.main_QtPlot.update_plot()
        # self.secondary_QtPlot.update_plot()

    # def update(self):
    #     # If tuid has changed, we need to initialize the figure
    #     if self.tuid() != self._last_tuid:
    #         # need to initialize the plot monitor
    #         self._initialize_plot_monitor()
    #         self._last_tuid = self.tuid()

    #     # otherwise we simply update
    #     elif self.tuid() is not None:
    #         dset = load_dataset(tuid=self.tuid())
    #         # This is necessary to be able to check for compatibility of
    #         # plotting together with persistent/previous datasets
    #         self._dset = dset
    #         set_parnames = _get_parnames(dset, "x")
    #         get_parnames = _get_parnames(dset, "y")
    #         # Only updates the main monitor currently.

    #         #############################################################
    #         i = 0
    #         for yi in get_parnames:
    #             for xi in set_parnames:
    #                 self.curves[i]["config"]["x"] = dset[xi].values
    #                 self.curves[i]["config"]["y"] = dset[yi].values
    #                 i += 1
    #         self.main_QtPlot.update_plot()

    #         #############################################################
    #         # Add a square heatmap
    #         if dset.attrs["2D-grid"]:
    #             for yidx, yi in enumerate(get_parnames):
    #                 Z = np.reshape(
    #                     dset[yi].values,
    #                     (dset.attrs["xlen"], dset.attrs["ylen"]),
    #                     order="F",
    #                 ).T
    #                 self.secondary_QtPlot.traces[yidx]["config"]["z"] = Z
    #             self.secondary_QtPlot.update_plot()

    #         #############################################################
    #         # if data is not on a grid but is 2D it makes sense to interpolate
    #         elif len(set_parnames) == 2:
    #             for yidx, yi in enumerate(get_parnames):
    #                 # exists to force reset the x- and y-axis scale
    #                 new_sc = TransformState(0, 1, True)

    #                 x = dset["x0"].values[~np.isnan(dset)["y0"]]
    #                 y = dset["x1"].values[~np.isnan(dset)["y0"]]
    #                 z = dset[yi].values[~np.isnan(dset)["y0"]]
    #                 # interpolation needs to be meaningful
    #                 if len(z) < 8:
    #                     break
    #                 x_grid, y_grid, z_grid = interpolate_heatmap(
    #                     x=x, y=y, z=z, interp_method="linear"
    #                 )

    #                 trace = self._im_curves[yidx]
    #                 trace["config"]["x"] = x_grid
    #                 trace["config"]["y"] = y_grid
    #                 trace["config"]["z"] = z_grid
    #                 # force rescale axis so marking datapoints works
    #                 trace["plot_object"]["scales"]["x"] = new_sc
    #                 trace["plot_object"]["scales"]["y"] = new_sc

    #                 # Mark all measured points on which the interpolation
    #                 # is based
    #                 trace = self._im_scatters[yidx]
    #                 trace["config"]["x"] = x
    #                 trace["config"]["y"] = y

    #                 trace = self._im_scatters_last[yidx]
    #                 trace["config"]["x"] = x[-5:]
    #                 trace["config"]["y"] = y[-5:]

    #             self.secondary_QtPlot.update_plot()


def _get_parnames(dset, par_type):
    return sorted(get_keys_containing(dset, par_type))


def _mk_legend(dset):
    return dset.attrs["tuid"].split("-")[-1] + " " + dset.attrs["name"]
