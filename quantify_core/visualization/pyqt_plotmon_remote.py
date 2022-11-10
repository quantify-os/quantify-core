# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing the pyqtgraph-based remote plotting monitor manager."""
from __future__ import annotations

import itertools
import os
import warnings
from collections import deque
from collections.abc import Iterable
from multiprocessing import Queue, Event

import numpy as np
from filelock import FileLock
from pyqtgraph.Qt import QtCore
from qcodes.plots.colors import color_cycle
from qcodes.plots.pyqtgraph import QtPlot, TransformState

from quantify_core.data.handling import (
    DATASET_NAME,
    _get_parnames,
    _locate_experiment_file,
    _xi_and_yi_match,
    load_dataset,
    set_datadir,
)
from quantify_core.data.types import TUID
from quantify_core.utilities.general import last_modified
from quantify_core.visualization import _appnope
from quantify_core.visualization.color_utilities import make_fadded_colors
from quantify_core.visualization.plot_interpolation import interpolate_heatmap

warnings.filterwarnings(
    action="ignore", category=RuntimeWarning, message=r"All-NaN slice"
)


class RemotePlotmon:  # pylint: disable=too-many-instance-attributes
    """
    A remote Pyqtgraph-based plot monitor manager.

    We make it remote to avoid bottlenecks with updates and initialization of
    the plots/windows.

    A plot monitor is intended to provide a real-time visualization of datasets.
    """

    color_map = "viridis"
    """The color_map to use for the secondary plot."""

    def __init__(self, instr_name: str, dataset_locks_dir: str):
        # Used to mirror the name of the instrument in the windows titles
        self.instr_name = instr_name

        # Use the same locking files as in the main process
        self.dataset_locks_dir = dataset_locks_dir

        # used to track the tuids of previous datasets
        self._tuids = deque()

        # Used to assign which dataset will be plotted on the secondary
        # window
        self._tuid_2d = None
        self._tuids_2d = None

        # keep all datasets in one place and update/remove as needed
        self._dsets = {}
        # keeps track of when a dataset was last modified
        self._last_modified = {}

        self._tuids_max_num = 3

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

        self.queue = Queue()
        self.timer_queue = None  # timer to check for new commands and update the plot
        self._update_interval_ms = 100

        self.timer_appnope = None
        self._appnope_refresh_ms = 30

        # Create plotting windows and start listening to commands on the
        # queue
        self.create_plot_monitor()

        # This will start a timer and periodically take care of the queue
        # of commands sent from the main process
        # This is based on Qt timer and needs to be attached to a QtObject

        # We attach it to the main window
        # This likely requires to not close the main window
        self._run()

    # ##################################################################
    # Multiprocessing communication and logic
    # ##################################################################

    # This is intended to be able to instantly liberate the main process

    def _run(self):
        """
        Start a timer in this process that calls `self._exec_queue`
        periodically.
        """
        # This line requires the QtPlot's to be created with `remote=False`
        self.timer_queue = QtCore.QTimer(self.main_QtPlot.win)
        self.timer_queue.timeout.connect(self._exec_queue)
        self.timer_queue.setSingleShot(True)  # Explicitly reload later for robustness
        self.is_stopped = Event()
        self.to_be_stopped = Event()
        self.timer_queue.start(self._update_interval_ms)

        if _appnope.requires_appnope():
            # Start a timer to ensure the App Nap of macOS does not idle this process.
            # The process is sent to App Nap after a window is minimized or not
            # visible for a few seconds, this ensure we avoid that.
            # If this is not performed very long and cryptic errors will rise
            # (which is fatal for a running measurement)

            # This line requires the QtPlot's to be created with `remote=False`
            self.timer_appnope = QtCore.QTimer(self.main_QtPlot.win)
            self.timer_appnope.timeout.connect(_appnope.refresh_nope)
            self.timer_appnope.start(self._appnope_refresh_ms)

    def _exec_queue(self):
        """
        To be called periodically by `_run` in order to execute the pending
        commands in `self.queue`
        """
        unique_updates = {}
        while not self.queue.empty():
            attr_name, args = self.queue.get()
            # collect unique update calls
            if attr_name != "update":
                getattr(self, attr_name)(*args)
            else:
                # Abusing ordered dict to get an ordered set
                unique_updates[(attr_name, args)] = None

        # Periodically issue an update if there are monitored tuids
        if len(unique_updates) == 0 and len(self._tuids) > 0:
            unique_updates[("update", tuple())] = None

        for attr_name, args in unique_updates.keys():
            getattr(self, attr_name)(*args)

        if self.to_be_stopped.is_set():
            self.is_stopped.set()
        else:  # Reload timer
            self.timer_queue.start(self._update_interval_ms)

    def stop(self) -> None:
        """
        To be called by main process to stop timer_queue QTimer before closing the remote process.
        """
        # It would be nicer to wait for is_stopped in this function, but this also blocks the QEventloop / QTimer and
        # results in a deadlock. This may be solved by putting QTimer in a separate thread.
        self.to_be_stopped.set()

    # ##################################################################

    def _pop_dset(self, tuid: str) -> None:
        self._dsets.pop(tuid, None)
        self._last_modified.pop(tuid, None)

    def _get_tuids_max_num(self):
        return self._tuids_max_num

    def _set_tuids_max_num(self, val):
        """
        used only to update relevant variables
        """
        init = len(self._tuids) > val
        self._pop_old_dsets(val)
        if init:
            # Only need to update if datasets were discarded
            self._initialize_plot_monitor()
        self._tuids_max_num = val

    def _pop_old_dsets(self, max_tuids):
        while len(self._tuids) > max_tuids:
            discard_tuid = self._tuids.pop()
            if discard_tuid not in self._tuids_extra:
                self._pop_dset(discard_tuid)

    def tuids_append(self, tuid: str, datadir: str):
        """Appends a tuid to be plotted"""
        # ensures the same datadir as in the main process
        set_datadir(datadir)
        # verify tuid
        TUID(tuid)

        dset = _safe_load_dataset(tuid, self.dataset_locks_dir)

        # Now we ensure all datasets are compatible to be plotted together

        # Last dataset has priority, reset the others
        if not _xi_and_yi_match(tuple(self._dsets[t] for t in self._tuids) + (dset,)):
            # Reset the previous datasets
            self._pop_old_dsets(max_tuids=0)

        if not _xi_and_yi_match(
            tuple(self._dsets[t] for t in self._tuids_extra) + (dset,)
        ):
            # Force reset the user-defined extra datasets
            # Must be manual otherwise we go in circles checking for _xi_and_yi_match
            _ = [self._pop_dset(t) for t in self._tuids_extra]  # discard dsets
            self._tuids_extra = []

        self._tuids.appendleft(tuid)
        self._dsets[tuid] = dset

        # discard older datasets when max_num overflows
        self._pop_old_dsets(self._tuids_max_num)

        self._initialize_plot_monitor()

    def _get_tuids(self):
        return list(self._tuids)

    def _set_tuids(self, tuids: Iterable, datadir: str):
        """
        Set cmd for tuids
        """
        # ensures the same datadir as in the main process
        set_datadir(datadir)

        dsets = {
            tuid: _safe_load_dataset(tuid, self.dataset_locks_dir) for tuid in tuids
        }

        # Now we ensure all datasets are compatible to be plotted together
        if dsets and not _xi_and_yi_match(dsets.values()):
            raise NotImplementedError(
                "Datasets with different x and/or y variables not supported"
            )

        # it is enough to compare one dataset from each dict
        if dsets and not _xi_and_yi_match(
            itertools.chain(dsets.values(), self._dsets.values())
        ):
            # Reset the extra tuids
            _ = [self._pop_dset(t) for t in self._tuids_extra if t not in tuids]

        # Discard old dsets
        _ = [self._pop_dset(t) for t in self._tuids if t not in self._tuids_extra]

        self._tuids = deque(tuids)
        self._dsets.update(dsets)

        self._initialize_plot_monitor()
        return True

    def _get_tuids_extra(self):
        """
        Set cmd for tuids_extra
        """
        return self._tuids_extra

    def _set_tuids_extra(self, tuids: Iterable, datadir: str):
        """
        Set cmd for tuids_extra
        """
        # ensures the same datadir as in the main process
        set_datadir(datadir)

        extra_dsets = {
            tuid: _safe_load_dataset(tuid, self.dataset_locks_dir) for tuid in tuids
        }

        # Now we ensure all datasets are compatible to be plotted together

        if extra_dsets and not _xi_and_yi_match(extra_dsets.values()):
            raise NotImplementedError(
                "Datasets with different x and/or y variables not supported"
            )

        # it is enough to compare one dataset from each dict
        if extra_dsets and not _xi_and_yi_match(
            itertools.chain(extra_dsets.values(), self._dsets.values())
        ):
            # Reset the tuids because the user has specified persistent dsets
            self._pop_old_dsets(max_tuids=0)

        # Discard old dsets
        _ = [self._dsets.pop(t, None) for t in self._tuids_extra if t not in tuids]

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
            self.main_QtPlot.win.close()  # Close window to prevent having multiple windows open
            del self.main_QtPlot
        if hasattr(self, "secondary_QtPlot"):
            self.secondary_QtPlot.win.close()  # Close window to prevent having multiple windows open
            del self.secondary_QtPlot

        self.secondary_QtPlot = QtPlot(
            window_title=f"Secondary plotmon of {self.instr_name}",
            figsize=(600, 400),
            # We want the process to run locally to be able to
            # attach a QtTimer for the main loop of this process
            remote=False,
        )
        # Create last to appear on top
        self.main_QtPlot = QtPlot(
            window_title=f"Main plotmon of {self.instr_name}",
            figsize=(600, 400),
            remote=False,
        )

    # pylint: disable=too-many-statements, too-many-locals, too-many-branches
    def _initialize_plot_monitor(self):
        """
        Clears data in plot monitors and sets it up with the data from the dataset
        """

        # Clear the plot monitors if required.
        if self.main_QtPlot.traces:
            self.main_QtPlot.clear()

        if self.secondary_QtPlot.traces:
            self.secondary_QtPlot.clear()

        self.curves = {}

        if not self._dsets:
            # Nothing to be done
            return

        # we have forced all xi and yi to match so any dset will do here
        a_dset = next(iter(self._dsets.values()))
        set_parnames = _get_parnames(a_dset, par_type="x")
        get_parnames = _get_parnames(a_dset, par_type="y")

        ################# MAIN PLOT #################################

        fadded_colors = make_fadded_colors(
            num=len(self._tuids), color=color_cycle[0], to_hex=True
        )
        extra_colors = tuple(
            self.colors[i % len(self.colors)] for i in range(len(self._tuids_extra))
        )
        all_colors = fadded_colors + extra_colors
        all_colors = tuple(reversed(all_colors))
        # We reserve "o" symbol for the latest dataset
        symbols = tuple(
            self.symbols[i % len(self.symbols)]
            for i in range(len(all_colors) - bool(fadded_colors))
        )
        # In case only extra datasets are present
        symbols = (symbols + ("o",)) if fadded_colors else symbols
        symbols_brush = fadded_colors + ((0, 0, 0, 0),) * len(self._tuids_extra)
        symbols_brush = tuple(reversed(symbols_brush))

        plot_idx = 1
        for yi in get_parnames:
            for xi in set_parnames:
                for tuid, symb, symb_brush, color in zip(
                    itertools.chain(reversed(self._tuids_extra), reversed(self._tuids)),
                    symbols,
                    symbols_brush,
                    all_colors,
                ):
                    dset = self._dsets[tuid]
                    self.main_QtPlot.add(
                        x=dset[xi].values,
                        y=dset[yi].values,
                        subplot=plot_idx,
                        xlabel=dset[xi].attrs["long_name"],
                        xunit=dset[xi].attrs["units"],
                        ylabel=dset[yi].attrs["long_name"],
                        yunit=dset[yi].attrs["units"],
                        symbol=symb,
                        symbolSize=6,
                        symbolPen=color,
                        symbolBrush=symb_brush,
                        color=color,
                        name=_mk_legend(dset),
                    )

                    # Keep track of all traces so that any curves can be updated
                    if tuid not in self.curves:
                        self.curves[tuid] = {}
                    self.curves[tuid][xi + yi] = self.main_QtPlot.traces[-1]

                # Manual counter is used because we may want to add more
                # than one quantity per panel
                plot_idx += 1
            self.main_QtPlot.win.nextRow()

        self.main_QtPlot.update_plot()

        ################# SECONDARY PLOT ############################
        # On the secondary window we plot the first dset that is 2D
        # Below are some "extra" checks that are not currently strictly required

        all_tuids_it = itertools.chain(self._tuids, self._tuids_extra)
        self._tuids_2d = tuple(
            tuid
            for tuid in all_tuids_it
            if (len(_get_parnames(self._dsets[tuid], "x")) == 2)
        )
        self._tuid_2d = self._tuids_2d[0] if self._tuids_2d else None

        dset = self._dsets[self._tuid_2d] if self._tuid_2d else None
        if dset is not None:
            plot_info_xy = {
                "xlabel": dset["x0"].attrs["long_name"],
                "xunit": dset["x0"].attrs["units"],
                "ylabel": dset["x1"].attrs["long_name"],
                "yunit": dset["x1"].attrs["units"],
            }

        # Add a square heatmap

        self._im_curves = []
        self._im_scatters = []
        self._im_scatters_last = []

        is_uniformly_spaced = dset and dset.attrs.get(
            "grid_2d_uniformly_spaced", dset.attrs.get("2D-grid", False)
        )  # "2D-grid" is for legacy datasets support
        if is_uniformly_spaced:
            plot_idx = 1
            for yi in get_parnames:
                x = dset["x0"].values[: dset.attrs["xlen"]]
                y = dset["x1"].values[:: dset.attrs["xlen"]]
                z = np.reshape(dset[yi].values, (len(x), len(y)), order="F").T
                config_dict = {
                    "x": x,
                    "y": y,
                    "z": z,
                    **plot_info_xy,
                    "zlabel": dset[yi].attrs["long_name"],
                    "zunit": dset[yi].attrs["units"],
                    "subplot": plot_idx,
                    "cmap": self.color_map,
                }
                self.secondary_QtPlot.add(**config_dict)
                plot_idx += 1

        #############################################################
        # if data has two uniformly spaced settables in 1D, do not interpolate
        # (because it fails) and just put the values on the diagonal
        elif (
            dset
            and len(set_parnames) == 2
            and dset.attrs["1d_2_settables_uniformly_spaced"]
        ):
            plot_idx = 1
            x = dset["x0"].values
            y = dset["x1"].values
            shape = (len(x), len(y))
            for yi in get_parnames:
                # the background values of z_matrix are filled to be np.nan. Note that
                # qcodes.plots.pyqtgraph.QtPlot._update_image converts np.nan to the
                # minimum value though it should be possible to have NaNs in pyqtgraph
                z_matrix = np.full(shape, np.nan)
                np.fill_diagonal(z_matrix, dset[yi].values)
                z = np.reshape(z_matrix, shape, order="F").T
                config_dict = {
                    "x": x,
                    "y": y,
                    "z": z,
                    **plot_info_xy,
                    "zlabel": dset[yi].attrs["long_name"],
                    "zunit": dset[yi].attrs["units"],
                    "subplot": plot_idx,
                    "cmap": self.color_map,
                }
                self.secondary_QtPlot.add(**config_dict)
                self._im_curves.append(self.secondary_QtPlot.traces[-1])
                plot_idx += 1

        #############################################################
        # if data is not on a uniformly spaced grid but is 2D then interpolate
        elif dset and len(set_parnames) == 2:
            plot_idx = 1
            for yi in get_parnames:
                config_dict = {
                    "x": [0, 1],
                    "y": [0, 1],
                    "z": np.zeros([2, 2]),
                    **plot_info_xy,
                    "zlabel": dset[yi].attrs["long_name"],
                    "zunit": dset[yi].attrs["units"],
                    "subplot": plot_idx,
                    "cmap": self.color_map,
                }
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
                    **plot_info_xy,
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
                    **plot_info_xy,
                )
                self._im_scatters_last.append(self.secondary_QtPlot.traces[-1])
                plot_idx += 1

        self.secondary_QtPlot.update_plot()

    # pylint: disable=too-many-locals
    def update(self, tuid: str = None, datadir: str = None):
        """
        Updates the plots to reflect the latest data.
        """

        if tuid is not None and tuid not in self._dsets:
            # makes it easy to directly add a dataset and monitor it
            # this avoids having to set the tuid before the file was created
            self.tuids_append(tuid, datadir)

        if not self._dsets:
            # Nothing to update
            return

        tuid = self._tuids[0] if tuid is None else tuid

        # Only load a dataset again if it was modified
        last_modified_prev = self._last_modified.get(tuid, 0)
        self._last_modified[tuid] = _last_modified(tuid)
        if last_modified_prev < self._last_modified[tuid]:
            try:
                dset = _safe_load_dataset(tuid, self.dataset_locks_dir)
            except (ValueError, KeyError, OSError) as e:
                return
            self._dsets[tuid] = dset
        else:
            # Nothing to be done here, skip any plot updates
            return

        try:
            dset = _safe_load_dataset(tuid, self.dataset_locks_dir)
        except (ValueError, KeyError, OSError) as e:
            return
        self._dsets[tuid] = dset

        set_parnames = _get_parnames(dset, "x")
        get_parnames = _get_parnames(dset, "y")

        update_2d = tuid is not None and tuid == self._tuid_2d

        ################# MAIN PLOT #################################
        for yi in get_parnames:
            for xi in set_parnames:
                key = xi + yi
                self.curves[tuid][key]["config"]["x"] = dset[xi].values
                self.curves[tuid][key]["config"]["y"] = dset[yi].values

        try:
            self.main_QtPlot.update_plot()
        except RuntimeError:  # Suppress RuntimeError: wrapped C/C++ object of type PlotDataItem has been deleted
            return

        ################# SECONDARY PLOT ############################
        # Add a square heatmap
        is_uniformly_spaced = dset.attrs.get(
            "grid_2d_uniformly_spaced", dset.attrs.get("2D-grid", False)
        )  # "2D-grid" is for legacy datasets support"
        if update_2d and is_uniformly_spaced:
            for yidx, yi in enumerate(get_parnames):
                z_data = np.reshape(
                    dset[yi].values,
                    (dset.attrs["xlen"], dset.attrs["ylen"]),
                    order="F",
                ).T
                self.secondary_QtPlot.traces[yidx]["config"]["z"] = z_data

        #############################################################
        # if data has two uniformly spaced settables in 1D, do not interpolate
        # (because it fails) and just put the values on the diagonal
        elif (
            update_2d
            and len(set_parnames) == 2
            and dset.attrs["1d_2_settables_uniformly_spaced"]
        ):
            for yidx, yi in enumerate(get_parnames):
                # the background values of z_matrix are filled to be np.nan.
                # Note that qcodes.plots.pyqtgraph.QtPlot._update_image converts
                # np.nan to the minimum value although it _should_ be possible to
                # have NaNs in pyqtgraph.
                shape = (len(dset["x0"].values), len(dset["x1"].values))
                z_matrix = np.full(shape, np.nan)
                np.fill_diagonal(z_matrix, dset[yi].values)
                z_data = np.reshape(z_matrix, shape, order="F").T
                self.secondary_QtPlot.traces[yidx]["config"]["z"] = z_data

        #############################################################
        # if data is not on a grid but is 2D it makes sense to interpolate
        elif update_2d and len(set_parnames) == 2:
            for yidx, yi in enumerate(get_parnames):
                # exists to force reset the x- and y-axis scale
                new_sc = TransformState(0, 1, True)

                x = dset["x0"].values[~np.isnan(dset)["y0"].data]
                y = dset["x1"].values[~np.isnan(dset)["y0"].data]
                z = dset[yi].values[~np.isnan(dset)["y0"].data]
                # interpolation needs to be meaningful
                if len(z) < 8:
                    break

                # If data is one-dimensional, skip plotting.
                # Scipy's LinearNDInterpolator cannot interpolate 1d data
                # (used in interpolate_heatmap).
                all_x_equal = (x == x[0]).all()
                all_y_equal = (y == y[0]).all()
                if all_x_equal or all_y_equal:
                    break

                x_grid, y_grid, z_grid = interpolate_heatmap(
                    x=x, y=y, z=z, interp_method="linear"
                )

                trace = self._im_curves[yidx]
                trace["config"].update({"x": x_grid, "y": y_grid, "z": z_grid})

                # force rescale axis so marking datapoints works
                trace["plot_object"]["scales"].update({"x": new_sc, "y": new_sc})

                # Mark all measured points on which the interpolation is based
                trace = self._im_scatters[yidx]
                trace["config"].update({"x": x, "y": y})

                trace = self._im_scatters_last[yidx]
                trace["config"].update({"x": x[-5:], "y": y[-5:]})

        self.secondary_QtPlot.update_plot()

    def _get_curves_config(self):
        """
        For testing purposes only, some objects cannot be pickled to be retrieved
        """
        curves_dict = {}
        for tuid, xiyi_dict in self.curves.items():
            curves_dict[tuid] = {}
            for xiyi, items in xiyi_dict.items():
                curves_dict[tuid][xiyi] = {"config": items["config"]}

        return curves_dict

    def _get_traces_config(self, which="main_QtPlot"):
        """
        For testing purposes only, some objects cannot be pickled to be sent
        """
        traces = [{"config": trace["config"]} for trace in getattr(self, which).traces]

        return traces

    # pylint: disable=too-many-arguments
    def _set_qt_plot_geometry(self, x, y, width, height, which="main_QtPlot"):
        """
        Sets position and size of the window on screen
        """
        getattr(self, which).win.setGeometry(x, y, width, height)

    def _get_qt_plot_geometry(self, which="main_QtPlot"):
        """
        Gets position and size of the window on screen
        """
        win = getattr(self, which).win
        return win.x(), win.y(), win.width(), win.height()


def _last_modified(tuid) -> float:
    return last_modified(_locate_experiment_file(tuid))


def _safe_load_dataset(tuid, dataset_locks_dir):
    lockfile = os.path.join(dataset_locks_dir, tuid[:26] + "-" + DATASET_NAME + ".lock")
    with FileLock(lockfile, 5):
        dset = load_dataset(tuid)

    return dset


def _mk_legend(dset):
    hh_mm_ss = dset.attrs["tuid"].split("-")[1]
    # HH:mm:SS
    return (
        ":".join([hh_mm_ss[:2], hh_mm_ss[2:4], hh_mm_ss[4:]]) + " " + dset.attrs["name"]
    )
