# -----------------------------------------------------------------------------
# Description:    Module containing the pyqtgraph based plotting monitor.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------

from qcodes import validators as vals
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.helpers import strip_attrs

import pyqtgraph.multiprocess as pgmp
from quantify.data.handling import get_datadir

import warnings


class PlotMonitor_pyqt(Instrument):
    """
    Pyqtgraph based plot monitor instrument.

    A plot monitor is intended to provide a real-time visualization of a dataset.

    The interaction with this virtual instrument are virtually instantaneous.
    All the heavier computations and plotting happens in a separate QtProcess.
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

        # pyqtgraph multiprocessing
        # We setup a remote process which creates a queue to which
        # "commands" will be sent
        self.proc = pgmp.QtProcess(processRequests=False)
        # quantify module(s) in the remote process
        self.remote_quantify = self.proc._import("quantify")
        self.remote_ppr = self.proc._import(
            "quantify.visualization.pyqt_plotmon_remote"
        )
        datadir = get_datadir()
        # the interface to the remote object
        self.remote_plotmon = self.remote_ppr.RemotePlotmon(
            instr_name=self.name, datadir=datadir
        )

        self.add_parameter(
            name="tuids_max_num",
            docstring=(
                "The maximum number of auto-accumulated datasets in "
                "`.tuids()`.\n"
                "Older dataset are discarded when `.tuids_append()` is "
                "called [directly or from `.update(tuid)`]"
            ),
            parameter_class=Parameter,
            vals=vals.Ints(min_value=1, max_value=100),
            set_cmd=self._set_tuids_max_num,
            get_cmd=self._get_tuids_max_num,
            # avoid set_cmd being called at __init__
            initial_cache_value=3,
        )
        self.add_parameter(
            name="tuids",
            docstring=(
                "The tuids of the auto-accumulated previous datasets when "
                "specified through `.tuids_append()`.\n"
                "Can also be set to any list `['tuid_one', 'tuid_two', ...]`\n"
                "Can be reset by setting to `[]`\n"
                "See also `tuids_extra`."
            ),
            parameter_class=Parameter,
            get_cmd=self._get_tuids,
            set_cmd=self._set_tuids,
            # avoid set_cmd being called at __init__
            initial_cache_value=[],
        )

        self.add_parameter(
            name="tuids_extra",
            docstring=(
                "Extra tuids whose datasets are never affected by "
                "`.tuids_append()` or `.tuids_max_num()`.\n"
                "As opposed to the `.tuids()`, these ones never vanish.\n"
                "Can be reset by setting to `[]`.\n"
                "Intended to perform realtime measurements and have a "
                "live comparison with previously measured datasets."
            ),
            parameter_class=Parameter,
            vals=vals.Lists(),
            set_cmd=self._set_tuids_extra,
            get_cmd=self._get_tuids_extra,
            # avoid set_cmd being called at __init__
            initial_cache_value=[],
        )

        # Jupyter notebook support

        self.main_QtPlot = QtPlotObjForJupyter(self.remote_plotmon, "main_QtPlot")
        self.secondary_QtPlot = QtPlotObjForJupyter(self.remote_plotmon, "secondary_QtPlot")

    # Wrappers for the remote methods
    # We just put "commands" on a queue that will be consumed by the
    # remote_plotmon
    # the commands are just a tuple:
    # (
    #   <str: attr to be called in the remote process>,
    #   <tuple: a tuple with the arguments passed to the attr>
    # )
    # see `remote_plotmon._exec_queue`

    # For consistency we mirror the label of all methods and set_cmd/get_cmd's
    # with the remote_plotmon

    # NB: before implementing the queue, _callSync="off" could be used
    # to avoid waiting for a return
    # e.g. self.remote_plotmon.update(tuid, _callSync="off")

    def create_plot_monitor(self):
        """
        Creates the PyQtGraph plotting monitors.
        Can also be used to recreate these when plotting has crashed.
        """
        self.remote_plotmon.queue.put(("create_plot_monitor", tuple()))
        # Without queue it will be:
        # self.remote_plotmon.create_plot_monitor()

    def update(self, tuid: str = None):
        """
        Updates the curves/heatmaps os a specific dataset.

        If the dataset is not specified the latest on in `.tuids()`
        is used.

        If `.tuids()` is empty and `tuid` is provided
        then `.tuids_append(tuid)` will be called.
        NB: this is intended mainly for MC to avoid issues when the file
        was not yet created or is empty.
        """
        try:
            self.remote_plotmon.queue.put(("update", (tuid,)))
            # self.remote_plotmon.update(tuid)
        except Exception as e:
            warnings.warn(f"At update encountered: {e}", Warning)

    def tuids_append(self, tuid: str = None):
        """
        Appends a tuid to `.tuids()` and also discards older datasets
        according to `.tuids_max_num()`.

        The the corresponding data will be plotted in the main window
        with blue circles.

        NB: do not call before the corresponding dataset file was created and filled
        with data
        """
        self.remote_plotmon.queue.put(("tuids_append", (tuid,)))
        # self.remote_plotmon.tuids_append(tuid)

    def _set_tuids_max_num(self, val):
        self.remote_plotmon.queue.put(("_set_tuids_max_num", (val,)))
        # self.remote_plotmon._set_tuids_max_num(val)

    def _set_tuids(self, tuids: list):
        self.remote_plotmon.queue.put(("_set_tuids", (tuids,)))
        # self.remote_plotmon._set_tuids(tuids)

    def _set_tuids_extra(self, tuids: list):
        self.remote_plotmon.queue.put(("_set_tuids_extra", (tuids,)))
        # self.remote_plotmon._set_tuids_extra(tuids)

    # Blocking calls
    # For this ones we wait to get the return

    def _get_tuids_max_num(self):
        # wait to finish the queue
        self.remote_plotmon._exec_queue()
        return self.remote_plotmon._get_tuids_max_num()

    def _get_tuids(self):
        # wait to finish the queue
        self.remote_plotmon._exec_queue()
        return self.remote_plotmon._get_tuids()

    def _get_tuids_extra(self):
        # wait to finish the queue
        self.remote_plotmon._exec_queue()
        return self.remote_plotmon._get_tuids_extra()

    # Workaround for test due to pickling issues of certain objects
    def _get_curves_config(self):
        # wait to finish the queue
        self.remote_plotmon._exec_queue()
        return self.remote_plotmon._get_curves_config()

    def _get_traces_config(self, which="main_QtPlot"):
        # wait to finish the queue
        self.remote_plotmon._exec_queue()
        return self.remote_plotmon._get_traces_config(which)

    def close(self) -> None:
        """
        (Modified form Instrument class)

        Irreversibly stop this instrument and free its resources.

        Subclasses should override this if they have other specific
        resources to close.
        """
        if hasattr(self, 'connection') and hasattr(self.connection, 'close'):
            self.connection.close()

        # Essential!!!
        # Close the process
        self.proc.join()

        strip_attrs(self, whitelist=['_name'])
        self.remove_instance(self)


class QtPlotObjForJupyter:
    """
    A wrapper to be able to display a QtPlot window in Jupyter notebooks
    """

    def __init__(self, remote_plotmon, attr_name):
        # Save reference of the remote object
        self.remote_plotmon = remote_plotmon
        self.attr_name = attr_name

    def _repr_png_(self):
        # always get the remote object, avoid keeping object references
        return getattr(self.remote_plotmon, self.attr_name)._repr_png_()
