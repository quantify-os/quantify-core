# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Module containing the pyqtgraph based plotting monitor."""
import warnings

import pyqtgraph.multiprocess as pgmp
from qcodes import validators as vals
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.helpers import strip_attrs

from quantify_core.data.handling import get_datadir
from quantify_core.measurement.control import _DATASET_LOCKS_DIR


# pylint: disable=invalid-name
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
        name
            Name of this instrument instance
        """
        super().__init__(name=name)

        # pyqtgraph multiprocessing
        # We setup a remote process which creates a queue to which
        # "commands" will be sent
        self.proc = pgmp.QtProcess(processRequests=False)
        # quantify_core module(s) in the remote process
        timeout = 60
        self._remote_quantify = self.proc._import("quantify_core", timeout=timeout)
        self._remote_ppr = self.proc._import(
            "quantify_core.visualization.pyqt_plotmon_remote", timeout=timeout
        )
        # the interface to the remote object
        self._remote_plotmon = self._remote_ppr.RemotePlotmon(
            instr_name=self.name, dataset_locks_dir=_DATASET_LOCKS_DIR
        )

        # `initial_cache_value` avoid set_cmd being called at __init__
        self.tuids_max_num = Parameter(
            vals=vals.Ints(min_value=1, max_value=100),
            set_cmd=self._set_tuids_max_num,
            get_cmd=self._get_tuids_max_num,
            initial_cache_value=3,
            name="tuids_max_num",
            instrument=self,
        )
        """The maximum number of auto-accumulated datasets in :attr:`.tuids`.
        Older dataset are discarded when :attr:`.tuids_append` is called [directly or
        from :meth:`.update`]."""

        # `initial_cache_value` avoid set_cmd being called at __init__
        self.tuids = Parameter(
            initial_cache_value=[],
            vals=vals.Lists(elt_validator=vals.Strings()),
            get_cmd=self._get_tuids,
            set_cmd=self._set_tuids,
            name="tuids",
            instrument=self,
        )
        """The tuids of the auto-accumulated previous datasets when specified through
        :attr:`.tuids_append`.
        Can be set to a list ``['tuid_one', 'tuid_two', ...]``.
        Can be reset by setting to ``[]``.
        See also :attr:`.tuids_extra`."""

        # `initial_cache_value` avoid set_cmd being called at __init__
        self.tuids_extra = Parameter(
            initial_cache_value=[],
            vals=vals.Lists(elt_validator=vals.Strings()),
            set_cmd=self._set_tuids_extra,
            get_cmd=self._get_tuids_extra,
            name="tuids_extra",
            instrument=self,
        )
        """Extra tuids whose datasets are never affected by :attr:`.tuids_append` or
        :attr:`.tuids_max_num`.
        As opposed to the :attr:`.tuids`, these ones never vanish.
        Can be reset by setting to ``[]``. Intended to perform realtime measurements and
        have a live comparison with previously measured datasets."""

        # Jupyter notebook support
        # pylint: disable=invalid-name
        self.main_QtPlot = QtPlotObjForJupyter(self._remote_plotmon, "main_QtPlot")
        """Retrieves the image of the main window when used as the final statement in a
        cell of a Jupyter-like notebook."""

        # pylint: disable=invalid-name
        self.secondary_QtPlot = QtPlotObjForJupyter(
            self._remote_plotmon, "secondary_QtPlot"
        )
        """Retrieves the image of the secondary window when used as the final statement
        in a cell of a Jupyter-like notebook."""

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
    # e.g. self._remote_plotmon.update(tuid, _callSync="off")

    def create_plot_monitor(self):
        """
        Creates the PyQtGraph plotting monitors.
        Can also be used to recreate these when plotting has crashed.
        """
        self._remote_plotmon.queue.put(("create_plot_monitor", tuple()))
        # Without queue it will be:
        # self._remote_plotmon.create_plot_monitor()

    def update(self, tuid: str = None):
        """
        Updates the curves/heatmaps of a specific dataset.

        If the dataset is not specified the latest dataset in :attr:`.tuids` is used.

        If :attr:`.tuids` is empty and ``tuid`` is provided
        then :meth:`tuids_append(tuid) <.tuids_append>` will be called.
        NB: this is intended mainly for MC to avoid issues when the file
        was not yet created or is empty.
        """
        try:
            self._remote_plotmon.queue.put(("update", (tuid, get_datadir())))
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(f"At update encountered: {e}", Warning)

    def tuids_append(self, tuid: str = None):
        """
        Appends a tuid to :attr:`.tuids` and also discards older datasets
        according to :attr:`.tuids_max_num`.

        The the corresponding data will be plotted in the main window
        with blue circles.

        NB: do not call before the corresponding dataset file was created and filled
        with data
        """
        self._remote_plotmon.queue.put(("tuids_append", (tuid, get_datadir())))

    def _set_tuids_max_num(self, val):
        self._remote_plotmon.queue.put(("_set_tuids_max_num", (val,)))

    def _set_tuids(self, tuids: list):
        self._remote_plotmon.queue.put(("_set_tuids", (tuids, get_datadir())))

    def _set_tuids_extra(self, tuids: list):
        self._remote_plotmon.queue.put(("_set_tuids_extra", (tuids, get_datadir())))

    # Blocking calls
    # For this ones we wait to get the return

    def _get_tuids_max_num(self):
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        return self._remote_plotmon._get_tuids_max_num()

    def _get_tuids(self):
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        return self._remote_plotmon._get_tuids()

    def _get_tuids_extra(self):
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        return self._remote_plotmon._get_tuids_extra()

    # Workaround for test due to pickling issues of certain objects
    def _get_curves_config(self):
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        return self._remote_plotmon._get_curves_config()

    def _get_traces_config(self, which="main_QtPlot"):
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        return self._remote_plotmon._get_traces_config(which)

    def close(self) -> None:
        """
        (Modified from Instrument class)

        Irreversibly stop this instrument and free its resources.

        Subclasses should override this if they have other specific
        resources to close.
        """
        if hasattr(self, "connection") and hasattr(self.connection, "close"):
            self.connection.close()

        # Essential!!! Close the process
        self.proc.join()

        strip_attrs(self, whitelist=["_name"])
        self.remove_instance(self)

    # pylint: disable=invalid-name
    def setGeometry_main(self, x: int, y: int, w: int, h: int):
        """Set the geometry of the main plotmon

        Parameters
        ----------
        x
            Horizontal position of the top-left corner of the window
        y
            Vertical position of the top-left corner of the window
        w
            Width of the window
        h
            Height of the window
        """
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        self._remote_plotmon._set_qt_plot_geometry(x, y, w, h, which="main_QtPlot")

    # pylint: disable=invalid-name
    def setGeometry_secondary(self, x: int, y: int, w: int, h: int):
        """Set the geometry of the secondary plotmon

        Parameters
        ----------
        x
            Horizontal position of the top-left corner of the window
        y
            Vertical position of the top-left corner of the window
        w
            Width of the window
        h
            Height of the window
        """
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        self._remote_plotmon._set_qt_plot_geometry(x, y, w, h, which="secondary_QtPlot")


# pylint: disable=too-few-public-methods
class QtPlotObjForJupyter:
    """
    A wrapper to be able to display a QtPlot window in Jupyter notebooks
    """

    def __init__(self, remote_plotmon, attr_name):
        # Save reference of the remote object
        self._remote_plotmon = remote_plotmon
        self.attr_name = attr_name

    def _repr_png_(self):
        # wait to finish the queue
        self._remote_plotmon._exec_queue()
        # always get the remote object, avoid keeping object references
        return getattr(self._remote_plotmon, self.attr_name)._repr_png_()
