# -----------------------------------------------------------------------------
# Description:    Module containing the pyqtgraph based plotting monitor.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import time
import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from quantify.data.handling import snapshot
from quantify.utilities.general import traverse_dict
from quantify.visualization.ins_mon_widget import qc_snapshot_widget

import warnings


class InstrumentMonitor(Instrument):
    """
    Creates a pyqtgraph widget that displays the instrument monitor window.

    Example:

        .. code-block:: python

            from quantify.measurement import MeasurementControl
            from quantify.visualization.instrument_monitor import InstrumentMonitor

            MC = MeasurementControl('MC')
            insmon = InstrumentMonitor("Ins Mon custom")
            MC.instrument_monitor(insmon.name)
            insmon.update()


    """

    proc = None
    rpg = None

    def __init__(self, name, window_size=(600, 600), remote=True, **kwargs):
        """
        Initializes the pyqtgraph window

        Parameters
        ----------
        name : str
            name of the :class:`~quantify.visualization.instrument_monitor.InstrumentMonitor` object
        window_size : tuple (width, height)
            The size of the :class:`~quantify.visualization.instrument_monitor.InstrumentMonitor` window in px
        remote : bool
            Switch to use a remote instance of the pyqtgraph class
        """
        super().__init__(name=name)
        self.add_parameter(
            "update_interval",
            unit="s",
            vals=vals.Numbers(min_value=0.001),
            initial_value=5,
            parameter_class=ManualParameter,
        )
        if remote:
            if not self.__class__.proc:
                self._init_qt()
        else:
            # overrule the remote modules
            self.rpg = pg
            self.rwidget = qc_snapshot_widget

        # initial value is fake but ensures it will update the first time
        self.last_update_time = 0
        self.create_tree(window_size=window_size)

    def update(self):
        """
        Updates the Qc widget with the current snapshot of the instruments.
        This function is also called within the class :class:`~quantify.measurement.control.MeasurementControl`
        in the function :meth:`~quantify.measurement.control.MeasurementControl.run`.
        """
        time_since_last_update = time.time() - self.last_update_time
        if time_since_last_update > self.update_interval():
            self.last_update_time = time.time()
            # Take an updated, clean snapshot
            snap = snapshot(update=False, clean=True)
            try:
                self.tree.setData(snap["instruments"])
            except AttributeError as e:
                # This is to catch any potential pickling problems with the snapshot.
                # We do so by converting all lowest elements of the snapshot to string.
                snap_collated = traverse_dict(snap["instruments"])
                self.tree.setData(snap_collated)
                warnings.warn(f"Encountered: {e}", Warning)

    def _init_qt(self):
        # starting the process for the pyqtgraph plotting
        # You do not want a new process to be created every time you start a
        # run, so this only starts once and stores the process in the class
        self.__class__.proc = pgmp.QtProcess(processRequests=False)  # pyqtgraph multiprocessing
        self.__class__.rpg = self.proc._import("pyqtgraph")
        qc_widget = "quantify.visualization.ins_mon_widget.qc_snapshot_widget"
        self.__class__.rwidget = self.proc._import(qc_widget)

    def create_tree(self, window_size=(1000, 600)):
        """
        Saves an instance of the :class:`~quantify.visualization.ins_mon_widget.qc_snapshot_widget.QcSnaphotWidget`
        class during startup. Creates the :class:`~quantify.data.handling.snapshot` tree to display within the
        remote widget window.

        Parameters
        ----------
        window_size : tuple (width, height)
            The size of the :class:`~quantify.visualization.instrument_monitor.InstrumentMonitor` window in px
        """

        self.tree = self.rwidget.QcSnaphotWidget()
        self.update()
        self.tree.show()
        self.tree.setWindowTitle(self.name)
        self.tree.resize(*window_size)
