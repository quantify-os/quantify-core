# -----------------------------------------------------------------------------
# Description:    Module containing the pyqtgraph based plotting monitor.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import time
import PyQt5
import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from quantify.data.handling import snapshot

import warnings
from quantify.utilities import pprint_custom as pprint
import json
import re
import ast
import pickle


def traverse(obj):
    """
    Traversal implementation which recursively visits each node in a dict.
    We modify this function so that at the lowest hierarchy,
    we convert the element to a string.
    https://nvie.com/posts/modifying-deeply-nested-structures/
    """
    if isinstance(obj, dict):
        out_dict = {}
        for k, v in obj.items():
            out_dict[k] = traverse(v)
        return out_dict
    elif isinstance(obj, list):
        return [traverse(elem) for elem in obj]
    else:
        return str(obj)


def _recreate_snapshot_dict(unpickleable_snapshot: dict):
    """
    This function is used internally as a fallback option if a snapshot contains
    any entries or values which cannot be pickled. When this happens, the
    :meth:`~quantify.visualization.instrument.instrument_monitor.update` function
    will call this to make a string representation of the current snaphot.
    The snapshot string will be located in the key ['snapshot_string']['parameters']['snapshot']['value']
    """
    snap_collated = traverse(unpickleable_snapshot)
    return snap_collated


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
        self.add_parameter('update_interval',
                           unit='s',
                           vals=vals.Numbers(min_value=0.001),
                           initial_value=5,
                           parameter_class=ManualParameter)
        if remote:
            if not self.__class__.proc:
                self._init_qt()
        else:
            # overrule the remote pyqtgraph class
            self.rpg = pg
        # initial value is fake but ensures it will update the first time
        self.last_update_time = 0
        self.create_tree(window_size=window_size)

    def update(self):
        """
        Updates the Qc widget with the current snapshot of the instruments.
        This function is also called within the class :class:`~quantify.measurement.control.MeasurementControl`
        in the function :meth:`~quantify.measurement.control.MeasurementControl.run`.
        """
        time_since_last_update = time.time()-self.last_update_time
        if time_since_last_update > self.update_interval():
            self.last_update_time = time.time()
            # Take an updated, clean snapshot
            snap = snapshot(update=False, clean=True)
            try:
                self.tree.setData(snap['instruments'])
            except AttributeError as e:
                # This is to catch any potential pickling problems with the snapshot.
                snap_collated = _recreate_snapshot_dict(snap['instruments'])
                self.tree.setData(snap_collated)
                warnings.warn(f"Encountered: {e}", Warning)

    def _init_qt(self):
        # starting the process for the pyqtgraph plotting
        # You do not want a new process to be created every time you start a
        # run, so this only starts once and stores the process in the class
        pg.mkQApp()
        self.__class__.proc = pgmp.QtProcess()  # pyqtgraph multiprocessing
        self.__class__.rpg = self.proc._import('pyqtgraph')
        ins_mon_mod = 'quantify.visualization.ins_mon_widget.qc_snapshot_widget'
        self.__class__.rpg = self.proc._import(ins_mon_mod)

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

        self.tree = self.rpg.QcSnaphotWidget()
        self.update()
        self.tree.show()
        self.tree.setWindowTitle(self.name)
        self.tree.resize(*window_size)
