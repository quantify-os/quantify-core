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
import pprint


def _recreate_snapshot_dict(unpickleable_snapshot: dict):
    snap_corrected_string = pprint.pformat(unpickleable_snapshot)
    snap_corrected_string = snap_corrected_string.replace("'", "\"")
    snap_collated = {'snapshot_string':
                            {'name': 'snapshot_string',
                             'parameters':
                                    {'snapshot':
                                            {
                                                'ts': 'latest',
                                                'label': "",
                                                'unit': '',
                                                'name': 'string_representation',
                                                'value': snap_corrected_string
                                            }
                                    }
                            }
                    }
    return snap_collated


class InstrumentMonitor(Instrument):
    """
    Creates a pyqtgraph widget that displays the instrument monitor window.
    """
    proc = None
    rpg = None

    def __init__(self, name,
                 figsize=(600, 600),
                 window_title='', theme=((60, 60, 60), 'w'),
                 show_window=True, remote=True, **kwargs):
        """
        Initializes the plotting window
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
        self.create_tree(figsize=figsize)


    def update(self):
        time_since_last_update = time.time()-self.last_update_time
        if time_since_last_update > self.update_interval():
            self.last_update_time = time.time()
            snap = snapshot(update=False, clean=True)  # Take an updated, clean snapshot
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


    def create_tree(self, figsize=(1000, 600)):
        self.tree = self.rpg.QcSnaphotWidget()
        self.update()
        self.tree.show()
        self.tree.setWindowTitle(self.name)
        self.tree.resize(*figsize)


