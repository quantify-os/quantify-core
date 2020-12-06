# -----------------------------------------------------------------------------
# Description:    Module containing the pyqtgraph based plotting monitor.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------

from qcodes import validators as vals
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter

# from qcodes.utils.helpers import strip_attrs
import pyqtgraph.multiprocess as pgmp
from quantify.data.handling import get_datadir


class PlotMonitor_pyqt(Instrument):
    """
    Pyqtgraph based plot monitor instrument.

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

        # pyqtgraph multiprocessing
        self.proc = pgmp.QtProcess(processRequests=False)
        # quantify module in the remote process
        self.remote_quantify = self.proc._import("quantify")
        self.remote_ppr = self.proc._import(
            "quantify.visualization.pyqt_plotmon_remote"
        )
        datadir = get_datadir()
        self.remote_plotmon = self.remote_ppr.RemotePlotmon(
            instr_name=self.name, datadir=datadir
        )

        self.add_parameter(
            name="tuids_max_num",
            docstring="The maximum number of auto-accumulated datasets in `tuids`",
            parameter_class=Parameter,
            vals=vals.Ints(min_value=0, max_value=100),
            set_cmd=self._set_tuids_max_num,
            get_cmd=self._get_tuids_max_num,
            # avoid set_cmd being called at __init__
            initial_cache_value=3,
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
            # avoid set_cmd being called at __init__
            initial_cache_value=[],
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
            get_cmd=self._get_tuids_extra,
            # avoid set_cmd being called at __init__
            initial_cache_value=[],
        )

        # Jupyter notebook support

        self.main_QtPlot = QtPlotObjForJupyter(self.remote_plotmon, "main_QtPlot")
        self.secondary_QtPlot = QtPlotObjForJupyter(
            self.remote_plotmon, "secondary_QtPlot"
        )

    # Wrappers for the remote methods
    # _callSync="off" avoids waiting for a return

    def create_plot_monitor(self):
        self.remote_plotmon.create_plot_monitor(_callSync="off")

    def update(self, tuid: str = None):
        # return None
        self.remote_plotmon.update(tuid, _callSync="off")

    def tuids_append(self, tuid):
        # return None
        self.remote_plotmon.tuids_append(tuid, _callSync="off")

    def _get_tuids_max_num(self):
        return self.remote_plotmon._get_tuids_max_num()

    def _set_tuids_max_num(self, val):
        self.remote_plotmon._set_tuids_max_num(val, _callSync="off")

    def _get_tuids(self):
        return self.remote_plotmon._get_tuids()

    def _set_tuids(self, tuids):
        # return None
        self.remote_plotmon._set_tuids(tuids, _callSync="off")

    def _get_tuids_extra(self):
        return self.remote_plotmon._get_tuids_extra()

    def _set_tuids_extra(self, tuids):
        # return None
        self.remote_plotmon._set_tuids_extra(tuids, _callSync="off")

    # Not sure if this is necessary at the moment, might kill QT parent process
    # that are still needed
    # def close(self) -> None:
    #     """
    #     (Modified form Instrument class)

    #     Irreversibly stop this instrument and free its resources.

    #     Subclasses should override this if they have other specific
    #     resources to close.
    #     """

    #     # Closing the process
    #     self.proc.join()

    #     strip_attrs(self, whitelist=['_name'])
    #     self.remove_instance(self)


class QtPlotObjForJupyter:
    """
    A wrapper to be able to display a QtPlot window in Jupyter notebooks
    """

    def __init__(self, remote_plotmon, attr_name):
        # Save reference of the remote object
        self.remote_plotmon = remote_plotmon
        self.attr_name = attr_name

    def _repr_png_(self):
        return getattr(self.remote_plotmon, self.attr_name)._repr_png_()
