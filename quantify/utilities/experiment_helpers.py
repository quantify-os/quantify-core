# -----------------------------------------------------------------------------
# Description:  Helpers for performing experiments.
# Repository:   https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
from qcodes import Instrument
from quantify.data.types import TUID
from quantify.data.handling import load_snapshot
from quantify.visualization.pyqt_plotmon import PlotMonitor_pyqt


def load_settings_onto_instrument(instrument: Instrument, tuid: TUID, datadir: str = None):
    """
    Loads settings from a previous experiment onto a current :class:`~qcodes.instrument.base.Instrument`. This
    information is loaded from the 'snapshot.json' file in the provided experiment directory.

    Parameters
    ----------
    instrument : :class:`~qcodes.instrument.base.Instrument`
        the instrument to be configured.
    tuid : :class:`~quantify.data.types.TUID`
        the TUID of the experiment.
    datadir : str
        path of the data directory. If `None`, uses `get_datadir()` to determine the data directory.
    Raises
    ------
    ValueError
        if the provided instrument has no match in the loaded snapshot.
    """
    instruments = load_snapshot(tuid, datadir)['instruments']
    if instrument.name not in instruments:
        raise ValueError('Instrument "{}" not found in snapshot {}:{}'.format(instrument.name, datadir, tuid))
    for parname, par in instruments[instrument.name]["parameters"].items():
        val = par["value"]
        if val:  # qcodes doesn't like setting to none
            instrument.set(parname, par["value"])


def create_plotmon_from_historical(tuid: TUID):
    """
    Creates a plotmon using the dataset of the provided experiment denoted by the tuid in the datadir.
    Loads the data and draws any required figures.

    Parameters
    ----------
    tuid : :class:`~quantify.data.types.TUID`
        the TUID of the experiment.
    Returns
    -------
    :class:`quantify.visualization.pyqt_plotmon.PlotMonitor_pyqt`
        the plot
    """
    plot = PlotMonitor_pyqt(tuid)
    plot.tuids_append(tuid)
    return plot
