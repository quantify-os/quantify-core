# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Helpers for performing experiments."""
import warnings
from qcodes import Instrument
from quantify.data.types import TUID
from quantify.data.handling import load_snapshot, get_latest_tuid
from quantify.visualization.pyqt_plotmon import PlotMonitor_pyqt


def load_settings_onto_instrument(
    instrument: Instrument, tuid: TUID = None, datadir: str = None
):
    """
    Loads settings from a previous experiment onto a current
    :class:`~qcodes.instrument.base.Instrument`. This information
    is loaded from the 'snapshot.json' file in the provided experiment
    directory.

    Parameters
    ----------
    instrument : :class:`~qcodes.instrument.base.Instrument`
        the instrument to be configured.
    tuid : :class:`~quantify.data.types.TUID`
        the TUID of the experiment. If None use latest TUID.
    datadir : str
        path of the data directory. If `None`, uses `get_datadir()` to
        determine the data directory.
    Raises
    ------
    ValueError
        if the provided instrument has no match in the loaded snapshot.
    """
    if tuid is None:
        tuid = get_latest_tuid()

    instruments = load_snapshot(tuid, datadir)["instruments"]
    if instrument.name not in instruments:
        raise ValueError(
            'Instrument "{}" not found in snapshot {}:{}'.format(
                instrument.name, datadir, tuid
            )
        )
    for parname, par in instruments[instrument.name]["parameters"].items():
        if (
            parname in instrument.__dict__["parameters"]
        ):  # Check that the parameter exists in this instrument
            if "set" in dir(
                instrument.__dict__["parameters"][parname]
            ):  # Make sure the parameter is actually a settable
                try:
                    instrument.set(parname, par["value"])
                except Exception as exp:
                    warnings.warn(
                        f"Parameter {parname} of instrument {instrument.name} could "
                        "not be set to {val} due to error:\n{exp}"
                    )
        else:
            warnings.warn(
                f"{instrument.name} does not possess a parameter {parname}. Could not "
                "set parameter."
            )


def create_plotmon_from_historical(tuid: TUID):
    """
    Creates a plotmon using the dataset of the provided experiment denoted by the tuid
    in the datadir.
    Loads the data and draws any required figures.

    Parameters
    ----------
    tuid : :class:`~quantify.data.types.TUID`
        the TUID of the experiment.
    Returns
    -------
    :class:`~quantify.visualization.PlotMonitor_pyqt`
        the plot
    """
    plot = PlotMonitor_pyqt(tuid)
    plot.tuids_append(tuid)
    return plot
