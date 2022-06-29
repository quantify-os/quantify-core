# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Helpers for performing experiments."""
import warnings
from typing import Any, Optional, Union, Dict

import numpy as np

from qcodes.instrument import Instrument, InstrumentChannel

from quantify_core.data.handling import get_latest_tuid, load_snapshot
from quantify_core.data.types import TUID
from quantify_core.utilities.general import get_subclasses
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt


def load_settings_onto_instrument(
    instrument: Instrument, tuid: TUID = None, datadir: str = None
) -> None:
    """
    Loads settings from a previous experiment onto a current
    :class:`~qcodes.instrument.Instrument`. This information
    is loaded from the 'snapshot.json' file in the provided experiment
    directory.

    Parameters
    ----------
    instrument : :class:`~qcodes.instrument.Instrument`
        the instrument to be configured.
    tuid : :class:`~quantify_core.data.types.TUID`
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
    instruments_numpy_array = load_snapshot(tuid, datadir, list_to_ndarray=True)[
        "instruments"
    ]
    if instrument.name not in instruments:
        raise ValueError(
            'Instrument "{}" not found in snapshot {}:{}'.format(
                instrument.name, datadir, tuid
            )
        )

    instr_snap = instruments[instrument.name]
    instr_snap_numpy_array = instruments_numpy_array[instrument.name]

    def _try_to_set_par_safe(
        instr_mod: Union[Instrument, InstrumentChannel],
        parname: str,
        value: Any,
    ):
        """Tries to set a parameter and emits a warning if not successful."""
        # do not try to set parameters that are not settable
        if not "set" in dir(instr_mod.parameters[parname]):
            return
        # do not set to None if value is already None
        if instr_mod.parameters[parname]() is None and value is None:
            return

        # Make sure the parameter is actually a settable
        try:
            instr_mod.set(parname, value)
        except (RuntimeError, KeyError, ValueError, TypeError) as exc:
            warnings.warn(
                f'Parameter "{parname}" of "{instr_mod.name}" '
                f'could not be set to "{value}" due to error:\n{exc}'
            )

    def _set_params_instr_mod(
        instr_mod_snap: Dict,
        instr_mod_snap_np: Dict,
        instr_mod: Union[Instrument, InstrumentChannel],
    ):
        """
        private function to set parameters and recursively set parameters of submodules.
        """
        # iterate over top-level parameters
        for parname, par in instr_mod_snap["parameters"].items():
            # Check that the parameter exists in this instrument
            if parname in instr_mod.parameters:
                value = par["value"]
                if isinstance(instr_mod.parameters[parname](), np.ndarray):
                    value = instr_mod_snap_np["parameters"][parname]["value"]
                _try_to_set_par_safe(instr_mod, parname, value)
            else:
                warnings.warn(
                    f"Could not set parameter {parname} in {instr_mod.name}. "
                    f"{instr_mod.name} does not possess a parameter named {parname}."
                )
        # recursively call this function for all submodules
        if "submodules" in instr_mod_snap.keys():
            for module_name, module_snap in instr_mod_snap["submodules"].items():
                submodule = instr_mod.submodules[module_name]
                module_snap_np = instr_mod_snap_np["submodules"][module_name]
                _set_params_instr_mod(
                    instr_mod_snap=module_snap,
                    instr_mod_snap_np=module_snap_np,
                    instr_mod=submodule,
                )

    # set the top-level parameters and then recursively set parameters of submodules
    _set_params_instr_mod(
        instr_mod_snap=instr_snap,
        instr_mod_snap_np=instr_snap_numpy_array,
        instr_mod=instrument,
    )


def create_plotmon_from_historical(
    tuid: Optional[TUID] = None, label: str = ""
) -> PlotMonitor_pyqt:
    """
    Creates a plotmon using the dataset of the provided experiment denoted by the tuid
    in the datadir.
    Loads the data and draws any required figures.

    NB Creating a new plotmon can be slow. Consider using
    :func:`!PlotMonitor_pyqt.tuids_extra` to visualize dataset in the same plotmon.

    Parameters
    ----------
    tuid
        the TUID of the experiment.
    label
        if the `tuid` is not provided, as label will be used to search for the latest
        dataset.

    Returns
    -------
    :
        the plotmon
    """
    # avoid creating a plotmon with the same name
    name = tuid = tuid or get_latest_tuid(contains=label)
    name_str = str(name)
    i = 0

    for plotmon_class in get_subclasses(PlotMonitor_pyqt, include_base=True):
        for plotmon in plotmon_class.instances():
            if name_str == plotmon.name:
                name_str += f"_{i}"

    plotmon = PlotMonitor_pyqt(name)
    plotmon.tuids_append(tuid)
    plotmon.update()  # make sure everything is drawn

    return plotmon
