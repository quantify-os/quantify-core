"""
-----------------------------------------------------------------------------
Description:    General utilities.
Repository:     https://gitlab.com/qblox/packages/software/quantify/
Copyright (C) Qblox BV (2020)
-----------------------------------------------------------------------------
"""
import pprint
from qcodes import Instrument
from quantify.data.types import TUID
from quantify.data.handling import load_snapshot


def setup_instrument(instrument: Instrument, tuid: TUID, datadir: str = None):
    instruments = load_snapshot(tuid, datadir)['instruments']
    if instrument.name not in instruments:
        raise ValueError('Instrument "{}" not found in snapshot {}:{}'.format(instrument.name, datadir, tuid))
    for parname, par in instruments[instrument.name]["parameters"].items():
        val = par["value"]
        if val:  # qcodes doesn't like setting to none
            instrument.set(parname, par["value"])
