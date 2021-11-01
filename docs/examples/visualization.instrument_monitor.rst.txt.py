# ---
# jupyter:
#   jupytext:
#     cell_markers: \"\"\"
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
rst_conf = {"jupyter_execute_options": [":hide-code:"]}
# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=duplicate-code


# %% [raw]
"""
.. admonition:: Example
"""

# %%
rst_conf = {"indent": "    ", "jupyter_execute_options": [":hide-code:"]}
from qcodes import Instrument

Instrument.close_all()

# %%
rst_conf = {"indent": "    "}

from quantify_core.measurement import MeasurementControl
from quantify_core.visualization import InstrumentMonitor

meas_ctrl = MeasurementControl("meas_ctrl")
instrument_monitor = InstrumentMonitor("instrument_monitor")
meas_ctrl.instrument_monitor(instrument_monitor.name)
# Set True if you want to query the instruments about each parameter
# before updating the window. Can be slow due to communication overhead.
instrument_monitor.update_snapshot(False)
instrument_monitor.update(force=True)
