# ---
# jupyter:
#   jupytext:
#     cell_markers: \"\"\"
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
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


# %%
rst_conf = {"jupyter_execute_options": [":hide-code:", ":hide-output:"]}

from qcodes import Instrument

Instrument.close_all()

# %% [raw]
"""
.. admonition:: Example

    When running the analysis on a specific file some step of the analysis
    might fail. It is possible to run a partial analysis by interrupting its flow
    at a specific step.
"""

# %%
rst_conf = {"indent": "    ", "jupyter_execute_options": [":hide-code:"]}

from pathlib import Path

import numpy as np
from qcodes import ManualParameter, Parameter, validators

from quantify_core.analysis import base_analysis as ba
from quantify_core.data.handling import set_datadir
from quantify_core.measurement import MeasurementControl

formats = list(ba.settings["mpl_fig_formats"])
ba.settings["mpl_fig_formats"] = []

set_datadir(Path.home() / "quantify-data")

time_a = ManualParameter(
    name="time_a", label="Time A", unit="s", vals=validators.Numbers(), initial_value=1
)
signal = Parameter(
    name="sig_a", label="Signal A", unit="V", get_cmd=lambda: np.exp(time_a())
)

meas_ctrl = MeasurementControl("meas_ctrl")
meas_ctrl.settables(time_a)
meas_ctrl.gettables(signal)
meas_ctrl.setpoints(np.linspace(0, 5, 10))
dataset = meas_ctrl.run("2D-mock")


# %%
rst_conf = {"indent": "    "}

from quantify_core.analysis.base_analysis import BasicAnalysis

a_obj = BasicAnalysis(tuid=dataset.tuid).run_until(interrupt_before="run_fitting")

# We can also continue from a specific step
a_obj.run_from(step="run_fitting")
