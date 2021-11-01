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


# %%
rst_conf = {"jupyter_execute_options": [":hide-code:", ":hide-output:"]}

from qcodes import Instrument

Instrument.close_all()


# %% [raw]
"""
.. admonition:: Examples
    :class: dropdown, tip
"""

# %%
rst_conf = {"indent": "    "}

from pathlib import Path

import numpy as np
from qcodes import ManualParameter, Parameter, validators

from quantify_core.data.handling import set_datadir, to_gridded_dataset
from quantify_core.measurement import MeasurementControl

set_datadir(Path.home() / "quantify-data")

time_a = ManualParameter(
    name="time_a", label="Time A", unit="s", vals=validators.Numbers(), initial_value=1
)
time_b = ManualParameter(
    name="time_b", label="Time B", unit="s", vals=validators.Numbers(), initial_value=1
)
signal = Parameter(
    name="sig_a",
    label="Signal A",
    unit="V",
    get_cmd=lambda: np.exp(time_a()) + 0.5 * np.exp(time_b()),
)

meas_ctrl = MeasurementControl("meas_ctrl")
meas_ctrl.settables([time_a, time_b])
meas_ctrl.gettables(signal)
meas_ctrl.setpoints_grid([np.linspace(0, 5, 10), np.linspace(5, 0, 12)])
dset = meas_ctrl.run("2D-single-float-valued-settable-gettable")

dset_grid = to_gridded_dataset(dset)

dset_grid.y0.plot(cmap="viridis")
