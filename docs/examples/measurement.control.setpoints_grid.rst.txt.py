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

    We first prepare some utilities necessarily for the examples.
"""

# %%
rst_conf = {"indent": "    "}

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qcodes import ManualParameter, Parameter

import quantify_core.data.handling as dh
from quantify_core.measurement import MeasurementControl

dh.set_datadir(Path.home() / "quantify-data")
meas_ctrl = MeasurementControl("meas_ctrl")

par0 = ManualParameter(name="x0", label="X0", unit="s")
par1 = ManualParameter(name="x1", label="X1", unit="s")
par2 = ManualParameter(name="x2", label="X2", unit="s")
par3 = ManualParameter(name="x3", label="X3", unit="s")
sig = Parameter(name="sig", label="Signal", unit="V", get_cmd=lambda: np.exp(par0()))

# %% [raw]
"""
    .. admonition:: Iterative-only settables
        :class: dropdown, tip
"""

# %%
rst_conf = {"indent": "        "}

par0.batched = False
par1.batched = False
par2.batched = False

sig.batched = False

meas_ctrl.settables([par0, par1, par2])
meas_ctrl.setpoints_grid(
    [
        np.linspace(0, 1, 4),
        np.linspace(1, 2, 5),
        np.linspace(2, 3, 6),
    ]
)
meas_ctrl.gettables(sig)
dset = meas_ctrl.run("demo")
list(xr.plot.line(xi, label=name) for name, xi in dset.coords.items())
plt.gca().legend()

# %% [raw]
"""
    .. admonition:: Batched-only settables
        :class: dropdown, tip

        Note that the settable with lowest `.batch_size`  will be correspond to the
        innermost loop.
"""

# %%
rst_conf = {"indent": "        "}

par0.batched = True
par1.batch_size = 8
par1.batched = True
par1.batch_size = 8
par2.batched = True
par2.batch_size = 4

sig = Parameter(name="sig", label="Signal", unit="V", get_cmd=lambda: np.exp(par2()))
sig.batched = True
sig.batch_size = 32

meas_ctrl.settables([par0, par1, par2])
meas_ctrl.setpoints_grid(
    [
        np.linspace(0, 1, 3),
        np.linspace(1, 2, 5),
        np.linspace(2, 3, 4),
    ]
)
meas_ctrl.gettables(sig)
dset = meas_ctrl.run("demo")
list(xr.plot.line(xi, label=name) for name, xi in dset.coords.items())
plt.gca().legend()

# %% [raw]
"""
    .. admonition:: Batched and iterative settables
        :class: dropdown, tip

        Note that the settable with lowest `.batch_size`  will be correspond to the
        innermost loop. Furthermore, the iterative settables will be the outermost loops.
"""

# %%
rst_conf = {"indent": "        "}

par0.batched = False
par1.batched = True
par1.batch_size = 8
par2.batched = False
par3.batched = True
par3.batch_size = 4

sig = Parameter(name="sig", label="Signal", unit="V", get_cmd=lambda: np.exp(par3()))
sig.batched = True
sig.batch_size = 32

meas_ctrl.settables([par0, par1, par2, par3])
meas_ctrl.setpoints_grid(
    [
        np.linspace(0, 1, 3),
        np.linspace(1, 2, 5),
        np.linspace(2, 3, 4),
        np.linspace(3, 4, 6),
    ]
)
meas_ctrl.gettables(sig)
dset = meas_ctrl.run("demo")
list(xr.plot.line(xi, label=name) for name, xi in dset.coords.items())
plt.gca().legend()
