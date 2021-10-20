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
# pylint: disable=duplicate-code
# pylint: disable=wrong-import-position


# %%
import matplotlib.pyplot as plt

from quantify_core.utilities.examples_support import mk_iq_shots

center_0, center_1, center_2 = 0.6 + 1.2j, -0.2 + 0.5j, 0 + 1.5j

data = mk_iq_shots(
    100, sigmas=[0.1] * 2, centers=(center_0, center_1), probabilities=[0.3, 1 - 0.3]
)

fig, ax = plt.subplots()
ax.plot(data.real, data.imag, "o", label="Shots")
ax.plot(center_0.real, center_0.imag, "^", label="|0>", markersize=10)
ax.plot(center_1.real, center_1.imag, "d", label="|1>", markersize=10)
_ = ax.legend()

data = mk_iq_shots(
    200,
    sigmas=[0.1] * 3,
    centers=(center_0, center_1, center_2),
    probabilities=[0.35, 0.35, 1 - 0.35 - 0.35],
)

fig, ax = plt.subplots()
ax.plot(data.real, data.imag, "o", label="Shots")
ax.plot(center_0.real, center_0.imag, "^", label="|0>", markersize=10)
ax.plot(center_1.real, center_1.imag, "d", label="|1>", markersize=10)
ax.plot(center_2.real, center_2.imag, "*", label="|2>", markersize=10)
_ = ax.legend()
