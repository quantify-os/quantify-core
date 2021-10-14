# ---
# jupyter:
#   jupytext:
#     cell_markers: \"\"\"
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
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

from quantify_core.utilities.examples_support import mk_trace_for_iq_shot, mk_trace_time

SHOT = 0.6 + 1.2j

time = mk_trace_time()
trace = mk_trace_for_iq_shot(SHOT)

fig, ax = plt.subplots(1, 1, figsize=(12, 12 / 1.61 / 2))
_ = ax.plot(time * 1e6, trace.imag, ".-", label="I-quadrature")
_ = ax.plot(time * 1e6, trace.real, ".-", label="Q-quadrature")
_ = ax.set_xlabel("Time [µs]")
_ = ax.set_ylabel("Amplitude [V]")
_ = ax.legend()
