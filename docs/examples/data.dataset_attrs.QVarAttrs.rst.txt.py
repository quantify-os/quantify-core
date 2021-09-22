# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement

# ---
# jupyter:
#   jupytext:
#     cell_markers: '\"\"\"'
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
from quantify_core.utilities import examples_support

examples_support.mk_exp_var_attrs(experiment_coords=["time"])

# %%
examples_support.mk_cal_var_attrs(experiment_coords=["cal"])
