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

import pendulum

from quantify_core.utilities import examples_support

examples_support.mk_dataset_attrs(
    dataset_name="Bias scan",
    timestamp_start=pendulum.now().to_iso8601_string(),
    timestamp_end=pendulum.now().add(minutes=2).to_iso8601_string(),
    dataset_state="done",
)
