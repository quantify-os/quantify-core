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
    dataset_name="My experiment",
    timestamp_start=pendulum.now().to_iso8601_string(),
    timestamp_end=pendulum.now().add(minutes=2).to_iso8601_string(),
    software_versions={
        "lab_fridge_magnet_driver": "v1.4.2",  # software version/tag
        "my_lab_repo": "9d8acf63f48c469c1b9fa9f2c3cf230845f67b18",  # git commit hash
    },
)
