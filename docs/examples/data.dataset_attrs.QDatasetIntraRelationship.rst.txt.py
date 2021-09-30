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

# %% [raw]
# This is how the attributes of a dataset containing a ``q0`` main variable and ``q0_cal`` secondary variables would look like.
# The ``q0_cal`` corresponds to calibrations datapoints.
# See :ref:`sec-quantify-dataset-examples` for examples with more context.

# %%
from quantify_core.utilities import examples_support
from quantify_core.data.dataset_attrs import QDatasetIntraRelationship
import pendulum

attrs = examples_support.mk_dataset_attrs(
    relationships=[
        QDatasetIntraRelationship(
            item_name="q0",
            relation_type="calibration",
            related_names=["q0_cal"],
        ).to_dict()
    ]
)
