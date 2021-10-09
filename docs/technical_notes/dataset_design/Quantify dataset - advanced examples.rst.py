# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
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
rst_json_conf = {"jupyter_execute_options": [":hide-code:"]}
# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=pointless-statement
# pylint: disable=invalid-name
# pylint: disable=expression-not-assigned


# %% [raw]
"""
.. _sec-dataset-advanced-examples:

Quantify dataset - advanced examples
====================================

.. seealso::

    The complete source code of this tutorial can be found in

    .. NB .py is from notebook_to_sphinx_extension

    :jupyter-download:notebook:`Quantify dataset - advanced examples.py`

    :jupyter-download:script:`Quantify dataset - advanced examples.py`
"""

# %% [raw]
"""

Here we will explore a few advanced usages of the quantify dataset and how it can
accommodate them.

.. admonition:: Imports and auxiliary utilities
    :class: dropdown
"""

# %%
rst_json_conf = {"indent": "    "}

import numpy as np
import xarray as xr
from quantify_core.data import handling as dh
from rich import pretty
from pathlib import Path
from quantify_core.data.handling import set_datadir
from quantify_core.utilities.inspect_utils import display_source_code
from quantify_core.utilities.examples_support import (
    mk_iq_shots,
    mk_dataset_attrs,
    mk_main_coord_attrs,
    mk_main_var_attrs,
    round_trip_dataset,
    mk_surface_7_sched,
)

pretty.install()

set_datadir(Path.home() / "quantify-data")  # change me!


# %% [raw]
"""
Dataset for an "unstructured" experiment
----------------------------------------

Let's take consider a Surface Code experiment, in particular the one portrayed in
Fig. 4b from :cite:`one of the papers from DiCarlo Lab <marques_logical_qubit_2021>`.

For simplicity, we will not use exactly the same schedule, because what matters here
are the measurements. It is difficult to deal with the results of these measurements
because we have a few repeating cycles followed by a final measurement that leaves the
overall dataset "unstructured".

.. figure:: /images/surface-7-sched.png
    :width: 100%

.. admonition:: Source code for generating this schedule and visualizing it
    :class: dropdown

    The schedule from the figure above was generates with
    :func:`quantify_core.utilities.examples_support.mk_surface_7_sched`.
"""

# %%
rst_json_conf = {"indent": "    "}
display_source_code(mk_surface_7_sched)

# %%
rst_json_conf = {"indent": "    "}
# If Quantify-Scheduler is installed you can create the schedule and visualize it
num_cycles = 3
try:
    import quantify_scheduler.visualization.circuit_diagram as qscd

    sched = mk_surface_7_sched(num_cycles=num_cycles)
    f, ax = qscd.circuit_diagram_matplotlib(sched)
    f.set_figwidth(30)
except ImportError:
    print("Quantify-Scheduler not installed.")

# %% [raw]
"""
How do we store all the shots for this measurement?
We might want this because, e.g., we know we have an issue with leakage to the second
excited state of a transmon and we would like to be able to store and inspect raw data.

To support such use-case we will have a dimension in dataset for the repeating cycles
and one for the final measurement.
"""

# %%
# mock data parameters
num_shots = 128  # NB usually >~1000 in real experiments
ground = -0.2 + 0.65j
excited = 0.7 + 4j
centroids = ground, excited
sigmas = [0.1] * 2

cycles = range(num_cycles)

mock_data = np.array(
    tuple(
        mk_iq_shots(
            n_shots=num_shots,
            sigmas=sigmas,
            centers=centroids,
            probabilities=[prob, 1 - prob],
        )
        for prob in [np.random.random() for _ in cycles]
    )
).T

mock_data_final = np.array(
    tuple(
        mk_iq_shots(
            n_shots=num_shots,
            sigmas=sigmas,
            centers=centroids,
            probabilities=[prob, 1 - prob],
        )
        for prob in [np.random.random()]
    )
).T
mock_data.shape, mock_data_final.shape

# %%
data_vars = {}

# NB same random data is used for all qubits only for the simplicity of the mock!
for q in (f"A{i}" for i in range(3)):
    data_vars[f"{q}_shots"] = (
        ("repetitions", "dim_cycle"),
        mock_data,
        mk_main_var_attrs(unit="V", long_name=f"IQ amplitude {q}", coords=["cycles"]),
    )

for q in (f"D{i}" for i in range(4)):
    data_vars[f"{q}_shots"] = (
        ("repetitions", "dim_final"),
        mock_data_final,
        mk_main_var_attrs(
            unit="V", long_name=f"IQ amplitude {q}", coords=["final_msmt"]
        ),
    )

cycle_attrs = mk_main_coord_attrs(long_name="Surface code cycle number")
final_msmt_attrs = mk_main_coord_attrs(long_name="Final measurement")
coords = dict(
    cycle=("dim_cycle", cycles, cycle_attrs),
    final_msmt=("dim_final", [0], final_msmt_attrs),
)

dataset = xr.Dataset(
    data_vars=data_vars,
    coords=coords,
    attrs=mk_dataset_attrs(repetitions_dims=["repetitions"]),
)

assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %%
dataset.A1_shots.shape, dataset.D1_shots.shape

# %%
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="dim_cycle", coords_names=["cycle"]
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded, dimension="dim_final", coords_names=["final_msmt"]
)
dataset_gridded

# %% [raw]
"""
.. _sec-nested-mc-example:

Dataset for a "nested MeasurementControl" experiment
----------------------------------------------------

Now consider a dataset that has been constructed by an experiment involving the
operation of two
:class:`~quantify_core.measurement.control.MeasurementControl` objects. The second of
them performs a "meta" outer loop in which we sweep a flux bias and then perform
several experiments to characterize a transmon qubit, e.g. determining the frequency of
a read-out resonator, the frequency of the transmon, and its T1 lifetime.

Since the raw data is now split among several datasets, we would like to keep a
reference to all these datasets. Below we showcase how this can be achieved, along with
some useful xarray features and known limitations.
"""

# %%
flux_bias_values = np.linspace(-0.04, 0.04, 12)

resonator_frequencies = np.linspace(7e9, 8.5e9, len(flux_bias_values))
qubit_frequencies = np.linspace(4.5e9, 4.6e9, len(flux_bias_values))
t1_values = np.linspace(20e-6, 50e-6, len(flux_bias_values))

resonator_freq_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]
qubit_freq_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]
t1_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]

# %%
coords = dict(
    flux_bias=(
        "main_dim",
        flux_bias_values,
        mk_main_coord_attrs(long_name="Flux bias", unit="A"),
    ),
    resonator_freq_tuids=(
        "main_dim",
        resonator_freq_tuids,
        mk_main_coord_attrs(long_name="Dataset TUID", is_dataset_ref=True),
    ),
    qubit_freq_tuids=(
        "main_dim",
        qubit_freq_tuids,
        mk_main_coord_attrs(long_name="Dataset TUID", is_dataset_ref=True),
    ),
    t1_tuids=(
        "main_dim",
        t1_tuids,
        mk_main_coord_attrs(long_name="Dataset TUID", is_dataset_ref=True),
    ),
)

# A tuple instead of a single str will indicate that these coordinates can used as
# a Multindex
vars_coords = tuple(coords.keys())

data_vars = dict(
    resonator_freq=(
        "main_dim",
        resonator_frequencies,
        mk_main_var_attrs(
            long_name="Resonator frequency", unit="Hz", coords=[vars_coords]
        ),
    ),
    qubit_freq=(
        "main_dim",
        qubit_frequencies,
        mk_main_var_attrs(long_name="Qubit frequency", unit="Hz", coords=[vars_coords]),
    ),
    t1=(
        "main_dim",
        t1_values,
        mk_main_var_attrs(long_name="T1", unit="s", coords=[vars_coords]),
    ),
)
dataset_attrs = mk_dataset_attrs()

dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset_attrs)

assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset

# %% [raw]
"""
In this case the four main coordinates are not orthogonal coordinates, but instead
just different label for the same data points, also known as a "multi-index". It is
possible to work with an explicit MultiIndex within a (python) xarray object:
"""

# %%
dataset_multi_indexed = dataset.set_index({"main_dim": vars_coords})

dataset_multi_indexed

# %% [raw]
"""
The MultiIndex is very handy for selecting data in different ways, e.g.:
"""

# %%
index = 2
dataset_multi_indexed.qubit_freq.sel(resonator_freq_tuids=resonator_freq_tuids[index])

# %%
dataset_multi_indexed.qubit_freq.sel(t1_tuids=t1_tuids[index])

# %% [raw]
"""
Known limitations
^^^^^^^^^^^^^^^^^
"""

# %% [raw]
"""
Unfortunately, at the moment the MultiIndex has the problem of not being compatible with
the NetCDF format used to write to disk:
"""

# %%
try:
    assert dataset_multi_indexed == round_trip_dataset(dataset_multi_indexed)
except NotImplementedError as exp:
    print(exp)

# %% [raw]
"""
We could make our load/write utilities to take care of setting and resetting the index
under the hood. Though there are some nuances there as well. If we would do that then
some extra metadata needs to be stored in order to store/restore the multi-index.
At the moment, the MultiIndex is not supported when writing a Quantify dataset to
disk. Below we show a few complications related to this.
"""

# %% [raw]
"""
Fortunately, the MultiIndex can be reset back:
"""

# %%
dataset_multi_indexed.reset_index(dims_or_levels="main_dim")

# %%
all(dataset_multi_indexed.reset_index("main_dim").t1_tuids == dataset.t1_tuids)

# %% [raw]
"""
But, for example, the ``dtype`` has been changed to ``object``
(from fixed-length string):
"""

# %%
dataset.t1_tuids.dtype, dataset_multi_indexed.reset_index("main_dim").t1_tuids.dtype

# %%
dataset.t1_tuids.dtype == dataset_multi_indexed.reset_index("main_dim").t1_tuids.dtype
