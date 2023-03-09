---
file_format: mystnb
kernelspec:
  name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---

(sec-dataset-advanced-examples)=
# Quantify dataset - advanced examples

```{seealso}
The complete source code of this tutorial can be found in

{nb-download}`Quantify dataset - advanced examples.ipynb`
```

Here we will explore a few advanced usages of the Quantify dataset and how it can
accommodate them.

```{code-cell} ipython3
---
tags: ['hide-cell']
mystnb:
  code_prompt_show: Imports and auxiliary utilities
---
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from rich import pretty

from quantify_core.analysis.calibration import rotate_to_calibrated_axis
from quantify_core.analysis.fitting_models import exp_decay_func
from quantify_core.data import handling as dh
from quantify_core.utilities import dataset_examples
from quantify_core.utilities.dataset_examples import (
    mk_nested_mc_dataset,
    mk_shots_from_probabilities,
    mk_surface7_cyles_dataset,
)
from quantify_core.utilities.examples_support import (
    mk_iq_shots,
    round_trip_dataset,
)
from quantify_core.utilities.inspect_utils import display_source_code

pretty.install()

dh.set_datadir(Path.home() / "quantify-data")  # change me!
```

## Dataset for an "unstructured" experiment

Let's take consider a Surface Code experiment, in particular the one portrayed in
Fig. 4b from one of the papers from DiCarlo Lab {cite}`marques_logical_qubit_2021`.

For simplicity, we will not use exactly the same schedule, because what matters here
are the measurements. It is difficult to deal with the results of these measurements
because we have a few repeating cycles followed by a final measurement that leaves the
overall dataset "unstructured".

```{figure} /images/surface-7-sched.png
:width: 100%
```

``````{admonition} Source code for generating this schedule and visualizing it
:class: dropdown, info

If you want to create and visualize the schedule above using `quantify-scheduler`,
you can use this code:

```{code-block} python
from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import CZ, Y90, Measure, Reset, X
from quantify_scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib

def mk_surface7_sched(num_cycles: int = 3):
    """Generates a schedule with some of the feature of a Surface 7 experiment as
    portrayed in Fig. 4b of :cite:`marques_logical_qubit_2021`.

    Parameters
    ----------
    num_cycles
        The number of times to repeat the main cycle.

    Returns
    -------
    :
        A schedule similar to a Surface 7 dance.
    """

    sched = Schedule("S7 dance")

    q_d1, q_d2, q_d3, q_d4 = [f"D{i}" for i in range(1, 5)]
    q_a1, q_a2, q_a3 = [f"A{i}" for i in range(1, 4)]
    all_qubits = q_d1, q_d2, q_d3, q_d4, q_a1, q_a2, q_a3

    sched.add(Reset(*all_qubits))

    for cycle in range(num_cycles):
        sched.add(Y90(q_d1))
        for qubit in [q_d2, q_d3, q_d4]:
            sched.add(Y90(qubit), ref_pt="start", rel_time=0)
        sched.add(Y90(q_a2), ref_pt="start", rel_time=0)

        for qubit in [q_d2, q_d1, q_d4, q_d3]:
            sched.add(CZ(qC=qubit, qT=q_a2))

        sched.add(Y90(q_d1))
        for qubit in [q_d2, q_d3, q_d4]:
            sched.add(Y90(qubit), ref_pt="start", rel_time=0)
        sched.add(Y90(q_a2), ref_pt="start", rel_time=0)

        sched.add(Y90(q_a1), ref_pt="end", rel_time=0)
        sched.add(Y90(q_a3), ref_pt="start", rel_time=0)

        sched.add(CZ(qC=q_d1, qT=q_a1))
        sched.add(CZ(qC=q_d2, qT=q_a3))
        sched.add(CZ(qC=q_d3, qT=q_a1))
        sched.add(CZ(qC=q_d4, qT=q_a3))

        sched.add(Y90(q_a1), ref_pt="end", rel_time=0)
        sched.add(Y90(q_a3), ref_pt="start", rel_time=0)

        sched.add(Measure(q_a2, acq_index=cycle))
        for qubit in (q_a1, q_a3):
            sched.add(Measure(qubit, acq_index=cycle), ref_pt="start", rel_time=0)

        for qubit in [q_d1, q_d2, q_d3, q_d4]:
            sched.add(X(qubit), ref_pt="start", rel_time=0)

    # final measurements

    sched.add(Measure(*all_qubits[:4], acq_index=0), ref_pt="end", rel_time=0)

    return sched

sched = mk_surface7_sched(num_cycles=3)
f, ax = circuit_diagram_matplotlib(sched)
f.set_figwidth(30)
```
``````

How do we store all the shots for this measurement?
We might want this because, e.g., we know we have an issue with leakage to the second
excited state of a transmon and we would like to be able to store and inspect raw data.

To support such use-cases we will have a dimension in the dataset for the repeating cycles
and one extra dimension for the final measurement.

```{code-cell} ipython3
# mock data parameters
num_shots = 128  # NB usually >~1000 in real experiments
ground = -0.2 + 0.65j
excited = 0.7 + 4j
centroids = ground, excited
sigmas = [0.1] * 2

display_source_code(mk_iq_shots)
display_source_code(mk_shots_from_probabilities)
display_source_code(mk_surface7_cyles_dataset)
```

```{code-cell} ipython3
dataset = mk_surface7_cyles_dataset(
    num_shots=num_shots, sigmas=sigmas, centers=centroids
)

assert dataset == round_trip_dataset(dataset)  # confirm read/write

dataset
```

```{code-cell} ipython3
dataset.A1_shots.shape, dataset.D1_shots.shape
```

```{code-cell} ipython3
dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="dim_cycle", coords_names=["cycle"]
)
dataset_gridded = dh.to_gridded_dataset(
    dataset_gridded, dimension="dim_final", coords_names=["final_msmt"]
)
dataset_gridded
```

```{code-cell} ipython3
dataset_gridded.A0_shots.real.mean("repetitions").plot(marker="o", label="I-quadrature")
dataset_gridded.A0_shots.imag.mean("repetitions").plot(marker="^", label="Q-quadrature")
_ = plt.gca().legend()
```

(sec-nested-mc-example)=
## Dataset for a "nested MeasurementControl" experiment

Now consider a dataset that has been constructed by an experiment involving the
operation of two
{class}`.MeasurementControl` objects. The second of
them performs a "meta" outer loop in which we sweep a flux bias and then perform
several experiments to characterize a transmon qubit, e.g. determining the frequency of
a read-out resonator, the frequency of the transmon, and its T1 lifetime.

Below we showcase what the data from the dataset containing the T1 experiment results
could look like

```{code-cell} ipython3
fig, ax = plt.subplots()
rng = np.random.default_rng(seed=112244)  # random number generator

num_t1_datasets = 7
t1_times = np.linspace(0, 120e-6, 30)

for tau in rng.uniform(10e-6, 50e-6, num_t1_datasets):
    probabilities = exp_decay_func(
        t=t1_times, tau=tau, offset=0, n_factor=1, amplitude=1
    )
    dataset = dataset_examples.mk_t1_av_with_cal_dataset(t1_times, probabilities)

    round_trip_dataset(dataset)  # confirm read/write
    dataset_g = dh.to_gridded_dataset(
        dataset, dimension="main_dim", coords_names=["t1_time"]
    )
    # rotate the iq data
    rotated_and_normalized = rotate_to_calibrated_axis(
        dataset_g.q0_iq_av.values, *dataset_g.q0_iq_av_cal.values
    )
    rotated_and_normalized_da = xr.DataArray(dataset_g.q0_iq_av)
    rotated_and_normalized_da.values = rotated_and_normalized
    rotated_and_normalized_da.attrs["long_name"] = "|1> Population"
    rotated_and_normalized_da.attrs["units"] = ""
    rotated_and_normalized_da.real.plot(ax=ax, label=dataset.tuid, marker=".")
ax.set_title("Results from repeated T1 experiments\n(different datasets)")
_ = ax.legend()
```

Since the raw data is now split among several datasets, we would like to keep a
reference to all these datasets in our "combined" datasets. Below we showcase how this
can be achieved, along with some useful xarray features and known limitations.

We start by generating a mock dataset that combines all the information that would have
been obtained from analyzing a series of other datasets.

```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
  code_prompt_show: Source code for mk_nested_mc_dataset function
---
display_source_code(mk_nested_mc_dataset)
```

```{code-cell} ipython3
dataset = mk_nested_mc_dataset(num_points=num_t1_datasets)
assert dataset == round_trip_dataset(dataset)  # confirm read/write
dataset
```

In this case the four main coordinates are not orthogonal coordinates, but instead
just different labels for the same data points, also known as a "multi-index".

```{code-cell} ipython3
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

_ = dataset.t1.plot(x="flux_bias", marker="o", ax=axs[0].twiny(), color="C0")
x = "t1_tuids"
_ = dataset.t1.plot(x=x, marker="o", ax=axs[0], color="C0")
_ = dataset.resonator_freq.plot(x=x, marker="o", ax=axs[1], color="C1")
_ = dataset.qubit_freq.plot(x=x, marker="o", ax=axs[2], color="C2")
for tick in axs[2].get_xticklabels():
    tick.set_rotation(15)  # avoid tuid labels overlapping
```

It is possible to work with an explicit MultiIndex within a (python) xarray object:

```{code-cell} ipython3
dataset_multi_indexed = dataset.set_index({"main_dim": tuple(dataset.t1.coords.keys())})
dataset_multi_indexed
```

The MultiIndex is very handy for selecting data in different ways, e.g.:

```{code-cell} ipython3
index = 2
dataset_multi_indexed.qubit_freq.sel(
    qubit_freq_tuids=dataset_multi_indexed.qubit_freq_tuids.values[index]
)
```

```{code-cell} ipython3
dataset_multi_indexed.qubit_freq.sel(t1_tuids=dataset.t1_tuids.values[index])
```

### Known limitations

Unfortunately, at the moment the MultiIndex has the problem of not being compatible with
the NetCDF format used to write to disk:

```{code-cell} ipython3
try:
    assert dataset_multi_indexed == round_trip_dataset(dataset_multi_indexed)
except NotImplementedError as exp:
    print(exp)
```

We could make our load/write utilities take care of setting and resetting the index
under the hood. Though there are some nuances there as well. If we would do that then
some extra metadata needs to be stored in order to store/restore the multi-index.
At the moment, the MultiIndex is not supported when writing a Quantify dataset to
disk. Below we show a few complications related to this.

Fortunately, the MultiIndex can be reset back:

```{code-cell} ipython3
dataset_multi_indexed.reset_index(dims_or_levels="main_dim")
```

```{code-cell} ipython3
all(dataset_multi_indexed.reset_index("main_dim").t1_tuids == dataset.t1_tuids)
```

But, for example, the `dtype` has been changed to `object`
(from fixed-length string):

```{code-cell} ipython3
dataset.t1_tuids.dtype, dataset_multi_indexed.reset_index("main_dim").t1_tuids.dtype
```

```{code-cell} ipython3
dataset.t1_tuids.dtype == dataset_multi_indexed.reset_index("main_dim").t1_tuids.dtype
```
