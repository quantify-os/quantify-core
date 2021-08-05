Quantify dataset specification
==============================

.. note::
    
    Along this page we show exemplary datasets to highlight the details of this specification.
    However, note that we attempt to always show a valid dataset with all the required properties (except when exemplifying a bad dataset).


.. admonition:: imports and auxiliary utilities
    :class: dropdown


    .. jupyter-execute::
        :hide-code:
        :hide-output:

        import numpy as np
        import xarray as xr
        import matplotlib.pyplot as plt
        from quantify_core.data import handling as dh
        from quantify_core.measurement import grid_setpoints
        from qcodes import ManualParameter

        def assign_dataset_attrs(ds: xr.Dataset) -> dict:
            tuid = dh.gen_tuid()
            ds.attrs.update({
                "grid_2d": True,  # necessary for live plotting
                "grid_2d_uniformly_spaced": True,  # pyqt requires interpolation
                "tuid": tuid,
                "quantify_dataset_version": "v1.0",
            })
            return ds.attrs

        def dataset_round_trip(ds: xr.Dataset) -> xr.Dataset:
            assign_dataset_attrs(ds)
            tuid = ds.attrs["tuid"]
            dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, ds)
            return dh.load_dataset(tuid)

        def par_to_attrs(par):
            return {"units": par.unit, "long_name": par.label, "standard_name": par.name}

        from pathlib import Path
        from quantify_core.data.handling import get_datadir, set_datadir

        set_datadir(Path.home() / "quantify-data")  # change me!


.. jupyter-execute::

    # some auxiliary constant used to build exemplary datasets
    l_t, l_a = 4, 3
    num_setpoints = l_t * l_a

    x0s = np.linspace(1, 4, l_t)
    x1s = np.linspace(-1, 0, l_a)

    x1s = np.repeat(x1s, l_t)
    x0s = np.tile(x0s, l_a)

    assert len(x0s) == len(x1s) == num_setpoints


Introduction
------------


Xarray overview
~~~~~~~~~~~~~~~


This is a brief overview of some concepts and functionalities of ``xarray`` that are leveraged to define the Quantify dataset.

The dataset has **Dimensions** and **Variables**. Variables "lie" along at least one dimension:


.. jupyter-execute::

    n = 5
    name_dim_a = "position_x"
    name_dim_b = "velocity_x"
    dataset = xr.Dataset(
        data_vars={
            "position": (name_dim_a, np.linspace(-5, 5, n), {"units": "m"}),
            "velocity": (name_dim_b, np.linspace(0, 10, n), {"units": "m/s"}),
        },
        attrs={"key": "my metadata"}
    )
    dataset


A variable can be set as coordinate for its dimension(s):


.. jupyter-execute::

    position = np.linspace(-5, 5, n)
    dataset = xr.Dataset(
        data_vars={
            "position": (name_dim_a, position, {"units": "m"}),
            "velocity": (name_dim_a, 1 + position ** 2 , {"units": "m/s"})
        },
        attrs={"key": "my metadata"}
    )
    dataset = dataset.set_coords(["position"])
    dataset


Xarray coordinates can be set to **index** other variables. (:func:`~quantify_core.data.handling.to_gridded_dataset` does this under the hood.)


.. jupyter-execute::

    dataset = dataset.set_index({"position_x": "position"})
    dataset.position_x.attrs["units"] = "m"
    dataset


An example of how this can be usefull:


.. jupyter-execute::

    dataset.velocity.sel(position_x=2.5)


Automatic plotting:


.. jupyter-execute::

    dataset.velocity.plot();


Key dataset conventions
~~~~~~~~~~~~~~~~~~~~~~~


We define the following naming conventions in the Quantify dataset:

- **Experiment coordinate(s)**
    - ``xarray`` **Coordinates** following the naming convention ``f"x{i}"`` with ``i >= 0`` a integer.
    - Often correspond to physical coordinates, e.g., a signal frequency or amplitude.
- **Exeperiment variable(s)**
    - ``xarray`` **Variables** following the naming convention ``f"y{i}"`` with ``i >= 0`` a integer.
    - Often correspond to a physical quantity being measured, e.g., the signal magnitude at a specific frequency measured on a metal contact of a quantum chip.


2D Dataset example
~~~~~~~~~~~~~~~~~~

.. admonition:: Generate data
    :class: dropdown


    .. jupyter-execute::

        x0s = np.linspace(0.45, 0.55, 30)
        x1s = np.linspace(0, 100e-9, 40)
        time_par = ManualParameter(name="time", label="Time", unit="s")
        amp_par =  ManualParameter(name="amp", label="Flux amplitude", unit="V")
        pop_q0_par = ManualParameter(name="pop_q0", label="Population Q0", unit="arb. un.")
        pop_q1_par = ManualParameter(name="pop_q1", label="Population Q1", unit="arb. un.")

        x0s, x1s = grid_setpoints(
            [x0s, x1s],
            [amp_par, time_par]
        ).T
        x0s_norm = np.abs((x0s - x0s.mean()) / (x0s - x0s.mean()).max())
        y0s = (1 - x0s_norm) * np.sin(2 * np.pi * x1s * 1/30e-9 * (x0s_norm + 0.5)) # ~chevron
        y1s = - y0s + 0.1

        dataset = xr.Dataset(
            data_vars={
                "y0": (("repetition", "acq_set_0"), [y0s], par_to_attrs(pop_q0_par)),
                "y1": (("repetition", "acq_set_0"), [y1s], par_to_attrs(pop_q1_par)),
            },
            coords={
                "x0": ("acq_set_0", x0s, par_to_attrs(amp_par)),
                "x1": ("acq_set_0", x1s, par_to_attrs(time_par)),
            }
        )

        assert dataset == dataset_round_trip(dataset)  # confirm read/write


On the dataset below we have two experiment coordinates ``x0`` and ``x1``; and two experiment variables ``y0`` and ``y0``. Both experiment coordinates lie along one dimension, ``acq_set_0``. Both experiment variables lie along two dimensions ``acq_set_0`` and ``repetitions``.


.. jupyter-execute::

    dataset


As seen above, in the Quantify dataset the experiment coordinates do not index the experiment variables because not all use cases fit within this paradigm. However, when possible the dataset can be converted to take advange of the ``xarray`` built-in utlities.


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(dataset, dimension="acq_set_0")
    dataset_gridded.y0.plot(x="x0"); plt.show();
    dataset_gridded.y1.plot(x="x0"); plt.show();


Detailed specification
----------------------


Xarray dimensions
~~~~~~~~~~~~~~~~~


Xarray coordinates (variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Xarray variables
~~~~~~~~~~~~~~~~


Calibration points
~~~~~~~~~~~~~~~~~~


Dataset attributes
~~~~~~~~~~~~~~~~~~


### 2D Dataset with explicit single repetition


dataset = xr.Dataset(
    data_vars={
        "y0": (("repetition", "acq_set_0"), [np.linspace(1, 4, num_setpoints)]),
        "y1": (("repetition", "acq_set_0"), [np.linspace(-4, 5, num_setpoints) + 0.2]),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded


dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded


# Plotting still works even though there is an extra dimension
dataset_gridded.y0.plot()
plt.show()
dataset_gridded.y1.plot()
plt.show()


### 2D Dataset with multiple repetitions


rep_num = 5
dataset = xr.Dataset(
    data_vars={
        "y0": (
            ("repetition", "acq_set_0"),
            [np.linspace(1, 4, num_setpoints) + i for i in range(rep_num)],
        ),
        "y1": (
            ("repetition", "acq_set_0"),
            [np.linspace(-4, 5, num_setpoints) + 2 * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded


dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded


dataset_gridded.y0.plot(x="x0", y="x1", col="repetition")
dataset_gridded.y1.plot(x="x0", y="x1", col="repetition")


### 2D Dataset with repetitions and (complex) time "traces"


rep_num = 4
time = np.arange(0, 400e-9, 10e-9)
# cos = np.cos(2 * np.pi * 3e6 * time)
# plt.plot(time, cos, ".-")

# NB: just some "random" frequency and amplitude change, not really dependent on x0 and x1
traces = np.array(
    [
        (1.2 - f / 7e6) * np.exp(-2j * np.pi * f * time)
        for f in np.linspace(3e6, 7e6, num_setpoints)
    ]
)

dataset = xr.Dataset(
    data_vars={
        "y0_tseries": (
            ("repetition", "acq_set_0", "time"),
            [traces + (0.4 - 0.8j) * i for i in range(rep_num)],
        ),
        "y1_tseries": (
            ("repetition", "acq_set_0", "time"),
            [traces + (-0.6 + 0.9j) * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
        # NB there is a dimension named `time` and also a coordinate with the same name!
        # NB2 xarray automatically understands that we want to index the `time` dimension
        # using the `time` coordinate (it will appear in bold below).
        "time": ("time", time, {"unit": "s"}),
    },
)
dataset = dataset.set_coords(["x0", "x1"])
# dataset = dataset.set_index({"time": "time"})

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded


dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded


darray = dataset_gridded.y0_tseries
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y0_tseries
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)

darray = dataset_gridded.y1_tseries
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y1_tseries
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)


### `acq_channel`\s with different datashapes per measured "data point"
This would be case when we want to save e.g. the qubit population **AND** the time traces.


rep_num = 4
time = np.arange(0, 400e-9, 10e-9)
# cos = np.cos(2 * np.pi * 3e6 * time)
# plt.plot(time, cos, ".-")

# NB: just some "random" frequency and amplitude change, not really dependent on x0 and x1
traces = np.array(
    [
        (1.2 - f / 7e6) * np.exp(-2j * np.pi * f * time)
        for f in np.linspace(3e6, 7e6, num_setpoints)
    ]
)

dataset = xr.Dataset(
    data_vars={
        "y0": (
            ("repetition", "acq_set_0"),
            [np.linspace(1, 4, num_setpoints) + i for i in range(rep_num)],
        ),
        "y0_time": (
            ("repetition", "acq_set_0", "time"),
            [traces + (0.4 - 0.8j) * i for i in range(rep_num)],
        ),
        "y1_time": (
            ("repetition", "acq_set_0", "time"),
            [traces + (-0.6 + 0.9j) * i for i in range(rep_num)],
        ),
        "y1": (
            ("repetition", "acq_set_0"),
            [np.linspace(-4, 5, num_setpoints) + 2 * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
        "time": ("time", time, {"unit": "s"}),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded


dataset.y0.shape, dataset.y1.shape, dataset.y0.dtype, dataset.y1.dtype


dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded


dataset_gridded.y0.plot(x="x0", y="x1", col="repetition")
dataset_gridded.y1.plot(x="x0", y="x1", col="repetition")

darray = dataset_gridded.y0_time
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y0_time
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)

darray = dataset_gridded.y1_time
darray.real.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} real part", y=1.03)

darray = dataset_gridded.y1_time
darray.imag.plot(x="time", hue="repetition", col="x0", row="x1", marker=".")
plt.gcf().suptitle(f"Trace {darray.name} imaginary part", y=1.03)


### 2D dataset with calibration points

One possibility is to use a dedicated variable(s) with an "independent" dedicated `xarray` dimension and a naming convention.


rep_num = 5
dataset = xr.Dataset(
    data_vars={
        "y0": (
            ("repetition", "acq_set_0"),
            [np.linspace(1, 4, num_setpoints) + i for i in range(rep_num)],
        ),
        "y0_calib": (
            ("repetition", "acq_set_0_calib"),
            [[1, 4] for i in range(rep_num)],
        ),
        "y1": (
            ("repetition", "acq_set_0"),
            [np.linspace(-4, 5, num_setpoints) + 2 * i for i in range(rep_num)],
        ),
        "x0": ("acq_set_0", x0s),
        "x1": ("acq_set_0", x1s),
    },
)
dataset = dataset.set_coords(["x0", "x1"])

dataset_loaded = dataset_round_trip(dataset)
assert dataset == dataset_loaded  # confirm read/write
dataset_loaded


dataset_gridded = dh.to_gridded_dataset(
    dataset, dimension="acq_set_0"
)  # seems to work out of the box
dataset_gridded








# from importlib import reload
# from quantify_core.utilities import _docs_helpers

# reload(_docs_helpers)

# file_name = "Quantify dataset specification"
# _docs_helpers.notebook_to_rst(f"{file_name}.ipynb", f"{file_name}.rst")
