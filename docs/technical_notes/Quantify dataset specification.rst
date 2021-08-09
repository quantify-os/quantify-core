Quantify dataset specification
==============================


.. admonition:: Development notes
    :class: warning

    We do not know yet if ``acq_set_{j}`` with ``j>0`` will be part of this specification (we lack some clear examples).


.. note::

    Along this page we show exemplary datasets to highlight the details of this specification.
    However, keep in mind that we always show a valid dataset with all the required properties (except when exemplifying a bad dataset).

.. admonition:: Imports and auxiliary utilities
    :class: dropdown


    .. jupyter-execute::
        :hide-output:

        import numpy as np
        import xarray as xr
        import matplotlib.pyplot as plt
        from quantify_core.data import handling as dh
        from quantify_core.measurement import grid_setpoints
        from qcodes import ManualParameter
        from rich import pretty

        pretty.install()


        def assign_dataset_attrs(ds: xr.Dataset) -> dict:
            tuid = dh.gen_tuid()
            ds.attrs.update(
                {
                    "grid": True,
                    "grid_uniformly_spaced": True,  # pyqt requires interpolation
                    "tuid": tuid,
                    "quantify_dataset_version": "v1.0",
                }
            )
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


Introduction
------------


Xarray overview
~~~~~~~~~~~~~~~


This is a brief overview of some concepts and functionalities of xarray that are leveraged to define the Quantify dataset.

The dataset has **Dimensions** and **Variables**. Variables "lie" along at least one dimension:


.. jupyter-execute::

    n = 5
    name_dim_a = "position_x"
    name_dim_b = "velocity_x"
    dataset = xr.Dataset(
        data_vars={
            "position": (name_dim_a, np.linspace(-5, 5, n), {"units": "m", "long_name": "Position"}),
            "velocity": (name_dim_b, np.linspace(0, 10, n), {"units": "m/s", "long_name": "Velocity"}),
        },
        attrs={"key": "my metadata"},
    )
    dataset


A variable can be set as coordinate for its dimension(s):


.. jupyter-execute::

    position = np.linspace(-5, 5, n)
    dataset = xr.Dataset(
        data_vars={
            "position": (name_dim_a, position, {"units": "m", "long_name": "Position"}),
            "velocity": (name_dim_a, 1 + position ** 2, {"units": "m/s", "long_name": "Velocity"}),
        },
        attrs={"key": "my metadata"},
    )
    dataset = dataset.set_coords(["position"])
    dataset


Xarray coordinates can be set to **index** other variables. (:func:`~quantify_core.data.handling.to_gridded_dataset` does this under the hood.)


.. jupyter-execute::

    dataset = dataset.set_index({"position_x": "position"})
    dataset.position_x.attrs["units"] = "m"
    dataset.position_x.attrs["long_name"] = "Position x"
    dataset


An example of how this can be usefull:


.. jupyter-execute::

    dataset.velocity.sel(position_x=2.5)


Automatic plotting:


.. jupyter-execute::

    dataset.velocity.plot();


.. _sec-experiment-coordinates-and-variables:

Key dataset conventions
~~~~~~~~~~~~~~~~~~~~~~~


We define the following naming conventions in the Quantify dataset:

- **Experiment coordinate(s)**
    - xarray **Coordinates** following the naming convention ``f"x{i}"`` with ``i >= 0`` an integer.
    - Often correspond to physical coordinates, e.g., a signal frequency or amplitude.
- **Experiment variable(s)**
    - xarray **Variables** following the naming convention ``f"y{i}"`` with ``i >= 0`` an integer.
    - Often correspond to a physical quantity being measured, e.g., the signal magnitude at a specific frequency measured on a metal contact of a quantum chip.



2D Dataset example
~~~~~~~~~~~~~~~~~~


In the dataset below we have two experiment coordinates ``x0`` and ``x1``; and two experiment variables ``y0`` and ``y0``. Both experiment coordinates lie along one dimension, ``acq_set_0``. Both experiment variables lie along two dimensions ``acq_set_0`` and ``repetitions``.


.. admonition:: Generate data
    :class: dropdown


    .. jupyter-execute::

        x0s = np.linspace(0.45, 0.55, 30)
        x1s = np.linspace(0, 100e-9, 40)
        time_par = ManualParameter(name="time", label="Time", unit="s")
        amp_par = ManualParameter(name="amp", label="Flux amplitude", unit="V")
        pop_q0_par = ManualParameter(name="pop_q0", label="Population Q0", unit="arb. un.")
        pop_q1_par = ManualParameter(name="pop_q1", label="Population Q1", unit="arb. un.")

        x0s, x1s = grid_setpoints([x0s, x1s], [amp_par, time_par]).T
        x0s_norm = np.abs((x0s - x0s.mean()) / (x0s - x0s.mean()).max())
        y0s = (1 - x0s_norm) * np.sin(
            2 * np.pi * x1s * 1 / 30e-9 * (x0s_norm + 0.5)
        )  # ~chevron
        y1s = -y0s + 0.1

        dataset = dataset_2d_example = xr.Dataset(
            data_vars={
                "y0": (("repetition", "acq_set_0"), [y0s], par_to_attrs(pop_q0_par)),
                "y1": (("repetition", "acq_set_0"), [y1s], par_to_attrs(pop_q1_par)),
            },
            coords={
                "x0": ("acq_set_0", x0s, par_to_attrs(amp_par)),
                "x1": ("acq_set_0", x1s, par_to_attrs(time_par)),
            },
        )

        assert dataset == dataset_round_trip(dataset)  # confirm read/write


.. jupyter-execute::

    dataset


As seen above, in the Quantify dataset the experiment coordinates do not index the experiment variables because not all use cases fit within this paradigm. However, when possible the dataset can be converted to take advantage of the xarray built-in utilities.


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(dataset, dimension="acq_set_0")
    dataset_gridded.y0.plot(x="x0")
    plt.show()
    dataset_gridded.y1.plot(x="x0")
    plt.show()


Detailed specification
----------------------


Xarray dimensions
~~~~~~~~~~~~~~~~~


The Quantify dataset has has the following required and optional dimensions:

- **[Required]** ``repetition``

    - The outermost dimension of the :ref:`experiment variables <sec-experiment-coordinates-and-variables>`.
    - Intuition for this xarray dimension: the equivalent would be to have ``dataset_reptition_0.hdf5``, ``dataset_reptition_1.hdf5``, etc. where each dataset was obtained from repeating exactly the same experiment. Instead we define an outer dimension for this.
    - Default behavior of plotting tools will be to average the dataset along this dimension.
    - The :ref:`experiment variables <sec-experiment-coordinates-and-variables>` must lie along this dimension (even when only one repetition of the experiment was executed).
    - **[Optional]** The ``repetition`` dimension can be indexed by an optional xarray coordinate variable.

        - **[Required]** The variable must be named ``repetition`` as well.

    - **[Required]** no other outer xarray dimensions allowed.




.. admonition:: Examples good datasets (repetition)
    :class: dropdown

    To be added:

    - More than one repetitions.
    - ``repetition`` dimensions indexed by a ``coordinate`` variables.


.. admonition:: Examples bad datasets (repetition)
    :class: dropdown

     To be added:

    - No repetition dimension.
    - An outer dimension.


- **[Required]** ``acq_set_0``

    - The outermost dimension of the :ref:`experiment coordinates <sec-experiment-coordinates-and-variables>`.
    - The first inner dimension of the :ref:`experiment variables <sec-experiment-coordinates-and-variables>` (the outermost is the ``repetition`` dimension).



.. admonition:: Examples good datasets (acq_set_0)
    :class: dropdown


    .. jupyter-execute::

        dataset_2d_example


.. admonition:: Examples bad datasets (acq_set_0)
    :class: dropdown

    To be added:

    - `x0` and `y0` with some other dimension then ``acq_set_0``.



- **[Optional, Advanced]** other nested xarray dimensions under each ``acq_set_{i}``

    - Intuition: intended primarily for time series, also known as "time trace" or simply trace.
    - Other, potentially arbitrarily nested, xarray dimensions under each ``acq_set_{i}`` is allowed. I.e., **each entry** in a, e.g., ``y3`` xarray variable can be a 1D, or nD array where each "D" has a corresponding xarray dimension.
    - Such xarray dimensions can be named arbitrarily.
    - Each of such xarray dimension can be *indexed* by an xarray coordinate variable. E.g. for a time trace we would have in the dataset:

        - ``assert "time" in dataset.coords``
        - ``assert "time" in dataset.dims``
        - ``assert len(dataset.time) == len(dataset.y3.isel(repetition=0, acq_set_0=0))`` where ``y3`` is a measured variable storing traces.

    - Note: When nesting data like this, it is required to have "hyper-cubic"-shaped data, meaning that e.g. ``dataset.y3.isel(repetition=0, acq_set_0=0) == [[2], [ 5, 6]]`` is not possible, but ``dataset.y3.isel(repetition=0, acq_set_0=0) == [[2, 3], [5, 6]]`` is. This is a direct consequence of numpy ``ndarray`` (with entries of type ``int``/``float``/``complex``).



.. admonition:: Examples good datasets (other nested dimensions)
    :class: dropdown

    To be added:

    - time series example
    - time series example with complex data
    - (fictitious examples) time series with a few distinct DACs, where the DACs names index an extra dimension.



.. admonition:: Examples bad datasets (other nested dimensions)
    :class: dropdown

    To be added:

    - ``time`` coordinate is not indexing the ``time`` dimension.



.. admonition:: To be refined (acq_set_{i})
    :class: dropdown, warning

    For reference from earlier discussion, requires some good example to justify this:

    - **[Optional, Advanced]** ``acq_set_{i}``, where ``i`` > 0 is an integer.

    - Reserves the possibility to store data for experiments that we have not yet encountered ourselves. I a gut feeling that we need this, but might not have a good realistic example, some help here is welcome.

        - (Example ?) Imagine measuring some qubits until all of them are in a desired state, returning the data of these measurements and then proceeding to doing the "real" experiment you are interested in. I think having these extra *independent* xarray dimensions
    - **[Required]** all ``acq_set_{i}`` dimensions (including ``acq_set_0``) are mutually excluding. This means variables in the dataset cannot depend on more than one of these dimensions.

        - **Bad** variable: ``y0(repetition, acq_set_0, acq_set_1)``, this should never happen in the dataset.
        - **Good** variable: ``y0(repetition, acq_set_0)`` or ``y1(repetition, acq_set_1)``.



Xarray coordinates (variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Only the following `xarray` coordinates are allowed in the dataset:

- **[Required]** The ``x0`` :ref:`experiment coordinate <sec-experiment-coordinates-and-variables>`.
    - Usually equivalent to a settable, usually a parameter that an experimentalist "sweeps" in order to observe the effect on some other property of the system being studied.
    - For some experiments it might not be suitable to think of a parameter that is being varied. In such cases ``x0`` can be simply an array of integers, e.g. ``np.linspace(0, number_of_points)``.
- **[Optional]** Other ``f"x{i}"`` :ref:`experiment coordinates <sec-experiment-coordinates-and-variables>`, with ``i`` a positive integer.

    - These are the coordinates that index the :ref:`experiment variables <sec-experiment-coordinates-and-variables>`. This indexing can be made explicit in a (separate) :class:`xarray.Dataset` instance returned by :func:`quantify_core.data.handling.to_gridded_dataset()` (when the data corresponds to a multi-dimensional grid).
    - **[Required]** Each ``x{i}`` must lie along one (and only one) ``acq_set_{j}`` xarray dimension.
- **[Optional]** Other xarray coordinates (that are not :ref:`experiment coordinates <sec-experiment-coordinates-and-variables>`) used to index the nested dimensions.

    - Allowed dimension names:
        - ``repetition``, or
        - ``acq_set_{i}``, or
        - ``<arbitrary_name>`` but with the same name as one of the **nested** dimensions (see :ref:`Xarray dimensions` section above).
    - **[Required]** These other xarray coordinates must "lie" along a single dimension (and have the same name).



.. admonition:: Examples good datasets (coordinates)
    :class: dropdown

    To be added...


Xarray data variables
~~~~~~~~~~~~~~~~~~~~~


The only xarray data variables allowed in the dataset are the :ref:`experiment variables <sec-experiment-coordinates-and-variables>`. Each entry in one of these experiment variables is a data-point in the broad sense, i.e. it can be ``int``/``float``/``complex`` **OR** a nested ``numpy.ndarray`` (of one of these ``dtypes``).

All the xarray data variables in the dataset (that are not xarray coordinates) comply with:

- Naming:
    - ``y{i}`` where  is an integer; **OR**
    - ``y{i}_<arbitrary>`` where ``i => 0`` is an integer such that matches an existing ``y{i}`` in the same dataset.
        - This is intended to denote a meaningful connection between ``y{i}`` and ``y{i}_<arbitrary>``.
        - **[Required]** The number of elements in``y{i}`` and ``y{i}_<arbitrary>`` must be the same along the ``acq_set_{j}`` dimension.
        - E.g., the digitized time traces stored in ``y0_trace(repetition, acq_set_0, time)`` and the demodulated values ``y0(repetition, acq_set_0)`` represent the same measurement with different levels of detail.
    - Rationale: facilitates inspecting and processing the dataset in an intuitive way.
- **[Required]** Lie along at least the ``repetition`` and ``acq_set_{i}`` dimensions.
- **[Optional]** Lie along additional nested xarray dimensions.



.. admonition:: Examples good datasets (variables)
    :class: dropdown

    To be added...

    - ``y0_trace(repetition, acq_set_0, time)`` and the demodulated values ``y0(repetition, acq_set_0)``



Dataset with two ``y{i}``:


    .. jupyter-execute::

        dataset_2d_example


Dataset attributes
~~~~~~~~~~~~~~~~~~


The dataset must have the following attributes:

- ``grid`` (``bool``)
    - Specifies if the experiment coordinates are the "unrolled" points (also known as "unstacked") corresponding to a grid. If ``True`` than it is possible to use :func:`quantify_core.data.handling.to_gridded_dataset()` to convert the dataset.
- ``grid_uniformly_spaced`` (``bool``)
    - Can be ``True`` only if ``grid`` is also ``True``.
    - Specifies if all the experiment coordinates are homogeneously spaced. If, e.g., ``x0`` was generated with ``np.logspace(0, 15, 10)`` then this attribute must be ``False``.
- ``tuid`` (``str``)
    - The unique identifier of the dataset. See :class:`quantify_core.data.types.TUID`.
- ``quantify_dataset_version`` (``str``)
    - The quantify dataset version.


.. jupyter-execute::

    dataset.attrs


Note that xarray automatically provides the attributes as python attributes:


.. jupyter-execute::

    dataset.quantify_dataset_version, dataset.tuid


Experiment coordinates and variables attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Both, the experiment coordinates and the experiment variables, are required to have the following attributes:

- ``standard_name`` (``str``)
    - Usually a short name. Often corresponding to the name of a :class:`~qcodes.instrument.parameter.Parameter`.
    - The name should be a valid python variable composed of lower-case alphanumeric characters and ``_`` (unserscore).
- ``long_name`` (``str``)
    - A human readable name. Usually used as the label of a plot axis.
- ``units`` (``str``)
    - The unit(s) of this experiment coordinate. If has no units, use an empty string: ``""``. If the units are arbitrary use ``"arb. un."``.
    - NB This attribute was not named ``unit`` to preserve compatibility with xarray plotting methods.

Optionally the following attributes may be present as well:

- ``batched`` (``bool``)
    - Specifies if the data acquisition supported the batched mode. See also :ref:`.batched and .batch_size <sec-bached-and-batch_size>` section.
- ``batch_size`` (``bool``)
    - When ``batched=True``, ``batch_size`` specifies the (maximum) size of a batch for this particular experiment coordinate/variables. See also :ref:`.batched and .batch_size <sec-bached-and-batch_size>` section.


.. jupyter-execute::

    dataset_2d_example.x0.attrs, dataset_2d_example.x0.standard_name


Calibration variables and dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Calibration points can be tricky to deal with. In addition to the specification above, we describe here how and which kind of calibration points are supported within the Quantify dataset.

Calibration points are stored as xarray data variables. We shall refer to them as *calibration variables*. They are similar to the experiment variables with the following differences:

- They are xarray data variables named as ``y{j}_calib``.
- They must lie along the ``acq_set_{i}_calib``, i.e. ``y{j}_calib(repetition, acq_set_{i}_calib, <other nested dimension(s)>)``.
    - Note that we would have ``y{j}(repetition, acq_set_{i}, <other nested dimension(s)>)``.
- ``y{i}_<arbitrary>_calib`` must be also present if both ``y{i}_calib`` and ``y{i}_<arbitary>`` are present in the dataset.

.. note::

    The number of elements in ``y{j}`` and ``y{j}_calib`` are independent. Usually there are only a few calibration points.



.. admonition:: Examples good datasets (variables)
    :class: dropdown

    To be added...

    - T1 with calibration points.
    - T1 with calibration points and raw traces included also for the calibration points.



T1 dataset examples
-------------------


.. admonition:: Mock data utilities
    :class: dropdown


    .. jupyter-execute::

        def generate_mock_iq_data(
            n_shots, sigma=0.3, center0=(1, 1), center1=(1, -1), prob=0.5
        ):
            """
            Generates two clusters of I,Q points with a Gaussian distribution.
            """
            i_data = np.zeros(n_shots)
            q_data = np.zeros(n_shots)
            for i in range(n_shots):
                c = center0 if (np.random.rand() >= prob) else center1
                i_data[i] = np.random.normal(c[0], sigma)
                q_data[i] = np.random.normal(c[1], sigma)
            return i_data + 1j * q_data


        def generate_exp_decay_probablity(time: np.ndarray, tau: float):
            return np.exp(-time / tau)

        def generate_trace_time(
            sampling_rate: float = 1e9,
            trace_duratation: float = 1e-6
        ):
            trace_length = sampling_rate * trace_duratation
            return np.arange(0, trace_length, 1) / sampling_rate

        def generate_trace_for_iq_point(
            iq_amp: complex,
            tbase: np.ndarray = generate_trace_time(),
            intermediate_freq: float = 50e6,
        ) -> tuple:
            """
            Generates mock traces that a physical instrument would digitize for the readout of
            a transmon qubit.
            """

            return iq_amp * np.exp(2.0j * np.pi * intermediate_freq * tbase)


        def plot_centroids(ax, ground, excited):
            ax.plot(
                [ground[0]],
                [ground[1]],
                label="|0>",
                marker="o",
                color="C3",
                markersize=10,
            )
            ax.plot(
                [excited[0]],
                [excited[1]],
                label="|1>",
                marker="^",
                color="C4",
                markersize=10,
            )


    .. jupyter-execute::

        center_ground = (-0.2, 0.65)
        center_excited = (0.7, -0, 4)

        shots = generate_mock_iq_data(
            n_shots=1024, sigma=0.15, center0=center_ground, center1=center_excited, prob=0.4
        )


    .. jupyter-execute::

        plt.hexbin(shots.real, shots.imag)
        plt.xlabel("I")
        plt.ylabel("Q")
        plot_centroids(plt.gca(), center_ground, center_excited)


    .. jupyter-execute::

        time = generate_trace_time()
        trace = generate_trace_for_iq_point(shots[0])

        fig, ax = plt.subplots(1, 1, figsize=(30, 5))
        ax.plot(time, trace.imag, ".-");
        ax.plot(time, trace.real, ".-");


T1 experiment averaged
~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    # parameters of our qubit model
    tau = 30e-6
    center_ground = (-0.2, 0.65)
    center_excited = (0.7, -0, 4)

    # mock of data acquisition configuration
    num_shots = 1024
    x0s = np.linspace(0, 150e-6, 30)
    time_par = ManualParameter(name="time", label="Time", unit="s")
    q0_iq_par = ManualParameter(name="q0_iq", label="Q0 IQ amplitude", unit="V")

    probabilities = generate_exp_decay_probablity(time=x0s, tau=tau)
    plt.ylabel("|1> probability")
    plt.suptitle("Typical T1 experiment processed data")
    plt.plot(x0s, probabilities, ".-");


.. jupyter-execute::

    y0s = np.fromiter(
        (
            np.average(
                generate_mock_iq_data(
                    n_shots=num_shots,
                    sigma=0.15,
                    center0=center_ground,
                    center1=center_excited,
                    prob=prob,
                )
            )
            for prob in probabilities
        ),
        dtype=complex,
    )

    dataset = dataset_2d_example = xr.Dataset(
        data_vars={
            "y0": ("acq_set_0", y0s, par_to_attrs(q0_iq_par)),
        },
        coords={
            "x0": ("acq_set_0", x0s, par_to_attrs(time_par)),
        },
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(dataset, dimension="acq_set_0")
    dataset_gridded


.. admonition:: Plotting utilities
    :class: dropdown


    .. jupyter-execute::

        def plot_decay_no_repetition(gridded_dataset, ax=None):
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            gridded_dataset.y0.real.plot(ax=ax, marker=".", label="I data")
            gridded_dataset.y0.imag.plot(ax=ax, marker=".", label="Q data")
            # ax.set_ylabel(f"{gridded_dataset.y0.standard_name} [{gridded_dataset.y0.units}]")
            ax.set_title(f"y0 shape = {gridded_dataset.y0.shape}")
            ax.legend()
            return ax.get_figure(), ax


        def plot_iq_no_repetition(gridded_dataset, ax=None):
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            ax.plot(
                gridded_dataset.y0.real,
                gridded_dataset.y0.imag,
                ".-",
                label="Data on IQ plane",
                color="C2",
            )
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            plot_centroids(ax, center_ground, center_excited)
            ax.legend()

            return ax.get_figure(), ax


.. jupyter-execute::

    plot_decay_no_repetition(dataset_gridded)
    plot_iq_no_repetition(dataset_gridded);


T1 experiment averaged with calibration points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    y0s = np.fromiter(
        (
            np.average(
                generate_mock_iq_data(
                    n_shots=num_shots,
                    sigma=0.15,
                    center0=center_ground,
                    center1=center_excited,
                    prob=prob,
                )
            )
            for prob in probabilities
        ),
        dtype=complex,
    )

    y0s_calib = np.fromiter(
        (
            np.average(
                generate_mock_iq_data(
                    n_shots=num_shots,
                    sigma=0.15,
                    center0=center_ground,
                    center1=center_excited,
                    prob=prob,
                )
            )
            for prob in [0, 1]
        ),
        dtype=complex,
    )

    dataset = dataset_2d_example = xr.Dataset(
        data_vars={
            "y0": ("acq_set_0", y0s, par_to_attrs(q0_iq_par)),
            "y0_calib": ("acq_set_0_calib", y0s_calib, par_to_attrs(q0_iq_par)),
        },
        coords={
            "x0": ("acq_set_0", x0s, par_to_attrs(time_par)),
            "x0_calib": (
                "acq_set_0_calib",
                ["|0>", "|1>"],
                {"standard_name": "q0_state", "long_name": "Q0 State", "unit": ""},
            ),
        },
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="acq_set_0", coords_names=["x0"]
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="acq_set_0_calib", coords_names=["x0_calib"]
    )
    dataset_gridded


.. jupyter-execute::

    fig = plt.figure(figsize=(8, 5))

    ax = plt.subplot2grid((1, 10), (0, 0), colspan=9, fig=fig)
    plot_decay_no_repetition(dataset_gridded, ax=ax)

    ax_calib = plt.subplot2grid((1, 10), (0, 9), colspan=1, fig=fig, sharey=ax)
    dataset_gridded.y0_calib.real.plot(marker="o", ax=ax_calib)
    dataset_gridded.y0_calib.imag.plot(marker="o", ax=ax_calib)
    ax_calib.yaxis.set_label_position("right")
    ax_calib.yaxis.tick_right()

    plot_iq_no_repetition(dataset_gridded);


We can use the calibration points to normalize the data and obtain the typical T1 decay.


.. admonition:: Data rotation and normalization utilities
    :class: dropdown


    .. jupyter-execute::

        def rotate_data(complex_data: np.ndarray, angle: float) -> np.ndarray:
            """
            Rotates data on the complex plane around `0 + 0j`.

            Parameters
            ----------
            complex_data
                Data to rotate.
            angle
                Angle to rotate it by (in degrees).

            Returns
            -------
            :
                Rotated data.
            """
            angle_r = np.deg2rad(angle)
            rotation = np.cos(angle_r) + 1j * np.sin(angle_r)
            return rotation * complex_data


        def find_rotation_angle(z1: complex, z2: complex) -> float:
            """
            Finds the angle of the line between two complex numbers on the complex plane with
            respect to the real axis.

            Parameters
            ----------
            z1
                First complex number.
            z2
                Second complex number.

            Returns
            -------
            :
                The angle found (in degrees).
            """
            return np.rad2deg(np.angle(z1 - z2))




The normalization to the calibration point could look like this:


.. jupyter-execute::

    angle = find_rotation_angle(*dataset_gridded.y0_calib.values)
    y0_rotated = rotate_data(dataset_gridded.y0, - angle)
    y0_calib_rotated = rotate_data(dataset_gridded.y0_calib, - angle)
    calib_0, calib_1 = y0_calib_rotated.sel(x0_calib="|0>").values, y0_calib_rotated.sel(x0_calib="|1>").values
    y0_norm = (y0_rotated - calib_0) / (calib_1 - calib_0)
    y0_norm.attrs["long_name"] = "|1> Population"
    y0_norm.attrs["units"] = ""
    plot_decay_no_repetition(y0_norm.to_dataset());


T1 experiment storing all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    y0s = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=0.15,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in probabilities
        )
    ).T

    y0s_calib = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=0.15,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in [0, 1]
        )
    ).T

    dataset = dataset_2d_example = xr.Dataset(
        data_vars={
            "y0": ("acq_set_0", y0s.mean(axis=0), par_to_attrs(q0_iq_par)),
            "y0_calib": ("acq_set_0_calib", y0s_calib.mean(axis=0), par_to_attrs(q0_iq_par)),
            "y0_shots": (("repetition", "acq_set_0"), y0s, par_to_attrs(q0_iq_par)),
            "y0_shots_calib": (("repetition", "acq_set_0_calib"), y0s_calib, par_to_attrs(q0_iq_par)),
        },
        coords={
            "x0": ("acq_set_0", x0s, par_to_attrs(time_par)),
            "x0_calib": (
                "acq_set_0_calib",
                ["|0>", "|1>"],
                {"standard_name": "q0_state", "long_name": "Q0 State", "unit": ""},
            ),
        },
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="acq_set_0", coords_names=["x0"]
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="acq_set_0_calib", coords_names=["x0_calib"]
    )
    dataset_gridded


In this dataset we have both the averaged values and all the shots. The averaged values can be plotted in the same way as before.


.. jupyter-execute::

    # plot_decay_no_repetition(dataset_gridded)
    # plot_iq_no_repetition(dataset_gridded);


Here we focus on inspecting how the individual shots are distributed on the IQ plane for some particular `Time` values.

Note that we are plotting the calibration points as well.


.. jupyter-execute::

    for t_example in [x0s[len(x0s) // 5], x0s[-5]]:
        shots_example = (
            dataset_gridded.y0_shots.real.sel(x0=t_example),
            dataset_gridded.y0_shots.imag.sel(x0=t_example),
        )
        plt.hexbin(*shots_example)
        plt.xlabel("I")
        plt.ylabel("Q")
        calib_0 = dataset_gridded.y0_calib.sel(x0_calib="|0>")
        calib_1 = dataset_gridded.y0_calib.sel(x0_calib="|1>")
        plot_centroids(plt.gca(), (calib_0.real, calib_0.imag), (calib_1.real, calib_1.imag))
        plt.suptitle(f"Shots fot t = {t_example:.5f} s")
        plt.show()


We can colapse (average along) the `repetion` dimension:


.. admonition:: Plotting utility
    :class: dropdown


    .. jupyter-execute::

        def plot_iq_decay_repetition(gridded_dataset):
            gridded_dataset.y0_shots.real.mean(dim="repetition").plot(
                marker=".", label="I data"
            )
            gridded_dataset.y0_shots.imag.mean(dim="repetition").plot(
                marker=".", label="Q data"
            )
            plt.ylabel(f"{gridded_dataset.y0_shots.long_name} [{gridded_dataset.y0_shots.units}]")
            plt.suptitle(f"{gridded_dataset.y0_shots.name} shape = {gridded_dataset.y0_shots.shape}")
            plt.legend()

            fig, ax = plt.subplots(1, 1)
            ax.plot(
                gridded_dataset.y0_shots.real.mean(
                    dim="repetition"
                ),  # "collapses" outer dimension
                gridded_dataset.y0_shots.imag.mean(
                    dim="repetition"
                ),  # "collapses" outer dimension
                ".-",
                label="Data on IQ plane",
                color="C2",
            )
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            plot_centroids(ax, center_ground, center_excited)
            ax.legend()


.. jupyter-execute::

    plot_iq_decay_repetition(dataset_gridded)


T1 experiment storing digitized signals for all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    y0s = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=0.15,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in probabilities
        )
    ).T

    _y0s_traces = np.array(tuple(map(generate_trace_for_iq_point, y0s.flatten())))
    y0s_traces = _y0s_traces.reshape(*y0s.shape, _y0s_traces.shape[-1])

    y0s_calib = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=0.15,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in [0, 1]
        )
    ).T

    _y0s_traces_calib = np.array(tuple(map(generate_trace_for_iq_point, y0s_calib.flatten())))
    y0s_traces_calib = _y0s_traces_calib.reshape(*y0s_calib.shape, _y0s_traces_calib.shape[-1])

    dataset = dataset_2d_example = xr.Dataset(
        data_vars={
            "y0": ("acq_set_0", y0s.mean(axis=0), par_to_attrs(q0_iq_par)),
            "y0_calib": ("acq_set_0_calib", y0s_calib.mean(axis=0), par_to_attrs(q0_iq_par)),
            "y0_shots": (("repetition", "acq_set_0"), y0s, par_to_attrs(q0_iq_par)),
            "y0_shots_calib": (("repetition", "acq_set_0_calib"), y0s_calib, par_to_attrs(q0_iq_par)),
            "y0_traces": (("repetition", "acq_set_0", "time"), y0s_traces, par_to_attrs(q0_iq_par)),
            "y0_traces_calib": (("repetition", "acq_set_0_calib", "time"), y0s_traces_calib, par_to_attrs(q0_iq_par)),
        },
        coords={
            "x0": ("acq_set_0", x0s, par_to_attrs(time_par)),
            "x0_calib": (
                "acq_set_0_calib",
                ["|0>", "|1>"],
                {"standard_name": "q0_state", "long_name": "Q0 State", "unit": ""},
            ),
            "time": ("time", generate_trace_time(), {"standard_name": "time_samples", "long_name": "Time", "unit": "V"})
        },
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="acq_set_0", coords_names=["x0"]
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="acq_set_0_calib", coords_names=["x0_calib"]
    )
    dataset_gridded


All the previous data is also present, but in this dataset we can inspect the IQ signal for each individual shot. Let's inspect the signal of the first shot number 777 of the last point of the T1 experiment:


.. jupyter-execute::

    dataset_gridded.y0_traces.shape # dimensions: (repetition, x0, time)


.. jupyter-execute::

    trace_example = dataset_gridded.y0_traces.sel(repetition=777, x0=dataset_gridded.x0[-1])
    trace_example.shape, trace_example.dtype


For clarity, we plot only part of this digitized signal:


.. jupyter-execute::

    trace_example_plt = trace_example[:200]
    trace_example_plt.real.plot(figsize=(15, 5), marker=".");
    trace_example_plt.imag.plot(marker=".");
