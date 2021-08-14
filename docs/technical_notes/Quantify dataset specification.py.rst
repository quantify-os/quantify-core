Quantify dataset specification
==============================


.. warning::

    I have "removed" all the text from the docs build so that you can focus on seeing how the same datasets would look like in a new format proposal.


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
        from pathlib import Path
        from quantify_core.data.handling import get_datadir, set_datadir
        from typing import List, Tuple

        pretty.install()


        def assign_dataset_attrs(ds: xr.Dataset) -> dict:
            tuid = dh.gen_tuid()
            ds.attrs.update(
                dict(
                    grid=True,
                    grid_uniformly_spaced=True,  # pyqt requires interpolation
                    tuid=tuid,
                    quantify_dataset_version="v1.0",
                    # experiment_coords=[],
                    # experiment_data_vars=[],
                    # calibration_data_vars_map=[] # List[Tuple[str, str]]
                )
            )
            return ds.attrs


        def dataset_round_trip(ds: xr.Dataset) -> xr.Dataset:
            assign_dataset_attrs(ds)
            tuid = ds.attrs["tuid"]
            dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, ds)
            return dh.load_dataset(tuid)


        def par_to_attrs(par):
            return {"units": par.unit, "long_name": par.label}


        set_datadir(Path.home() / "quantify-data")  # change me!


Introduction
------------


.. jupyter-execute::

    ## notebook-to-rst-json-conf: {"indent": "    "}

    x0s = np.linspace(0.45, 0.55, 30)
    x1s = np.linspace(0, 100e-9, 40)
    time_par = ManualParameter(name="time", label="Time", unit="s")
    amp_par = ManualParameter(name="amp", label="Flux amplitude", unit="V")
    pop_q0_par = ManualParameter(name="pop_q0", label="Population Q0", unit="arb. unit")
    pop_q1_par = ManualParameter(name="pop_q1", label="Population Q1", unit="arb. unit")

    x0s, x1s = grid_setpoints([x0s, x1s], [amp_par, time_par]).T
    x0s_norm = np.abs((x0s - x0s.mean()) / (x0s - x0s.mean()).max())
    y0s = (1 - x0s_norm) * np.sin(
        2 * np.pi * x1s * 1 / 30e-9 * (x0s_norm + 0.5)
    )  # ~chevron
    y1s = -y0s  # mock inverted population for q1

    dataset = dataset_2d_example = xr.Dataset(
        data_vars={
            pop_q0_par.name: (
                ("repetition_dim_0", "dim_0"),
                [y0s + np.random.random(y0s.shape) / k for k in (100, 10, 5)],
                par_to_attrs(pop_q0_par),
            ),
            pop_q1_par.name: (
                ("repetition_dim_0", "dim_0"),
                [y1s + np.random.random(y1s.shape) / k for k in (100, 10, 5)],
                par_to_attrs(pop_q1_par),
            ),
        },
        coords={
            amp_par.name: ("dim_0", x0s, par_to_attrs(amp_par)),
            time_par.name: ("dim_0", x1s, par_to_attrs(time_par)),
        },
        attrs=dict(
            experiment_coords=[amp_par.name, time_par.name],
            experiment_data_vars=[pop_q0_par.name, pop_q1_par.name],
            calibration_data_vars_map=[],
        ),
    )

    assert dataset == dataset_round_trip(dataset)  # confirm read/write


.. jupyter-execute::

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset_2d_example,
        dimension="dim_0",
        coords_names=dataset_2d_example.experiment_coords,
    )
    dataset_gridded.pop_q0.plot.pcolormesh(x="amp", col="repetition_dim_0")
    dataset_gridded.pop_q1.plot.pcolormesh(x="amp", col="repetition_dim_0")
    pass


.. jupyter-execute::

    dataset_gridded.pop_q0.mean(dim="repetition_dim_0").plot(x="amp")
    pass


Quantify dataset: detailed specification
----------------------------------------


Xarray dimensions
~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    ## notebook-to-rst-json-conf: {"indent": "    "}

    dataset = xr.Dataset(
        data_vars={
            pop_q0_par.name: (
                ("repetition_dim_0", "dim_0"),
                [y0s + np.random.random(y0s.shape) / k for k in (100, 10, 5)],
                par_to_attrs(pop_q0_par),
            ),
            pop_q1_par.name: (
                ("repetition_dim_0", "dim_0"),
                [y1s + np.random.random(y1s.shape) / k for k in (100, 10, 5)],
                par_to_attrs(pop_q1_par),
            ),
        },
        coords={
            amp_par.name: ("dim_0", x0s, par_to_attrs(amp_par)),
            time_par.name: ("dim_0", x1s, par_to_attrs(time_par)),
            # here we choose to index the repetition dimension with an array of strings
            "repetition_dim_0": (
                "repetition_dim_0",
                ["noisy", "very noisy", "very very noisy"],
            ),
        },
        attrs=dict(
            experiment_coords=[amp_par.name, time_par.name],
            experiment_data_vars=[pop_q0_par.name, pop_q1_par.name],
            calibration_data_vars_map=[],
        ),
    )

    dataset


.. jupyter-execute::

    ## notebook-to-rst-json-conf: {"indent": "    "}

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="dim_0", coords_names=dataset.experiment_coords
    )
    dataset_gridded


.. jupyter-execute::

    ## notebook-to-rst-json-conf: {"indent": "    "}

    dataset_gridded.pop_q0.sel(repetition_dim_0="very noisy").plot(x="amp")
    pass


Xarray coordinates
~~~~~~~~~~~~~~~~~~


Xarray variables
~~~~~~~~~~~~~~~~


Dataset attributes
~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    dataset_2d_example.attrs


Note that xarray automatically provides the attributes as python attributes:


.. jupyter-execute::

    dataset_2d_example.quantify_dataset_version, dataset_2d_example.tuid


Experiment coordinates and variables attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    dataset_2d_example.amp.attrs, dataset_2d_example.time.long_name


Calibration variables and dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


        def generate_trace_time(sampling_rate: float = 1e9, trace_duratation: float = 0.3e-6):
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
            n_shots=256, sigma=0.1, center0=center_ground, center1=center_excited, prob=0.4
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
        ax.plot(time, trace.imag, ".-")
        _ = ax.plot(time, trace.real, ".-")


T1 experiment averaged
~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    # parameters of our qubit model
    tau = 30e-6
    center_ground = (-0.2, 0.65)
    center_excited = (0.7, -0, 4)
    sigma = 0.1

    # mock of data acquisition configuration
    num_shots = 256
    x0s = np.linspace(0, 150e-6, 30)
    time_par = ManualParameter(name="time", label="Time", unit="s")
    q0_iq_par = ManualParameter(name="q0_iq", label="Q0 IQ amplitude", unit="V")

    probabilities = generate_exp_decay_probablity(time=x0s, tau=tau)
    plt.ylabel("|1> probability")
    plt.suptitle("Typical T1 experiment processed data")
    _ = plt.plot(x0s, probabilities, ".-")


.. jupyter-execute::

    y0s = np.fromiter(
        (
            np.average(
                generate_mock_iq_data(
                    n_shots=num_shots,
                    sigma=sigma,
                    center0=center_ground,
                    center1=center_excited,
                    prob=prob,
                )
            )
            for prob in probabilities
        ),
        dtype=complex,
    )

    dataset = xr.Dataset(
        data_vars={
            q0_iq_par.name: ("dim_0", y0s, par_to_attrs(q0_iq_par)),
        },
        coords={
            time_par.name: ("dim_0", x0s, par_to_attrs(time_par)),
        },
        attrs=dict(
            experiment_coords=[time_par.name],
            experiment_data_vars=[q0_iq_par.name],
            calibration_data_vars_map=[],
        ),
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="dim_0", coords_names=dataset.experiment_coords
    )
    dataset_gridded


.. admonition:: Plotting utilities
    :class: dropdown


    .. jupyter-execute::


        def plot_decay_no_repetition(gridded_dataset, ax=None):
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            y0 = gridded_dataset[gridded_dataset.experiment_data_vars[0]]
            y0.real.plot(ax=ax, marker=".", label="I data")
            y0.imag.plot(ax=ax, marker=".", label="Q data")
            ax.set_title(f"{y0.long_name} shape = {y0.shape}")
            ax.legend()
            return ax.get_figure(), ax


        def plot_iq_no_repetition(gridded_dataset, ax=None):
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            y0 = gridded_dataset[gridded_dataset.experiment_data_vars[0]]
            ax.plot(
                y0.real,
                y0.imag,
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
    _ = plot_iq_no_repetition(dataset_gridded)


T1 experiment averaged with calibration points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    y0s = np.fromiter(
        (
            np.average(
                generate_mock_iq_data(
                    n_shots=num_shots,
                    sigma=sigma,
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
                    sigma=sigma,
                    center0=center_ground,
                    center1=center_excited,
                    prob=prob,
                )
            )
            for prob in [0, 1]
        ),
        dtype=complex,
    )

    dataset = xr.Dataset(
        data_vars={
            q0_iq_par.name: ("dim_0", y0s, par_to_attrs(q0_iq_par)),
            f"{q0_iq_par.name}_cal": ("dim_0_cal", y0s_calib, par_to_attrs(q0_iq_par)),
        },
        coords={
            time_par.name: ("dim_0", x0s, par_to_attrs(time_par)),
            f"cal": (
                "dim_0_cal",
                ["|0>", "|1>"],
                {"long_name": "Q0 State", "unit": ""},
            ),
        },
        attrs=dict(
            experiment_coords=[time_par.name],
            experiment_data_vars=[q0_iq_par.name],
            calibration_data_vars_map=[(q0_iq_par.name, f"{q0_iq_par.name}_cal")],
            calibration_coords_map=[(time_par.name, "cal")],
        ),
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="dim_0", coords_names=dataset.experiment_coords
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="dim_0_cal", coords_names=["cal"]
    )
    dataset_gridded


.. jupyter-execute::

    fig = plt.figure(figsize=(8, 5))

    ax = plt.subplot2grid((1, 10), (0, 0), colspan=9, fig=fig)
    plot_decay_no_repetition(dataset_gridded, ax=ax)

    ax_calib = plt.subplot2grid((1, 10), (0, 9), colspan=1, fig=fig, sharey=ax)
    dataset_gridded.q0_iq_cal.real.plot(marker="o", ax=ax_calib)
    dataset_gridded.q0_iq_cal.imag.plot(marker="o", ax=ax_calib)
    ax_calib.yaxis.set_label_position("right")
    ax_calib.yaxis.tick_right()

    _ = plot_iq_no_repetition(dataset_gridded)


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

    angle = find_rotation_angle(*dataset_gridded.q0_iq_cal.values)
    y0_rotated = rotate_data(dataset_gridded.q0_iq, -angle)
    y0_calib_rotated = rotate_data(dataset_gridded.q0_iq_cal, -angle)
    calib_0, calib_1 = (
        y0_calib_rotated.sel(cal="|0>").values,
        y0_calib_rotated.sel(cal="|1>").values,
    )
    y0_norm = (y0_rotated - calib_0) / (calib_1 - calib_0)
    y0_norm.attrs["long_name"] = "|1> Population"
    y0_norm.attrs["units"] = ""
    dataset_tmp = y0_norm.to_dataset()
    dataset_tmp.attrs.update(dataset_gridded.attrs)
    _ = plot_decay_no_repetition(dataset_tmp)


T1 experiment storing all shots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    y0s = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=sigma,
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
                sigma=sigma,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in [0, 1]
        )
    ).T

    dataset = xr.Dataset(
        data_vars={
            q0_iq_par.name: ("dim_0", y0s.mean(axis=0), par_to_attrs(q0_iq_par)),
            f"{q0_iq_par.name}_cal": (
                "dim_0_cal",
                y0s_calib.mean(axis=0),
                par_to_attrs(q0_iq_par),
            ),
            f"{q0_iq_par.name}_shots": (
                ("repetition_dim_0", "dim_0"),
                y0s,
                par_to_attrs(q0_iq_par),
            ),
            f"{q0_iq_par.name}_shots_cal": (
                ("repetition_dim_0", "dim_0_cal"),
                y0s_calib,
                par_to_attrs(q0_iq_par),
            ),
        },
        coords={
            time_par.name: ("dim_0", x0s, par_to_attrs(time_par)),
            "cal": (
                "dim_0_cal",
                ["|0>", "|1>"],
                {"long_name": "Q0 State", "unit": ""},
            ),
        },
        attrs=dict(
            experiment_coords=[time_par.name],
            experiment_data_vars=[q0_iq_par.name, f"{q0_iq_par.name}_shots"],
            calibration_data_vars_map=[
                (q0_iq_par.name, f"{q0_iq_par.name}_cal"),
                (f"{q0_iq_par.name}_shots", f"{q0_iq_par.name}_shots_cal"),
            ],
            calibration_coords_map=[
                (time_par.name, "cal"),
            ],
        ),
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="dim_0", coords_names=dataset.experiment_coords
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="dim_0_cal", coords_names=["cal"]
    )
    dataset_gridded


In this dataset we have both the averaged values and all the shots. The averaged values can be plotted in the same way as before.


.. jupyter-execute::

    plot_decay_no_repetition(dataset_gridded)
    plot_iq_no_repetition(dataset_gridded)


Here we focus on inspecting how the individual shots are distributed on the IQ plane for some particular `Time` values.

Note that we are plotting the calibration points as well.


.. jupyter-execute::

    for t_example in [x0s[len(x0s) // 5], x0s[-5]]:
        shots_example = (
            dataset_gridded.q0_iq_shots.real.sel(time=t_example),
            dataset_gridded.q0_iq_shots.imag.sel(time=t_example),
        )
        plt.hexbin(*shots_example)
        plt.xlabel("I")
        plt.ylabel("Q")
        calib_0 = dataset_gridded.q0_iq_cal.sel(cal="|0>")
        calib_1 = dataset_gridded.q0_iq_cal.sel(cal="|1>")
        plot_centroids(
            plt.gca(), (calib_0.real, calib_0.imag), (calib_1.real, calib_1.imag)
        )
        plt.suptitle(f"Shots fot t = {t_example:.5f} s")
        plt.show()


.. admonition:: Plotting utility
    :class: dropdown


    .. jupyter-execute::


        def plot_iq_decay_repetition(gridded_dataset):
            y0_shots = gridded_dataset.q0_iq_shots
            y0_shots.real.mean(dim="repetition_dim_0").plot(marker=".", label="I data")
            y0_shots.imag.mean(dim="repetition_dim_0").plot(marker=".", label="Q data")
            plt.ylabel(f"{y0_shots.long_name} [{y0_shots.units}]")
            plt.suptitle(f"{y0_shots.name} shape = {y0_shots.shape}")
            plt.legend()

            fig, ax = plt.subplots(1, 1)
            ax.plot(
                y0_shots.real.mean(dim="repetition_dim_0"),  # "collapses" outer dimension
                y0_shots.imag.mean(dim="repetition_dim_0"),  # "collapses" outer dimension
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

    # NB this is not necessarily the most efficient way to generate this mock data
    y0s = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=sigma,
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
                sigma=sigma,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in [0, 1]
        )
    ).T

    _y0s_traces_calib = np.array(
        tuple(map(generate_trace_for_iq_point, y0s_calib.flatten()))
    )
    y0s_traces_calib = _y0s_traces_calib.reshape(
        *y0s_calib.shape, _y0s_traces_calib.shape[-1]
    )

    dataset = xr.Dataset(
        data_vars={
            f"{q0_iq_par.name}": ("dim_0", y0s.mean(axis=0), par_to_attrs(q0_iq_par)),
            f"{q0_iq_par.name}_cal": (
                "dim_0_cal",
                y0s_calib.mean(axis=0),
                par_to_attrs(q0_iq_par),
            ),
            f"{q0_iq_par.name}_shots": (
                ("repetition_dim_0", "dim_0"),
                y0s,
                par_to_attrs(q0_iq_par),
            ),
            f"{q0_iq_par.name}_shots_cal": (
                ("repetition_dim_0", "dim_0_cal"),
                y0s_calib,
                par_to_attrs(q0_iq_par),
            ),
            f"{q0_iq_par.name}_traces": (
                ("repetition_dim_0", "dim_0", "dim_1"),
                y0s_traces,
                par_to_attrs(q0_iq_par),
            ),
            f"{q0_iq_par.name}_traces_cal": (
                ("repetition_dim_0", "dim_0_cal", "dim_1"),
                y0s_traces_calib,
                par_to_attrs(q0_iq_par),
            ),
        },
        coords={
            time_par.name: ("dim_0", x0s, par_to_attrs(time_par)),
            "cal": (
                "dim_0_cal",
                ["|0>", "|1>"],
                {"long_name": "Q0 State", "unit": ""},
            ),
            "trace_time": (
                "dim_1",
                generate_trace_time(),
                {"long_name": "Time", "unit": "V"},
            ),
        },
        attrs=dict(
            experiment_coords=[time_par.name],
            experiment_data_vars=[
                q0_iq_par.name,
                f"{q0_iq_par.name}_shots",
                f"{q0_iq_par.name}_traces",
            ],
            calibration_data_vars_map=[
                (q0_iq_par.name, f"{q0_iq_par.name}_cal"),
                (f"{q0_iq_par.name}_shots", f"{q0_iq_par.name}_shots_cal"),
                (f"{q0_iq_par.name}_traces", f"{q0_iq_par.name}_traces_cal"),
            ],
            calibration_coords_map=[
                (time_par.name, "cal"),
            ],
        ),
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="dim_0", coords_names=dataset.experiment_coords
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="dim_0_cal", coords_names=["cal"]
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="dim_1", coords_names=["trace_time"]
    )
    dataset_gridded


.. jupyter-execute::

    dataset_gridded.q0_iq_traces.shape  # dimensions: (repetition, x0, time)


.. jupyter-execute::

    trace_example = dataset_gridded.q0_iq_traces.sel(
        repetition_dim_0=123, time=dataset_gridded.time[-1]
    )
    trace_example.shape, trace_example.dtype


.. jupyter-execute::

    trace_example_plt = trace_example[:200]
    trace_example_plt.real.plot(figsize=(15, 5), marker=".")
    _ = trace_example_plt.imag.plot(marker=".")


A "weird"/"unstructured" experiment and dataset example
=======================================================


Schdule reference: `one of the latest papers from DiCarlo Lab <https://arxiv.org/abs/2102.13071>`_, Fig. 4b.

NB not exactly the same schedule, but what matter are the measurements.


.. jupyter-execute::

    from quantify_scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib
    from quantify_scheduler import Schedule
    from quantify_scheduler.gate_library import Reset, Measure, CZ, Rxy, X90, X, Y, Y90, X90

    d1, d2, d3, d4 = [f"D{i}" for i in range(1, 5)]
    a1, a2, a3 = [f"A{i}" for i in range(1, 4)]

    all_qubits = d1, d2, d3, d4, a1, a2, a3

    sched = Schedule(f"S7 dance")

    sched.add(Reset(*all_qubits))

    num_cycles = 4

    for cycle in range(num_cycles):
        sched.add(Y90(d1))
        for q in [d2, d3, d4]:
            sched.add(Y90(q), ref_pt="start", rel_time=0)
        sched.add(Y90(a2), ref_pt="start", rel_time=0)

        for q in [d2, d1, d4, d3]:
            sched.add(CZ(qC=q, qT=a2))

        sched.add(Y90(d1))
        for q in [d2, d3, d4]:
            sched.add(Y90(q), ref_pt="start", rel_time=0)
        sched.add(Y90(a2), ref_pt="start", rel_time=0)

        sched.add(Y90(a1), ref_pt="end", rel_time=0)
        sched.add(Y90(a3), ref_pt="start", rel_time=0)

        sched.add(CZ(qC=d1, qT=a1))
        sched.add(CZ(qC=d2, qT=a3))
        sched.add(CZ(qC=d3, qT=a1))
        sched.add(CZ(qC=d4, qT=a3))

        sched.add(Y90(a1), ref_pt="end", rel_time=0)
        sched.add(Y90(a3), ref_pt="start", rel_time=0)

        sched.add(Measure(a2, acq_index=cycle))
        for q in (a1, a3):
            sched.add(Measure(q, acq_index=cycle), ref_pt="start", rel_time=0)

        for q in [d1, d2, d3, d4]:
            sched.add(X(q), ref_pt="start", rel_time=0)

    # final measurements

    sched.add(Measure(*all_qubits[:4], acq_index=0), ref_pt="end", rel_time=0)

    f, ax = circuit_diagram_matplotlib(sched)
    # f.set_figheight(10)
    f.set_figwidth(30)


How do we store all shots for this measurement? (we want it because, e.g., we know we have issue with leakage to the second excited state)


.. jupyter-execute::

    num_shots = 128
    center_ground = (-0.2, 0.65)
    center_excited = (0.7, -0, 4)
    sigma = 0.1

    cycles = range(num_cycles)

    radom_data = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=sigma,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in [np.random.random() for _ in cycles]
        )
    ).T

    radom_data_final = np.array(
        tuple(
            generate_mock_iq_data(
                n_shots=num_shots,
                sigma=sigma,
                center0=center_ground,
                center1=center_excited,
                prob=prob,
            )
            for prob in [np.random.random()]
        )
    ).T

    # NB same random data is used for all qubits only for the simplicity of the mock!

    data_vars = {}

    for q in (a1, a2, a3):
        data_vars[f"{q}_shots"] = (
            ("repetition_dim_0", "dim_0"),
            radom_data,
            dict(units="V", long_name=f"IQ amplitude {q}"),
        )

    for q in (d1, d2, d3, d4):
        data_vars[f"{q}_shots"] = (
            ("repetition_dim_0", "dim_1"),
            radom_data_final,
            dict(units="V", long_name=f"IQ amplitude {q}"),
        )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "cycle": (
                "dim_0",
                cycles,
                dict(units="", long_name="Surface code cycle number"),
            ),
            "final_msmt": ("dim_1", [0], dict(units="", long_name="Final measurement")),
        },
        attrs=dict(
            experiment_coords=["cycle"],
            experiment_data_vars=[a1],
            calibration_data_vars_map=[],
            calibration_coords_map=[],
        ),
    )


    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset.A1_shots.shape


.. jupyter-execute::

    dataset.D1_shots.shape


.. jupyter-execute::

    dataset_gridded = dh.to_gridded_dataset(
        dataset, dimension="dim_0", coords_names=["cycle"]
    )
    dataset_gridded = dh.to_gridded_dataset(
        dataset_gridded, dimension="dim_1", coords_names=["final_msmt"]
    )
    dataset_gridded


"Nested MeasurementControl" example
===================================


.. jupyter-execute::

    flux_bias_values = np.linspace(-0.04, 0.04, 12)

    resonator_frequencies = np.linspace(7e9, 8.5e9, len(flux_bias_values))
    qubit_frequencies = np.linspace(4.5e9, 4.6e9, len(flux_bias_values))
    t1_values = np.linspace(20e-6, 50e-6, len(flux_bias_values))

    resonator_freq_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]
    qubit_freq_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]
    t1_tuids = [dh.gen_tuid() for _ in range(len(flux_bias_values))]


.. jupyter-execute::

    dataset = xr.Dataset(
        data_vars={
            "resonator_freq": ("dim_0", res_frequencies, dict(long_name="Resonator frequency", units="Hz")),
            "qubit_freq": ("dim_0", qubit_frequencies, dict(long_name="Qubit frequency", units="Hz")),
            "t1": ("dim_0", t1_values, dict(long_name="T1", units="s")),
        },
        coords={
            "flux_bias": ("dim_0", flux_bias_values, dict(long_name="Flux bias", units="A")),
            "resonator_freq_tuids": ("dim_0", resonator_freq_tuids, dict(long_name="Dataset TUID", units="")),
            "qubit_freq_tuids": ("dim_0", qubit_freq_tuids, dict(long_name="Dataset TUID", units="")),
            "t1_tuids": ("dim_0", t1_tuids, dict(long_name="Dataset TUID", units="")),
        },
        attrs=dict(
            experiment_coords=[("flux_bias", "resonator_freq_tuids", "qubit_freq_tuids", "t1_tuids")],
            experiment_data_vars=[
                "resonator_freq", 
                "qubit_freq", 
                "t1",
        ],
            calibration_data_vars_map=[]
        )
    )

    assert dataset == dataset_round_trip(dataset)  # confirm read/write

    dataset


.. jupyter-execute::

    dataset_multi_indexed = dataset.set_index({
        "dim_0": dataset.experiment_coords[0]
    })

    dataset_multi_indexed


.. jupyter-execute::

    dataset_multi_indexed.qubit_freq.sel(resonator_freq_tuids=resonator_freq_tuids[2])


.. jupyter-execute::

    dataset_multi_indexed.qubit_freq.sel(t1_tuids=t1_tuids[2])


.. jupyter-execute::
    :raises:

    # notebook-to-rst-json-conf: {"jupyter_execute_options": [":raises:"]}

    assert dataset_multi_indexed == dataset_round_trip(dataset_multi_indexed)  # confirm read/write


.. jupyter-execute::

    all(dataset_multi_indexed.reset_index("dim_0").t1_tuids == dataset.t1_tuids)


But the `dtype` has been changed to `object` (from fixed-length string) and I do not know why, maybe bug, maybe good reasons to do it so.


.. jupyter-execute::

    dataset.t1_tuids.dtype, dataset_multi_indexed.reset_index("dim_0").t1_tuids.dtype


.. jupyter-execute::

    dataset.t1_tuids.dtype == dataset_multi_indexed.reset_index("dim_0").t1_tuids.dtype
