.. _usage:

===============
User guide
===============

Introduction
===============

A :mod:`quantify` experiment typically consists of a data-acquisition loop in which one or more parameters are set and one or more parameters are measured.

The core of Quantify can be understood by understanding the following concepts:

- `Instruments and Parameters`_
- `Measurement Control`_
- `Settables and Gettables`_
- `Data storage & Analysis`_

Code snippets
-------------

.. seealso::

    The complete source code of the examples on this page can be found in

    :jupyter-download:notebook:`usage`

    :jupyter-download:script:`usage`


Bellow we import common utilities used in the examples.

.. jupyter-execute::

    import numpy as np
    from qcodes import ManualParameter, Parameter, validators, Instrument
    from quantify.measurement import MeasurementControl
    from quantify.measurement import Gettable
    import quantify.data.handling as dh
    import xarray as xr
    import matplotlib.pyplot as plt
    from pathlib import Path
    from os.path import join
    from quantify.data.handling import set_datadir

    set_datadir(join(Path.home(), 'quantify-data'))
    MC = MeasurementControl("MC")


Instruments and Parameters
========================================
Parameter
-----------------

A parameter represents a state variable of the system.

    - A parameter can be get and/or set able.
    - Contains metadata such as units and labels.
    - Commonly implemented using the QCoDeS :class:`~qcodes.instrument.parameter.Parameter` class.
    - A parameter implmemented using the QCoDeS :class:`~qcodes.instrument.parameter.Parameter` class is a valid :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` and as such can be used directly in an experiment loop in the `Measurement Control`_. (see subsequent sections)

Instrument
-----------------

An Instrument is a container for parameters that typically (but not necessarily) corresponds to a physical piece of hardware.

Instruments provide the following functionality.

- Container for parameters.
- A standardized interface.
- Provide logging of parameters through the :meth:`~qcodes.instrument.base.Instrument.snapshot` method.
- All instruments inherit from the QCoDeS :class:`~qcodes.instrument.base.Instrument` class.
- Are shown by default in the :class:`~quantify.visualization.InstrumentMonitor`


Measurement Control
====================

The :class:`~quantify.measurement.MeasurementControl` (MC) is in charge of the data-acquisition loop and is based on the notion that, in general, an experiment consists of the following three steps:

1. Initialize (set) some parameter(s),
2. Measure (get) some parameter(s),
3. Store the data.

Quantify provides two helper classes, :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` to aid in these steps, which are explored further in later sections of this article.

:class:`~quantify.measurement.MeasurementControl` provides the following functionality

- Enforce standardization of experiments
- Standardized data storage
- Live plotting of the experiment
- n-dimensional sweeps
- Data acquisition controlled iteratively or in batches
- Adaptive sweeps (measurement points are not predetermined at the beginning of an experiment)


Basic example, a 1D iterative measurement loop
------------------------------------------------

Running an experiment is simple!
Simply define what parameters to set, and get, and what points to loop over.

In the example below we want to set frequencies on a microwave source and acquire the signal from the pulsar readout module.

.. jupyter-execute::
    :hide-code:

    mw_source1 = Instrument("mw_source1")
    # NB: for brevity only, this not the proper way of adding parameters to qcodes intruments
    mw_source1.freq = ManualParameter(
        name='freq',
        label='Frequency',
        unit='Hz',
        vals=validators.Numbers(),
        initial_value=1.0
    )

    pulsar_QRM = Instrument("pulsar_QRM")
    # NB: for brevity only, this not the proper way of adding parameters to qcodes intruments
    pulsar_QRM.signal = Parameter(
        name='sig_a',
        label='Signal',
        unit='V',
        get_cmd=lambda: mw_source1.freq() * 1e-8
    )

.. jupyter-execute::

    MC.settables(mw_source1.freq)               # We want to set the frequency of a microwave source
    MC.setpoints(np.arange(5e9, 5.2e9, 100e3))  # Scan around 5.1 GHz
    MC.gettables(pulsar_QRM.signal)             # acquire the signal from the pulsar QRM
    dset = MC.run(name='Frequency sweep')       # run the experiment


The :class:`~quantify.measurement.MeasurementControl` can also be used to perform more advanced experiments such as 2D scans, pulse-sequences where the hardware is in control of the acquisition loop, or adaptive experiments in which it is not known what data points to acquire in advance, they are determined dynamically during the experiment.
Take a look at some of the tutorial notebooks for more in-depth examples on usage and application.

Control Mode
-----------------

A very important aspect in the usage of the :class:`~quantify.measurement.MeasurementControl` is the Control Mode, which specifies whether the setpoints are processed iteratively or in batches.
Batched mode can be used to deal with constraints imposed by (hardware) resources or to reduce overhead.

In **Iterative** mode, the MC steps through each setpoint one at a time, processing them one by one.

In **Batched** mode, the MC vectorizes the setpoints such that they are processed in batches.
The size of these batches is automatically calculated but usually dependent on resource constraints; you may have a device which can hold 100 samples but you wish to sweep over 2000 points.

.. note:: The maximum batch size of the settable(s)/gettable(s) should be specified using the `.batch_size` attribute. If not specified infinite size is assumed and all setpoint are passed to the settable(s).

.. tip:: In *Batched* mode it is still possible to perform outer iterative sweeps with an inner batched sweep. This is performed automatically when batched settables (`.batched=True`) are mixed with iterative settables (`.batched=False`). To correctly grid the points in this mode use :meth:`~quantify.measurement.MeasurementControl.setpoints_grid`.

Control mode is detected automatically based on the `.batched` attribute of the settable(s) and gettable(s); this is expanded upon in subsequent sections.

.. note:: All gettables must have the same value for the `.batched` attribute. Only when all gettables have `.batched=True`, settables are allowed to have mixed `.batched` attribute (e.g. `settable_A.batched=True`, `settable_B.batched=False`).


Settables and Gettables
========================================

Experiments typically involve varying some parameters and reading others. In Quantify we encapsulate these concepts as the :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` respectively.
As their name implies, a Settable is a parameter you set values to, and a Gettable is a parameter you get values from.

The interfaces for Settable and Gettable parameters are encapsulated in the :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` helper classes respectively.
We set values to Settables; these values populate an `X`-axis.
Similarly, we get values from Gettables which populate a `Y`-axis.
These classes define a set of mandatory and optional attributes the :class:`~quantify.measurement.MeasurementControl` recognizes and will use as part of the experiment, which are expanded up in the API Reference.

For ease of use, we do not require users to inherit from a Gettable/Settable class, and instead provide contracts in the form of JSON schemas to which these classes must fit (see :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` docs for these schemas).
In addition to using a library which fits these contracts (such as the :class:`~qcodes.instrument.parameter.Parameter` family of classes) we can define our own Settables and Gettables.

.. jupyter-execute::

    t = ManualParameter('time', label='Time', unit='s')

    class WaveGettable:
        def __init__(self):
            self.unit = 'V'
            self.label = 'Amplitude'
            self.name = 'sine'

        def get(self):
            return np.sin(t() / np.pi)

        # optional methods to prepare can be left undefined
        def prepare(self) -> None:
            print("Preparing the WaveGettable for acquisition.")

        def finish(self) -> None:
            print("Finishing WaveGettable to wrap up the experiment.")

    # verify compliance with the Gettable format
    wave_gettable = WaveGettable()
    Gettable(wave_gettable)

.. admonition:: Note: "Grouped" gettable(s) are also allowed.
    :class: dropdown

    Below we create a Gettable which returns two distinct quantities at once:

    .. jupyter-execute::

        t = ManualParameter('time', label='Time', unit='s')

        class DualWave:
            def __init__(self):
                self.unit = ['V', 'V']
                self.label = ['Sine Amplitude', 'Cosine Amplitude']
                self.name = ['sin', 'cos']

            def get(self):
                return np.array([np.sin(t() / np.pi), np.cos(t() / np.pi)])

            # N.B. the optional prepare and finish methods are omitted in this Gettable.

        # verify compliance with the Gettable format
        wave_gettable = DualWave()
        Gettable(wave_gettable)

Depending on which Control Mode the :class:`~quantify.measurement.MeasurementControl` is running in, the interfaces for Settables (their input interface) and Gettables (their output interface) are slightly different.




.. note::

    It is also possible for batched Gettables return an array with length less than then the length of the setpoints, and similarly for the input of the Settables.
    This is often the case when working with resource constrained devices, for example if you have *n* setpoints but your device can load only less than *n* datapoints into memory.
    In this scenario, the MC tracks how many datapoints were actually processed, automatically adjusting the size of the next batch.

    .. admonition:: Example
        :class: dropdown, note

        .. jupyter-execute::

            time = ManualParameter(name='time', label='Time', unit='s', vals=validators.Arrays(), initial_value=np.array([1, 2, 3]))
            signal = Parameter(name='sig_a', label='Signal', unit='V', get_cmd=lambda: np.cos(time()))

            time.batched = True
            time.batch_size = 5
            signal.batched = True
            signal.batch_size = 10

            MC.settables(time)
            MC.gettables(signal)
            MC.setpoints(np.linspace(0, 7, 23))
            dset = MC.run("my experiment")
            dset_grid = dh.to_gridded_dataset(dset)

            dset_grid.y0.plot()



.batched and .batch_size
----------------------------------------

The :py:class:`~quantify.measurement.Gettable` and :py:class:`~quantify.measurement.Settable` objects can have a `bool` property `.batched` (defaults to `False` if not present); and a `int` property `.batch_size`.

Setting the `.batched` property to `True` enables the batch Control Mode in the :class:`~quantify.measurement.MeasurementControl`. In this mode, if present, the `.batch_size` attribute is used to determine the maximum size of a batch of setpoints.

.. admonition:: Heterogeneous batch size and effective batch size
    :class: dropdown, note

    The minimum `.batch_size` among all settables and gettables will determine the (maximum) size of a batch. During execution of a measurement the size of a batch will be reduced if necessary to comply to the setpoints grid and/or total number of setpoints.


.prepare() and .finish()
----------------------------------------

Optionally the :meth:`!.prepare` and :meth:`!.finish` can be added.
These methods can be used to setup and teardown work. For example, arming a piece of hardware with data and then closing a connection upon completion.

The :meth:`!.finish` runs once at the end of an experiment.

For `settables`, :meth:`!.prepare` runs once **before the start of a measurement**.

For batched `gettables`, :meth:`!.prepare` runs **before the measurement of each batch**. For iterative `gettables`, the :meth:`!.prepare` runs before each loop counting towards soft-averages [controlled by :meth:`!MC.soft_avg()` which resets to `1` at the end of each experiment].

Data storage & Analysis
=========================
Along with the produced dataset, every :class:`~qcodes.instrument.parameter.Parameter` attached to QCoDeS :class:`~qcodes.instrument.base.Instrument` in an experiment run through the :class:`~quantify.measurement.MeasurementControl` of Quantify is stored in the `snapshot`_.

This is intended to aid with reproducibility, as settings from a past experiment can easily be reloaded (see :func:`~quantify.utilities.experiment_helpers.load_settings_onto_instrument`) and re-run by anyone.

Data Directory
-----------------

The top level directory in the file system where output is saved to.
This directory can be controlled using the :meth:`~quantify.data.handling.get_datadir` and :meth:`~quantify.data.handling.set_datadir` functions.

We recommend to change the default directory when starting the python kernel (after importing Quantify); and to settle for a single common data directory for all notebooks/experiments within your measurement setup/PC (e.g. *D:\Data*).

Quantify provides utilities to find/search and extract data, which expects all your experiment containers to be located within the same directory (under the corresponding date subdirectory).

Within the data directory experiments are first grouped by date -
all experiments which take place on a certain date will be saved together in a subdirectory in the form ``YYYYmmDD``.

Experiment Container
----------------------------------

Individual experiments are saved to their own subdirectories (of the Data Directory) named based on the :class:`~quantify.data.types.TUID` and the ``<experiment name (if any)>``.

.. note::
    TUID: A Time-based Unique ID is of the form ``YYYYmmDD-HHMMSS-sss-<random 6 character string>`` and these subdirectories' names take the form ``YYYYmmDD-HHMMSS-sss-<random 6 character string><-experiment name (if any)>``.

These subdirectories are termed 'Experiment Containers', typical output being the Dataset in hdf5 format and a JSON format file describing Parameters, Instruments and such.

Furthermore, additional analysis such as fits can also be written to this directory, storing all data in one location.

A data directory with the name 'MyData' thus will look similar to:

.. code-block:: none

    MyData
    └─ 20200708
    │  └─ 20200708-145048-800-60cf37
    │  │  └─ file1.txt
    │  └─ 20200708-145205-042-6d068a-bell_test
    │     └─ dataset.hdf5
    │     └─ snapshot.json
    │     └─ lmfit.png
    └─ 20200710

Dataset
-----------------

The Dataset is implemented with a **specific** convention using the :class:`xarray.Dataset` class.

Quantify arranges data along two types of axes: `X` and `Y`.
In each dataset there will be *n* `X`-type axes and *m* `Y`-type axes. For example, the dataset produced in an experiment where we sweep 2 parameters (settables) and measure 3 other parameters (all 3 returned by a Gettable), we will have *n* = 2 and *m* = 3.
Each `X` axis represents a dimension of the setpoints provided. The `Y` axes represent the output of the Gettable.
Each axis type are numbered ascending from 0 (e.g. :code:`x0`, :code:`x1`, :code:`y0`, :code:`y1`, :code:`y2`), and each stores information described by the :class:`~quantify.measurement.Settable` and
:class:`~quantify.measurement.Gettable` classes, such as titles and units. The Dataset object also stores some further metadata,
such as the :class:`~quantify.data.types.TUID` of the experiment which it was generated from.

For example, consider an experiment varying time and amplitude against a Cosine function.
The resulting dataset will look similar to the following:

.. jupyter-execute::
    :hide-code:

    t = ManualParameter('t', initial_value=1, unit='s', label='Time')
    amp = ManualParameter('amp', initial_value=1, unit='V', label='Amplitude')
    amp.batched = True
    amp.batch_size = 3

    def CosFunc():
        return amp() * np.cos(t())

    sig = Parameter(name='sig', label='Signal level', unit='V', get_cmd=CosFunc)
    sig.batched = True
    sig.batch_size = 6

    MC.verbose(False) # Suppress printing
    MC.settables([amp, t])
    MC.setpoints_grid([np.linspace(-1, 1, 10), np.linspace(0, 10, 100)])
    MC.gettables(sig)
    quantify_dataset = MC.run('my experiment')

.. jupyter-execute::

    # plot the columns of the dataset
    _, axs = plt.subplots(3,1, sharex=True)
    xr.plot.line(quantify_dataset["x0"][:54], label="x0", ax=axs[0], marker=".")
    xr.plot.line(quantify_dataset["x1"][:54], label="x1", ax=axs[1], color="C1", marker=".")
    xr.plot.line(quantify_dataset["y0"][:54], label="y0", ax=axs[2], color="C2", marker=".")
    tuple(ax.legend() for ax in axs)
    # return the dataset
    quantify_dataset

Associating dimensions to coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To support both gridded and non-gridded data, we use :doc:`Xarray <xarray:index>` using only `Data Variables` and `Coordinates` **with a single** `Dimension` (corresponding to the order of the setpoints).

This is necessary as in the non-gridded case the dataset will be a perfect sparse array, usability of which is cumbersome.
A prominent example of non-gridded use-cases can be found :ref:`adaptive_tutorial`.

To allow for some of Xarray's more advanced functionality, such as the in-built graphing or query system we provide a dataset conversion utility :func:`~quantify.data.handling.to_gridded_dataset`.
This function reshapes the data and associates dimensions to the dataset [which can also be used for 1D datasets].

.. jupyter-execute::

    gridded_dset = dh.to_gridded_dataset(quantify_dataset)
    gridded_dset.y0.plot()
    gridded_dset


Snapshot
-----------------

The configuration for each QCoDeS :class:`~qcodes.instrument.base.Instrument` used in this experiment. This information is automatically collected for all Instruments in use.
It is useful for quickly reconstructing a complex set-up or verifying that :class:`~qcodes.instrument.parameter.Parameter` objects are as expected.


Examples
==================================
Below we give several examples of experiment using Settables and Gettables in different control modes.


Iterative control mode
----------------------

Single-float-valued settable(s) and gettable(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Each settable accepts a single float value.
- Gettables return a single float value.

.. admonition:: 1D
    :class: dropdown

    .. jupyter-execute::

        time = ManualParameter(name='time', label='Time', unit='s', vals=validators.Numbers(), initial_value=1)
        signal = Parameter(name='sig_a', label='Signal', unit='V', get_cmd=lambda: np.cos(time()))

        MC.settables(time)
        MC.gettables(signal)
        MC.setpoints(np.linspace(0, 7, 20))
        dset = MC.run("my experiment")
        dset_grid = dh.to_gridded_dataset(dset)

        dset_grid.y0.plot(marker='o')
        dset_grid

.. admonition:: 2D
    :class: dropdown

    .. jupyter-execute::

        time_a = ManualParameter(name='time_a', label='Time A', unit='s', vals=validators.Numbers(), initial_value=1)
        time_b = ManualParameter(name='time_b', label='Time B', unit='s', vals=validators.Numbers(), initial_value=1)
        signal = Parameter(name='sig_a', label='Signal A', unit='V', get_cmd=lambda: np.exp(time_a()) + 0.5 * np.exp(time_b()))

        MC.settables([time_a, time_b])
        MC.gettables(signal)
        MC.setpoints_grid([np.linspace(0, 5, 10), np.linspace(5, 0, 12)])
        dset = MC.run("my experiment")
        dset_grid = dh.to_gridded_dataset(dset)

        dset_grid.y0.plot(cmap="viridis")
        dset_grid

.. admonition:: ND
    :class: dropdown

        For more dimensions you only need to pass more settables and the corresponding setpoints.

.. admonition:: 1D adaptive
    :class: dropdown

    .. jupyter-execute::

        from scipy.optimize import minimize_scalar

        time = ManualParameter(name='time', label='Time', unit='s', vals=validators.Numbers(), initial_value=1)
        signal = Parameter(name='sig_a', label='Signal', unit='V', get_cmd=lambda: np.cos(time()))
        MC.settables(time)
        MC.gettables(signal)
        dset = MC.run_adaptive('1D minimizer', {"adaptive_function": minimize_scalar})

        dset_ad = dh.to_gridded_dataset(dset)
        # add a grey cosine for reference
        x = np.linspace(np.min(dset_ad['x0']), np.max(dset_ad['x0']), 101)
        y = np.cos(x)
        plt.plot(x,y,c='grey', ls='--')
        dset_ad.y0.plot(marker='o')

Single-float-valued settable(s) with multiple float-valued gettable(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- Each settable accepts a single float value.
- Gettables return a 1D array of floats, with each element corresponding to a *different Y dimension*.

We exemplify a 2D case, however there is no limitation on the number of settables.

.. admonition:: 2D
    :class: dropdown

    .. jupyter-execute::

        time_a = ManualParameter(name='time_a', label='Time A', unit='s', vals=validators.Numbers(), initial_value=1)
        time_b = ManualParameter(name='time_b', label='Time B', unit='s', vals=validators.Numbers(), initial_value=1)

        signal = Parameter(name='sig_a', label='Signal A', unit='V', get_cmd=lambda: np.exp(time_a()) + 0.5 * np.exp(time_b()))

        class DualWave:
            def __init__(self):
                self.unit = ['V', 'V']
                self.label = ['Sine Amplitude', 'Cosine Amplitude']
                self.name = ['sin', 'cos']

            def get(self):
                return np.array([np.sin(time_a() * np.pi), np.cos(time_b() * np.pi)])

        dual_wave = DualWave()
        MC.settables([time_a, time_b])
        MC.gettables([signal, dual_wave])
        MC.setpoints_grid([np.linspace(0, 3, 21), np.linspace(4, 0, 20)])
        dset = MC.run("my experiment")
        dset_grid = dh.to_gridded_dataset(dset)

        for yi, cmap in zip(("y0", "y1", "y2"), ("viridis", "inferno", "plasma")):
            dset_grid[yi].plot(cmap=cmap)
            plt.show()
        dset_grid

Batched control mode
--------------------

Float-valued array settable(s) and gettable(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Gettables return a 1D array of float values with each element corresponding to a datapoint *in a single Y dimension*.

.. admonition:: 1D
    :class: dropdown

    - Each settable accepts a 1D array of float values corresponding to all setpoints for a single *X dimension*.

    .. jupyter-execute::

        time = ManualParameter(name='time', label='Time', unit='s', vals=validators.Arrays(), initial_value=np.array([1, 2, 3]))
        signal = Parameter(name='sig_a', label='Signal', unit='V', get_cmd=lambda: np.cos(time()))

        time.batched = True
        signal.batched = True

        MC.settables(time)
        MC.gettables(signal)
        MC.setpoints(np.linspace(0, 7, 20))
        dset = MC.run("my experiment")
        dset_grid = dh.to_gridded_dataset(dset)

        dset_grid.y0.plot(marker='o')
        print(f"\nNOTE: The gettable returns an array:\n\n{signal.get()}")
        dset_grid

.. admonition:: 2D (1D batch with iterative outer loop)
    :class: dropdown

    - One settable (at least) accepts a 1D array of float values corresponding to all setpoints for the corresponding *X dimension*.
    - One settable (at least) accepts a float value corresponding to its *X dimension*. The MC will set the value of each of these iterative settables before each batch.


    .. jupyter-execute::

        time_a = ManualParameter(name='time_a', label='Time A', unit='s', vals=validators.Numbers(), initial_value=1)
        time_b = ManualParameter(name='time_b', label='Time B', unit='s', vals=validators.Arrays(), initial_value=np.array([1, 2, 3]))
        signal = Parameter(name='sig_a', label='Signal A', unit='V', get_cmd=lambda: np.exp(time_a()) + 0.5 * np.exp(time_b()))

        time_b.batched = True
        time_b.batch_size = 12
        signal.batched = True

        MC.settables([time_a, time_b])
        MC.gettables(signal)
        # `setpoints_grid` will take into account the `.batched` attribute
        MC.setpoints_grid([np.linspace(0, 5, 10), np.linspace(4, 0, time_b.batch_size)])
        dset = MC.run("my experiment")
        dset_grid = dh.to_gridded_dataset(dset)

        dset_grid.y0.plot(cmap="viridis")
        dset_grid

Float-valued array settable(s) with multi-return float-valued array gettable(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Each settable accepts a 1D array of float values corresponding to all setpoints for a single *X dimension*.
- Gettables return a 2D array of float values with each row representing a *different Y dimension*, i.e. each column is a datapoint corresponding to each setpoint.

.. admonition:: 1D
    :class: dropdown

    .. jupyter-execute::

        time = ManualParameter(name='time', label='Time', unit='s', vals=validators.Arrays(), initial_value=np.array([1, 2, 3]))

        class DualWave:
            def __init__(self):
                self.unit = ['V', 'V']
                self.label = ['Amplitude W1', 'Amplitude W2']
                self.name = ['sine', 'cosine']
                self.batched = True
                self.batch_size = 100

            def get(self):
                return np.array([np.sin(time() * np.pi), np.cos(time() * np.pi)])

        time.batched = True
        dual_wave = DualWave()

        MC.settables(time)
        MC.gettables(dual_wave)
        MC.setpoints(np.linspace(0, 7, 100))
        dset = MC.run("my experiment")
        dset_grid = dh.to_gridded_dataset(dset)

        _, ax=plt.subplots()
        dset_grid.y0.plot(marker='o', label="y0", ax=ax)
        dset_grid.y1.plot(marker='s', label="y1", ax=ax)
        ax.legend()
