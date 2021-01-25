===============
User guide
===============

Introduction
===============

A :mod:`Quantify` experiment typically consists of a data-acquisition loop in which one or more parameters are set and one or more parameters are measured.

The core of Quantify can be understood by understanding the following concepts:

- `Instruments and Parameters`_
- `Measurement Control`_
- `Settables and Gettables`_
- `Data storage & Analysis`_



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
- Are shown by default in the :class:`~quantify.visualization.instrument_monitor.InstrumentMonitor`


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


Basic example, a 1D Iterative measurement loop
------------------------------------------------

Running an experiment is simple!
Simply define what parameters to set, and get, and what points to loop over.

In the example below we want to set frequencies on a microwave source and acquire the signal from the pulsar readout module.

.. code-block:: python

    MC.settables(mw_source1.freq)               # We want to set the frequency of a microwave source
    MC.setpoints(np.arange(5e9, 5.2e9, 100e3))  # Scan around 5.1 GHz
    MC.gettables(pulsar_QRM.signal)             # acquire the signal from the pulsar QRM
    dataset = MC.run(name='Frequency sweep')    # Start the experiment


The :class:`~quantify.control.MeasurementControl` can also be used to perform more advanced experiments such as 2D scans, pulse-sequences where the hardware is in control of the acquisition loop, or adaptive experiments in which it is not known what data points to acquire in advance, they are determined dynamically during the experiment.
Take a look at some of the tutorial notebooks for more in-depth examples on usage and application.

Control Mode
-----------------

A very important aspect in the usage of the MeasurementControl is the Control Mode, which specifies whether the setpoints are processed iteratively or in batches.
Batched mode can be used to deal with constraints imposed by (hardware) resources or to reduce overhead.

In *Iterative* mode, the MC steps through each setpoint one at a time, processing them one by one.

In *Batched* mode, the MC vectorises the setpoints such that they are processed in batches.
The size of these batches is automatically calculated but usually dependent on resource constraints; you may have a device which can hold 2000 samples but wish to sweep over 40000 points.

Control mode is detected automatically based on the `.batched` attribute of the :class:`~quantify.measurement.Gettable`; this is expanded upon in subsequent sections.

.. note:: Every Settable and Gettable must have the same Control Mode.


Settables and Gettables
========================================

Experiments typically involve varying some parameters and reading others. In Quantify we encapsulate these concepts as the :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` respectively.
As their name implies, a Settable is a parameter you set values to, and a Gettable is a parameter you get values from.

The interfaces for Settable and Gettable parameters are encapsulated in the :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` helper classes respectively.
We set values to Settables; these values populate an x-axis.
Similarly, we get values from Gettables which populate a y-axis.
These classes define a set of mandatory and optional attributes the MeasurementControl recognizes and will use as part of the experiment, which are expanded up in the API Reference.

Depending on which Control Mode the MeasurementControl is running in, the interfaces for Settables (their input) and Gettables (their output) are slightly different:

**Iterative:**

- Each settable accepts a single float value.
- Gettables return a single float value, **OR**
- Gettables return a 1D array of floats, with each element corresponding to a *different y dimension*.

**Batched:**

- Each settable accepts a 1D array of float values corresponding to all setpoints for a single *X dimension*.
- Gettables return a 1D array of float values with each element corresponding to the datapoints *in that Y dimension*, **OR**
- Gettables return a 2D array of float values with each row representing a *different Y dimension* with the above structure, i.e. each column is a datapoint corresponding to each setpoint.

.. note::
    It is also possible for Batched Gettables to return a partial array with length less than the input. This is helpful when working with resource constrained devices,
    for example if you have *n* setpoints but your device can load only less than *n* datapoints into memory. In this scenario, the MC tracks how many datapoints were actually
    processed, automatically adjusting the size of the next batch.

For ease of use, we do not require users to inherit from a Gettable/Settable class, and instead provide contracts in the form of JSON schemas to which these classes must fit.
In addition to using a library which fits these contracts (such as the QCodes.Parameter family of classes) we can define our own Settables and Gettables.
Below we create a Gettable which returns values in two dimensions, one Sine wave and a Cosine wave:

.. jupyter-execute::
    :hide-code:

    from pathlib import Path
    from os.path import join
    from quantify.data.handling import set_datadir
    set_datadir(join(Path.home(), 'quantify-data'))

.. jupyter-execute::

    import numpy as np
    from qcodes import ManualParameter

    t = ManualParameter('time', label='Time', unit='s')

    class DualWave:
        def __init__(self):
            self.unit = ['V', 'V']
            self.label = ['Amplitude', 'Amplitude']
            self.name = ['sine', 'cosine']

        def get(self):
            return np.array([np.sin(t() / np.pi), np.cos(t() / np.pi)])

        def prepare(self) -> None:
            pass

        def finish(self) -> None:
            pass


.batched, .prepare() and .finish()
----------------------------------------

The :py:class:`~quantify.measurement.Gettable` and :py:class:`~quantify.measurement.Settable` class have a `bool` property `batched (default=False)`.
Setting the `batched` property to `True` enables the batch Control Mode in the MeasurementControl.

.. note:: Note that all :py:class:`~quantify.measurement.Gettable` and :py:class:`~quantify.measurement.Settable` entities must have the same Control Mode.

Optionally the :meth:`!prepare` and :meth:`!finish` can be added and are run before and after each MeasurementControl loop.
These methods can be used to setup and teardown work. For example, arming a piece of hardware with data and then closing a connection upon completion.

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

The Dataset is implemented using the :class:`xarray.Dataset` class.

Quantify arranges data along two types of axes: :code:`X` and :code:`Y`.
In each dataset there will be *n* :code:`X` axes and *m* :code:`Y` axes. For example, the dataset produced in an experiment where we sweep 2 parameters (settables) and measure 3 other parameters (all 3 returned by a Gettable), we will have *n* = 2 and *m* = 3.
Each :code:`X` axis represents a dimension of the setpoints provided. The :code:`Y` axes represent the output of the Gettable.
Each axis type are numbered ascending from 0 (e.g. :code:`x0`, :code:`x1`, :code:`y0`, :code:`y1`, :code:`y2`), and each stores information described by the :class:`~quantify.measurement.Settable` and
:class:`~quantify.measurement.Gettable` classes, such as titles and units. The Dataset object also stores some further metadata,
such as the :class:`~quantify.data.types.TUID` of the experiment which it was generated from.

For example, consider an experiment varying time and amplitude against a Cosine function.
The resulting dataset will look similar to the following:

.. jupyter-execute::
    :hide-code:

    from qcodes import ManualParameter, Parameter
    from quantify.measurement.control import MeasurementControl
    import numpy as np

    t = ManualParameter('t', initial_value=1, unit='s', label='Time')
    amp = ManualParameter('amp', initial_value=1, unit='V', label='Amplitude')
    def CosFunc():
        return amp() * np.cos(2 * np.pi * 1e6 * t())

    sig = Parameter(name='sig', label='Signal level', unit='V', get_cmd=CosFunc)

    MC = MeasurementControl('MC')
    MC.verbose(False) # Suppress printing
    MC.settables([t, amp])
    MC.setpoints_grid([np.linspace(0, 5, 20), np.linspace(-1, 1, 5)])
    MC.gettables(sig)
    MC.run('my experiment')


.. note:: To support both gridded and non-gridded data, we use :doc:`Xarray <xarray:index>` using only `datavariables` **without** any `coordinates`  or `dimensions`. This is necessary as in the non-gridded case the dataset will be a perfect sparse array, usability of which is cumbersome. This does mean that some of Xarray's more advanced functionality, such as the in-built graphing or query system, are unavailable without further processing.


Snapshot
-----------------

The configuration for each QCoDeS :class:`~qcodes.instrument.base.Instrument` used in this experiment. This information is automatically collected for all Instruments in use.
It is useful for quickly reconstructing a complex set-up or verifying that :class:`~qcodes.instrument.parameter.Parameter` objects are as expected.
