===============
User guide
===============

Introduction
===============

A :mod:`quantify` experiment typically consists of a data-acquisition loop in which one or more parameters are set and one or more parameters are measured.

Core concepts
====================

The core of quantify can be understood by understanding the following concepts:

- Parameter_
- Instrument_
- `Measurement Control <#Measurement Control>`_
- `Data storage & Analysis <#data-storage-analysis>`_

Parameter
-----------

A parameter represents a state variable of the system.

    - A parameter can be get and/or set able.
    - Contains metadata such as units and labels.
    - Commonly implemented using the QCoDeS :class:`~qcodes.instrument.parameter.Parameter` class.


Instrument
-----------

An Instrument is a container for parameters that typically (but not necesarily) corresponds to a physical piece of hardware.

Instruments provide the following functionality.

- Container for parameters.
- A standardized interface.
- Provide logging of parameters through the :meth:`~qcodes.instrument.base.Instrument.snapshot` method.
- All instruments inherit from the QCoDeS :class:`~qcodes.instrument.base.Instrument` class.



.. note::

    - Add example of instrument
    - Add overview of different kind of instruments (meta-instrument, virtual instrument, etc.)



Measurement Control
----------------------

The :class:`~quantify.measurement.MeasurementControl` is in charge of the data-acquisition loop and is based on the notion that every experiment exists of the following three steps:

1. Initialize (set) some parameter(s),
2. Measure (get) some parameter(s),
3. Store the data.

Quantify provides two helper classes, Settable and Gettable to aid in these steps, which are explored further in later sections of this article.

:class:`~quantify.measurement.MeasurementControl` provides the following functionality

- Enforce standardization of experiments
- Standardized data storage
- Live plotting of the experiment.
- Support *advanced* experiments

    + Software controlled
    + Hardware controlled
    + 1D/2D/nD


Basic example, a 1D soft-loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running an experiment is simple!
Simply define what parameters to set, and get, and what points to loop over.

In the example below we want to set frequencies on a microwave source and acquire the signal from the pulsar readout module.

.. code-block:: python

    MC.settables(mw_source1.freq)               # We want to set the frequency of a microwave source
    MC.setpoints(np.arange(5e9, 5.2e9, 100e3))  # Scan around 5.1 GHz
    MC.gettables(pulsar_QRM.signal)             # acquire the signal from the pulsar AQM
    dataset = MC.run(name='Frequency sweep')    # Start the experiment


The MeasurementControl can also be used to perform more advanced experiments such as 2D scans, pulse-sequences where the hardware is in control of the acquisition loop, or adaptive experiments in which it is not known what data points to acquire in advance.
Take a look at "nonexistent_example_notebook" for a tutorial on the MeasurementControl.


.. note::

    - Add example 2D measurement
    - Add example of adaptive loop
    - Explain difference between hard and soft-loop.


Settable and Gettable
----------------------

The interfaces for Settable and Gettable parameters are encapsulated in the :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` helper classes respectively.
We set values to Settables; these values populate an x-axis. Similarly, we get values from Gettables which populate a y-axis.
These classes define a set of mandatory and optional attributes the MeasurementControl will use as part of the experiment, which are expanded up in the API Reference.

For ease of use, we do not require users to inherit from a Gettable/Settable class, and instead provide contracts in the form of JSON schemas to which these classes must fit.
In addition to using a library which fits these contracts (such as the QCodes.Parameter family of classes) we can define our own Settables and Gettables.
Below we create a Gettable which returns values in two dimensions, one Sine wave and a Cosine wave, using a QCodes Settable:

.. jupyter-execute::

    import numpy as np
    from qcodes import ManualParameter


    t = ManualParameter('time', label='Time', unit='s')

    class DualWave:
        def __init__(self):
            self.unit = 'A'
            self.label = 'Amplitude'
            self.name = 'wave'

        def get(self):
            return np.array([np.sin(t() / np.pi), np.cos(t() / np.pi)])


.soft, .prepare() and .finish()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MeasurementControl checks for 3 other optional properties on parameters, the `soft` attribute and the `prepare()` and `finish()` methods.
`soft` declares whether this parameter is controlled by the MeasurementControl directly or manages itself, and typically delineates between
data originating in software (such as a Sine function in Python) or hardware (such as an AWG). It defaults to `True` (ie, is software controlled).

The `prepare()` and `finish()` methods are useful for performing work before each iteration of the measurement loop and once after completion.
For example, arming a piece of hardware with data and then closing a connection upon completion. The `prepare()` method for hardware
parameter optionally accepts a list of floats as a parameter. For example

.. jupyter-execute::

    class SinGenerator:
        def __init__(self):
            self.unit ='A'
            self.label = 'Amplitude'
            self.name = 'awg'
            self.soft = False
            self.data = None

        def prepare(setpoints):
            self.arm(setpoints)

        def arm(data):
            # send the data to the device etc.
            pass

        def trigger():
            # processes and returns data etc.
            return []

        def get():
            np.array([np.sin(self.trigger() / np.pi)])

        def shutdown():
            # power off the device etc.
            pass

        def finish():
            self.shutdown()

Data storage & Analysis
--------------------------

Folder structure
====================
