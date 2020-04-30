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

1. Set some parameter(s),
2. Measure (get) some parameter(s),
3. Store the data.

:class:`~quantify.measurement.MeasurementControl` provides the following functionality

- Enforce standardization of experiments
- Standardized data storage
- live plotting of the expermiment.
- Support *advanced* experiments

    + Software controlled
    + Hardware controlled
    + 1D/2D/nD
    + Adaptive loop

Basic example, a 1D soft-loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running an experiment is simple!
Simply define what parameters to set, and get, and what points to loop over.

In the example below we want to set frequencies on a microwave source and acquire the signal from the pulsar acquisition module.

.. code-block:: python

    MC.set_setpars(mw_source1.freq)                 # We want to set the frequency of a microwave source
    MC.set_setpoints(np.arange(5e9, 5.2e9, 100e3))  # Scan around 5.1 GHz
    MC.set_getpars(pulsar_AQM.signal)               # acquire the signal from the pulsar AQM
    dataset = MC.run(name='Frequency sweep')        # Start the experiment


The MeasurementControl can also be used to perform more advanced experiments such as 2D scans, pulse-sequences where the hardware is in control of the acquisition loop, or adaptive experiments in which it is not known what data points to acquire in advance.
Take a look at "nonexistent_example_notebook" for a tutorial on the MeasurementControl.


.. note::

    - Add example 2D measurement
    - Add example of adaptive loop
    - Explain difference between hard and soft-loop.



Data storage & Analysis
--------------------------

Folder structure
====================
