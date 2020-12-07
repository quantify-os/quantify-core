Tutorial 2. Advanced capabilities of the MeasurementControl
============================================================

Following this Tutorial requires familiarity with the **core concepts** of Quantify, we **highly recommended** to consult the (short) :ref:`User guide` before proceeding (see Quantify documentation). If you have some difficulties following the tutorial it might be worth reviewing the :ref:`User guide`!

We **highly recommended** to begin with :ref:`Tutorial 1. Controlling a basic experiment using MeasurementControl` before proceeding.

In this tutorial, we will explore the more advanced features of Quantify. By the end of this tutorial, we will have covered:

- Using hardware to drive experiments
- Software averaging
- Interrupting an experiment

.. jupyter-execute::

    import time
    import random

    import numpy as np
    import xarray as xr
    import scipy
    from qcodes import ManualParameter, Parameter
    # %matplotlib inline
    from quantify.measurement.control import MeasurementControl
    import quantify.visualization.pyqt_plotmon as pqm
    from quantify.visualization.instrument_monitor import InstrumentMonitor


.. include:: set_data_dir.rst


.. jupyter-execute::

    MC = MeasurementControl('MC')
    plotmon = pqm.PlotMonitor_pyqt('plotmon_MC')
    MC.instr_plotmon(plotmon.name)
    insmon = InstrumentMonitor("Instruments Monitor")
    MC.instrument_monitor(insmon.name)



A 1D Batched loop: Resonator Spectroscopy
------------------------------------------------------------

Defining a simple model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we want to find the resonance of some device. We expect to find it's resonance somewhere in the low 6GHz range, but manufacturing imperfections makes it impossible to know exactly without inspection.

We first create `freq`: a :ref:`Settable<Settable>` with a :class:`~qcodes.instrument.parameter.Parameter` to represent the frequency of the signal probing the resonator, followed by a custom :ref:`Gettable<Gettable>` to mock (i.e. emulate) the resonating material.
The Resonator will return a Lorentzian shape centered on the resonant frequency. Our :ref:`Gettable<Gettable>` will read the setpoints from `freq`, in this case a 1D array.

.. note:: The `Resonator` :ref:`Gettable<Gettable>` has a new field `batched` set to `True`. This property informs the :class:`~quantify.measurement.MeasurementControl` that it will not be in charge of iterating over the setpoints, instead the `Resonator` manages its own data acquisition.


.. jupyter-execute::

    # Note that in an actual experimental setup `freq` will be a QCoDeS parameter
    # contained in a QCoDeS Instrument
    freq = ManualParameter(name='frequency', unit='Hz', label='Frequency')

    # model of the frequency response
    def lorenz(amplitude, fwhm, x, x_0):
        return (amplitude * ((fwhm / 2.) ** 2) / ((x - x_0) ** 2 + (fwhm / 2.) ** 2))

    class Resonator:
        def __init__(self):
            self.name = 'resonator'
            self.unit = 'V'
            self.label = 'Amplitude'
            self.batched = True

            # variables specific to the emulated material
            self.test_resonance = 6.0001048e9 # in Hz
            self.test_width = 300 # FWHM in Hz

        def get(self):
            # Emulation of the frequency response
            return 1-np.array(list(map(lambda x: lorenz(1, self.test_width, x, self.test_resonance), freq())))


Running the experiment
~~~~~~~~~~~~~~~~~~~~~~~~

Just like our Iterative 1D loop, our complete experiment is expressed in just four lines of code.

The main difference is defining the `batched` property of our :ref:`Gettable<Gettable>` to `True`.
The :class:`~quantify.measurement.MeasurementControl` will detect these settings and run in the appropriate mode.


.. jupyter-execute::

    # At this point the `freq` parameter is empty
    print(freq())


.. jupyter-execute::

    MC.settables(freq)
    MC.setpoints(np.arange(6.0001e9, 6.00011e9, 5))
    MC.gettables(Resonator())
    dset = MC.run()


.. jupyter-execute::

    plotmon.main_QtPlot

As expected, we find a Lorentzian spike in the readout at the resonant frequency, finding the peak of which is trivial.


Software Averaging: T1 Experiment
----------------------------------

In many cases it is desirable to run an experiment many times and average the result, such as when filtering noise on instruments or measuring probability.
For this purpose, the :class:`~quantify.measurement.MeasurementControl` provides the `soft_avg` parameter.
If set to *x*, the experiment will run *x* times whilst performing a running average over each setpoint.

In this example, we want to find the relaxation time (aka T1) of a Qubit. As before, we define a :ref:`Settable<Settable>` and :ref:`Gettable<Gettable>`, representing the varying timescales we will probe through and a mock Qubit emulated in software.
The mock Qubit returns the expected decay sweep but with a small amount of noise (simulating the variable qubit characteristics). We set the qubit's T1 to 60 ms - obviously in a real experiment we would be trying to determine this, but for this illustration purposes in this tutorial we set it to a known value to verify our fit later on.

Note that in this example MC is still running in Batched mode.


.. jupyter-execute::

    MC.soft_avg(1)


.. jupyter-execute::

    # T1 experiment decay model
    def decay(t, tau):
        return np.exp(-t/tau)

    time_par = ManualParameter(name='time', unit='s', label='Measurement Time')

    class MockQubit:
        def __init__(self):
            self.name = 'qubit'
            self.unit = '%'
            self.label = 'High V'
            self.batched = True

            self.delay = 0.01 # sleep time in secs
            self.test_relaxation_time = 60e-6

        def get(self):
            time.sleep(self.delay) # adds a delay to be able to appreciate the data aquisition
            return np.array(list(map(lambda x: decay(x, self.test_relaxation_time) + random.uniform(-0.1, 0.1), time_par())))


We will then sweep through 0 to 300ms, getting our data from the mock Qubit. Let's first observe what a single run looks like:


.. jupyter-execute::

    MC.settables(time_par)
    MC.setpoints(np.linspace(0.0, 300.0e-6, 300))
    MC.gettables(MockQubit())
    MC.run('noisy')
    plotmon.main_QtPlot

Alas, the noise in the signal has made this result unusable! Let's set the `soft_avg` parameter of the :class:`~quantify.measurement.MeasurementControl` to 100, averaging the results and hopefully filtering out the noise.

.. jupyter-execute::

    MC.soft_avg(100)
    dset = MC.run('averaged')
    plotmon.main_QtPlot

Success! We now have a smooth decay curve based on the characteristics of our qubit. All that remains is to run a fit against the expected values and we can solve for T1.


.. jupyter-execute::

    from lmfit import Model

    model = Model(decay, independent_vars=['t'])
    fit_res = model.fit(dset['y0'].values, t=dset['x0'].values, tau=1)

    fit_res.plot_fit(show_init=True)
    fit_res.values


Interrupting
-------------

Sometimes experiments unfortunately do not go as planned and it is desirable to interrupt and restart them with new parameters. In the following example, we have a long running experiment where our Gettable is taking a long time to return data (maybe due to misconfiguration).
Rather than waiting for this experiment to complete, instead we can interrupt any :class:`~quantify.measurement.MeasurementControl` loop using the standard interrupt signal.
In a terminal environment this is usually achieved with a ``ctrl`` + ``c`` press on the keyboard or equivalent, whilst in a Jupyter environment interrupting the kernel will cause the same result.

When the :class:`~quantify.measurement.MeasurementControl` is interrupted, it will perform a final save of the data it has gathered, call the `finish()` method on Settables & Gettables (if it exists) and return the partially completed dataset.

.. note::
    The exact means of triggering an interrupt will differ depending on your platform and environment; the important part is to cause a `KeyboardInterrupt` exception to be raised in the Python process.

.. warning::
    Pressing ``ctrl`` + ``c`` more than once might result in the `KeyboardInterrupt` not being properly handled and corrupt the dataset!


.. jupyter-execute::

    class SlowGettable:
        def __init__(self):
            self.name = 'slow'
            self.label = 'Amplitude'
            self.unit = 'V'

        def get(self):
            time.sleep(0.5)
            return time_par()

    MC.settables(time_par)
    MC.setpoints(np.arange(20))
    MC.gettables(SlowGettable())
    # Try interrupting me!
    dset = MC.run('slow')


.. jupyter-execute::

    plotmon.main_QtPlot



.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 2. Advanced capabilities of the MeasurementControl`

    :jupyter-download:script:`Tutorial 2. Advanced capabilities of the MeasurementControl`
