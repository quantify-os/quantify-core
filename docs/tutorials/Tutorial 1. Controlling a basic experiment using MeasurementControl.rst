Tutorial 1. Controlling a basic experiment using MeasurementControl
=====================================================================

.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 1. Controlling a basic experiment using MeasurementControl`

    :jupyter-download:script:`Tutorial 1. Controlling a basic experiment using MeasurementControl`


Following this Tutorial requires familiarity with the **core concepts** of Quantify, we **highly recommended** to consult the (short) :ref:`User guide` before proceeding (see Quantify documentation). If you have some difficulties following the tutorial it might be worth reviewing the :ref:`User guide` !

This tutorial covers basic usage of Quantify focusing on running basic experiments using :class:`~quantify.measurement.MeasurementControl`.
The :class:`~quantify.measurement.MeasurementControl` is the main :class:`~qcodes.instrument.base.Instrument` in charge of running any experiment.

It takes care of saving the data in a standardized format as well as live plotting of the data during the experiment.
Quantify makes a distinction between :ref:`Iterative<Control Mode>` measurements and :ref:`Batched<Control Mode>` measurements.

In an :ref:`Iterative<Control Mode>` measurement, the :class:`~quantify.measurement.MeasurementControl` processes each setpoint fully before advancing to the next.

In a :ref:`Batched<Control Mode>` measurement, the :class:`~quantify.measurement.MeasurementControl` processes setpoints in batches, for example triggering 10 samples and then reading those 10 outputs.
This is useful in resource constrained or overhead heavy situations.

Both measurement policies can be 1D, 2D or higher dimensional. Quantify also supports adaptive measurements in which the datapoints are determined during the measurement loop, which are explored in subsequent tutorials.

---

This tutorial is structured as follows.
In the first section we use a 1D Iterative loop to explain the flow of a basic experiment.
We start by setting up a noisy cosine model to serve as our mock setup and then use the MC to measure this.
We then perform basic (manual) analysis on the data from this experiment. We show how to find and load a dataset, perform a basic fit, and store the results.

.. jupyter-execute::

    import lmfit
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from qcodes import ManualParameter, Parameter
    # %matplotlib inline


.. jupyter-execute::

    from quantify.measurement import MeasurementControl
    from quantify.measurement.control import Settable, Gettable
    import quantify.visualization.pyqt_plotmon as pqm
    from quantify.visualization.instrument_monitor import InstrumentMonitor

.. include:: set_data_dir.rst.txt


.. jupyter-execute::

    MC = MeasurementControl('MC')
    # Create the live plotting intrument which handles the graphical interface
    # Two windows will be created, the main will feature 1D plots and any 2D plots will go to the secondary
    plotmon = pqm.PlotMonitor_pyqt('plotmon')
    # Connect the live plotting monitor to the measurement control
    MC.instr_plotmon(plotmon.name)

    # The instrument monitor will give an overview of all parameters of all instruments
    insmon = InstrumentMonitor("Instruments Monitor")
    # By connecting to the MC the parameters will be updated in real-time during an experiment.
    MC.instrument_monitor(insmon.name)


A 1D Iterative loop
-------------------

Define a simple model
~~~~~~~~~~~~~~~~~~~~~~

We start by defining a simple model to mock our experiment setup (i.e. emulate physical setup for demonstration purpose).
We will be generating a cosine with some normally distributed noise added on top of it.


.. include:: cosine_instrument.rst.txt


Many experiments involving physical instruments are much slower than the time it takes to simulate our `cosine_model`, that is why we added a `sleep()` controlled by the `acq_delay`.

This allows us to exemplify (later in the tutorial) some of the features of the MC that would be imperceptible otherwise.

.. jupyter-execute::

    # by setting this to a non-zero value we can see the live plotting in action for a slower experiment
    pars.acq_delay(0.0)

Running the 1D experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete experiment is defined in just 4 lines of code. We specify what parameter we want to set, time `t` in this case, what points to measure at, and what parameter to measure.
We then tell the :ref:`MeasurementControl<Measurement Control>` `MC` to run which will return an :class:`~xarray.Dataset` object.

We use the :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` helper classes to ensure our parameters contain the correct attributes.

.. jupyter-execute::

    MC.settables(pars.t)                     # as a QCoDeS parameter, 't' obeys the JSON schema for a valid Settable and can be passed to the MC directly.
    MC.setpoints(np.linspace(0, 5, 50))
    MC.gettables(pars.sig)                   # as a QCoDeS parameter, 'sig' obeys the JSON schema for a valid Gettable and can be passed to the MC directly.
    dset = MC.run('Cosine test')

.. jupyter-execute::

    plotmon.main_QtPlot

.. jupyter-execute::

    # The dataset has a time-based unique identifier automatically assigned to it
    # The name of the experiment is stored as well
    dset.attrs['tuid'], dset.attrs['name']

The :ref:`dataset<Dataset>` is stored as a :class:`xarray.Dataset` (you can read more about xarray project at http://xarray.pydata.org/).

As shown below, a **Data variable** is assigned to each dimension of the settables and the gettable(s), following a format in which the settable take the form x0, x1, etc. and the gettable(s) the form y0, y1, y2, etc.. You can click on the icons on the right to see the attributes of each variable and the values.

See :ref:`data_storage` in the :ref:`User guide` for details.

.. jupyter-execute::

    dset

We can play with some live plotting options to see how the MC behaves when changing the update interval.

.. jupyter-execute::

    # By default the MC updates the datafile and live plot every 0.1 seconds (and not faster) to reduce overhead.
    MC.update_interval(0.1) # Setting it even to 0.01 creates a dramatic slowdown, try it out!

In order to avoid an experiment being bottlenecked by the `update_interval` we recommend setting it between ~0.1-1.0 s for a comfortable refresh rate and good performance.


.. jupyter-execute::

    MC.settables(pars.t)
    MC.setpoints(np.linspace(0, 50, 1000))
    MC.gettables(pars.sig)
    dset = MC.run('Many points live plot test')


.. jupyter-execute::

    plotmon.main_QtPlot


.. jupyter-execute::

    pars.noise_level(0) #let's disable noise from here on to get prettier figures


Analyzing the experiment
------------------------

Plotting the data and saving the plots for a simple 1D case can be achieve in a few lines using a standard analysis from the :mod:`quantify.analysis.base_analysis` module. In the same module you can find several common analyses that might fit your needs. It also provides a base data-analysis class (:class:`~quantify.analysis.base_analysis.BaseAnalysis`) -- a flexible framework for building custom analyses, which we explore in detail in :ref:`a dedicated tutorial <analysis_framework_tutorial>`.

The :class:`~xarray.Dataset` contains all the information required to perform basic analysis of the experiment and information on where the data is stored.
The analysis loads the dataset from disk based on it's :class:`~quantify.data.types.TUID`, a timestamp-based unique identifier. If you do not know the tuid of the experiment you can find the latest tuid containing a certain string in the experiment name using :meth:`~quantify.data.handling.get_latest_tuid`.
See the :ref:`data_storage` for more details on the folder structure and files contained in the data directory.

.. jupyter-execute::

    from quantify.data.handling import get_latest_tuid
    # here we look for the latest datafile in the data directory named "Cosine test"
    tuid = get_latest_tuid('Cosine test')
    print('tuid: {}'.format(tuid))

.. jupyter-execute::

    from quantify.analysis import base_analysis as ba
    a_obj = ba.Basic1DAnalysis(tuid=tuid).run()
    a_obj.display_figs_mpl()

For guidance on custom analyses, e.g., fitting a model to the data, see :ref:`analysis_framework_tutorial`.

A 2D Iterative loop
-------------------

It is often desired to measure heatmaps (2D grids) of some parameter.
This can be done by specifying two settables.
The setpoints of the grid can be specified in two ways.


Method 1 - a quick grid
~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    pars.acq_delay(0.0001)
    MC.update_interval(2.0)


.. jupyter-execute::

    times = np.linspace(0, 5, 500)
    amps = np.linspace(-1, 1, 31)

    MC.settables([pars.t, pars.amp])
    # MC takes care of creating a meshgrid
    MC.setpoints_grid([times, amps])
    MC.gettables(pars.sig)
    dset = MC.run('2D Cosine test')


.. jupyter-execute::

    plotmon.main_QtPlot


.. jupyter-execute::

    plotmon.secondary_QtPlot


Method 2 - custom tuples in 2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

N.B. it is also possible to do this for higher dimensional loops

.. jupyter-execute::

    r = np.linspace(0, 1.5, 2000)
    dt = np.linspace(0, 1, 2000)

    f = 10

    theta = np.cos(2*np.pi*f*dt)
    def polar_coords(r, theta):

        x = r*np.cos(2*np.pi*theta)
        y = r*np.sin(2*np.pi*theta)
        return x, y

    x, y = polar_coords(r, theta)
    setpoints = np.column_stack([x, y])
    setpoints


.. jupyter-execute::

    pars.acq_delay(0.0001)
    MC.update_interval(2.0)


.. jupyter-execute::

    MC.settables([pars.t, pars.amp])
    MC.setpoints(setpoints)
    MC.gettables(pars.sig)
    dset = MC.run('2D radial setpoints')


.. jupyter-execute::

    plotmon.main_QtPlot


.. jupyter-execute::

    plotmon.secondary_QtPlot
