Tutorial 1. Controlling a basic experiment using MeasurementControl
=====================================================================

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

.. include:: set_data_dir.rst


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
-------------------------------

Define a simple model
~~~~~~~~~~~~~~~~~~~~~~

We start by defining a simple model to mock our experiment setup (i.e. emulate physical setup for demonstration purpose).
We will be generating a cosine with some normally distributed noise added on top of it.

.. jupyter-execute::

    from time import sleep

    def cos_func(t, amplitude, frequency, phase, offset):
        """A simple cosine function"""
        return amplitude * np.cos(2 * np.pi * frequency * t + phase) + offset

    # Parameters are created to emulate a system being measured
    # ManualParameter's is a handy class that preserves the QCoDeS' Parameter
    # structure without necessarily having a connection to the physical world
    amp = ManualParameter('amp', initial_value=1, unit='V', label='Amplitude')
    freq = ManualParameter('freq', initial_value=.5, unit='Hz', label='Frequency')
    t = ManualParameter('t', initial_value=1, unit='s', label='Time')
    phi = ManualParameter('phi', initial_value=0, unit='Rad', label='Phase')

    # we add in some noise to make the fitting example later on more interesting
    noise_level = ManualParameter('noise_level', initial_value=0.05, unit='V', label='Noise level')

    acq_delay = ManualParameter('acq_delay', initial_value=.1, unit='s')

    def cosine_model():
        sleep(acq_delay()) # simulates the acquisition delay of an instrument
        return cos_func(t(), amp(), freq(), phase=phi(), offset=0) + np.random.randn() * noise_level()

    # We wrap our function in a Parameter to be able to associate metadata to it, e.g. units
    sig = Parameter(name='sig', label='Signal level', unit='V', get_cmd=cosine_model)


Many experiments involving physical instruments are much slower than the time it takes to simulate our `cosine_model`, that is why we added a `sleep()` controlled by the `acq_delay`.

This allows us to exemplify (later in the tutorial) some of the features of the MC that would be imperceptible otherwise.

.. jupyter-execute::

    # by setting this to a non-zero value we can see the live plotting in action for a slower experiment
    acq_delay(0.0)

Running the 1D experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete experiment is defined in just 4 lines of code. We specify what parameter we want to set, time `t` in this case, what points to measure at, and what parameter to measure.
We then tell the :ref:`MeasurementControl<Measurement Control>` `MC` to run which will return an :class:`~xarray.Dataset` object.

We use the :class:`~quantify.measurement.Settable` and :class:`~quantify.measurement.Gettable` helper classes to ensure our parameters contain the correct attributes.

.. jupyter-execute::

    MC.settables(Settable(t))
    MC.setpoints(np.linspace(0, 5, 50))
    MC.gettables(Gettable(sig))
    dset = MC.run('Cosine test')

.. jupyter-execute::

    plotmon.main_QtPlot

.. jupyter-execute::

    # The dataset has a time-based unique identifier automatically assigned to it
    # The name of the experiment is stored as well
    dset.attrs['tuid'], dset.attrs['name']

The dataset :ref:`dset<DataStorage specification>` is stored as a :class:`~xarray.Dataset` (you can read more about xarray project at http://xarray.pydata.org/).

As shown below, a **Data variable** is assigned to each dimension of the settables and the gettable(s), following a format in which the settable take the form x0, x1, etc. and the gettable(s) the form y0, y1, y2, etc.. You can click on the icons on the right to see the attributes of each variable and the values.

See :ref:`DataStorage specification` in the :ref:`User guide` for details.

.. jupyter-execute::

    dset

We can play with some live plotting options to see how the MC behaves when changing the update interval.

.. jupyter-execute::

    # By default the MC updates the datafile and live plot every 0.1 seconds (and not faster) to reduce overhead.
    MC.update_interval(0.1) # Setting it even to 0.01 creates a dramatic slowdown, try it out!

In order to avoid an experiment being bottlenecked by the `update_interval` we recommend setting it between ~0.1-1.0 s for a comfortable refresh rate and good performance.


.. jupyter-execute::

    MC.settables(Settable(t))
    MC.setpoints(np.linspace(0, 50, 1000))
    MC.gettables(Gettable(sig))
    dset = MC.run('Many points live plot test')


.. jupyter-execute::

    plotmon.main_QtPlot


.. jupyter-execute::

    noise_level(0) #let's disable noise from here on to get prettier figures

Analyzing the experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loading the data
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~xarray.Dataset` contains all the information required to perform basic analysis of the experiment and information on where the data is stored.
We can alternatively load the dataset from disk based on it's :class:`~quantify.data.types.TUID`, a timestamp-based unique identifier. If you do not know the tuid of the experiment you can find the latest tuid containing a certain string in the experiment name using :meth:`~quantify.data.handling.get_latest_tuid`.
See the data storage documentation for more details on the folder structure and files contained in the data directory.

.. jupyter-execute::

    from quantify.data.handling import load_dataset, get_latest_tuid

    # here we look for the latest datafile in the datadirectory named "Cosine test"
    # note that this is not he last dataset but one dataset earlier
    tuid = get_latest_tuid('Cosine test')
    print('tuid: {}'.format(tuid))
    dset = load_dataset(tuid)

    dset

Performing fits and extracting quantities of interest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have used a cosine function to "mock" an experiment, the goal of the experiment is to find the underlying parameters.
We extract these parameters by performing a fit to a model, which coincidentally, is based on the same cosine function.
For fitting we recommend using the lmfit library.  See https://lmfit.github.io/lmfit-py/model.html on how to fit data to a custom model.

.. jupyter-execute::

    import lmfit
    # we create a model based on our function
    mod = lmfit.Model(cos_func)
    # and specify initial guesses for each parameter
    mod.set_param_hint('amplitude', value=.8, vary=True)
    mod.set_param_hint('frequency', value=.4)
    mod.set_param_hint('phase', value=0, vary=False)
    mod.set_param_hint('offset', value=0, vary=False)
    params = mod.make_params()
    # and here we perform the fit.
    fit_res = mod.fit(dset['y0'].values, t=dset['x0'].values, params=params)

    # It is possible to get a quick visualization of our fit using a build-in method of lmfit
    fit_res.plot_fit(show_init=True)


.. jupyter-execute::

    fit_res.params


.. jupyter-execute::

    # And we can print an overview of the fitting results
    print(fit_res.fit_report())


Plotting and saving the results of the analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    # We include some visualization utilities in quantify
    from quantify.visualization.SI_utilities import set_xlabel, set_ylabel


.. jupyter-execute::

    f, ax = plt.subplots()

    ax.plot(dset['x0'], dset['y0'], marker='o', label='Data')
    x_fit = np.linspace(dset['x0'][0], dset['x0'][-1], 1000)
    y_fit = cos_func(t=x_fit, **fit_res.best_values)
    ax.plot(x_fit, y_fit, label='Fit')
    ax.legend()

    set_xlabel(ax, dset['x0'].attrs['long_name'], dset['x0'].attrs['unit'])
    set_ylabel(ax, dset['y0'].attrs['long_name'], dset['y0'].attrs['unit'])
    ax.set_title('{}\n{}'.format(tuid, 'Cosine test'))

Now that we have analyzed our data and created a figure, we probably want to store the results of our analysis.
We will want to store the figure and the results of the fit in the `experiment folder`.


.. jupyter-execute::

    from os.path import join
    from quantify.data.handling import create_exp_folder
    # Creates a new folder if it does not exist already and return the path to it
    # Here we are using this function as a convenient way of retrieving the experiment
    # folder without using an absolute path
    exp_folder = create_exp_folder(dset.tuid, dset.name)


.. jupyter-execute::

    # Save fit results
    lmfit.model.save_modelresult(fit_res, join(exp_folder, 'fit_res.json'))
    # Save figure
    f.savefig(join(exp_folder, 'Cosine fit.png'), dpi=300, bbox_inches='tight')

A 2D Iterative loop
---------------------------------

It is often desired to measure heatmaps (2D grids) of some parameter.
This can be done by specifying two settables.
The setpoints of the grid can be specified in two ways.


Method 1 - a quick grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

    acq_delay(0.0001)
    MC.update_interval(2.0)


.. jupyter-execute::

    times = np.linspace(0, 5, 500)
    amps = np.linspace(-1, 1, 31)

    MC.settables([Settable(t), Settable(amp)])
    # MC takes care of creating a meshgrid
    MC.setpoints_grid([times, amps])
    MC.gettables(Gettable(sig))
    dset = MC.run('2D Cosine test')


.. jupyter-execute::

    plotmon.main_QtPlot


.. jupyter-execute::

    plotmon.secondary_QtPlot


Method 2 - custom tuples in 2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    acq_delay(0.0001)
    MC.update_interval(2.0)


.. jupyter-execute::

    MC.settables([t, amp])
    MC.setpoints(setpoints)
    MC.gettables(sig)
    dset = MC.run('2D radial setpoints')


.. jupyter-execute::

    plotmon.main_QtPlot


.. jupyter-execute::

    plotmon.secondary_QtPlot


.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 1. Controlling a basic experiment using MeasurementControl`

    :jupyter-download:script:`Tutorial 1. Controlling a basic experiment using MeasurementControl`
