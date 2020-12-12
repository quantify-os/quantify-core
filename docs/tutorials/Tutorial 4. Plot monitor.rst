Tutorial 4. Plot monitor
=====================================================================

In this tutorial we dive into the capabilities of the plot monitor.
We will create a fictional device and showcase how the plot monitor can be used. Enjoy!

.. jupyter-execute::

    import random
    import numpy as np
    import time

    from qcodes import Station
    from qcodes.instrument.base import Instrument
    from qcodes.instrument.parameter import Parameter, ManualParameter

    from quantify.measurement import MeasurementControl
    from quantify.visualization.pyqt_plotmon import PlotMonitor_pyqt
    from quantify.data.handling import get_tuids_containing

    # Display any variable or statement on its own line
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"


.. include:: set_data_dir.rst


QCoDeS drivers for our instruments
-----------------------------------

.. jupyter-execute::

    class Device(Instrument):
        def __init__(self, name: str):
            super().__init__(name=name)

            self.add_parameter(
                name="amp_0",
                unit="A",
                parameter_class=ManualParameter,
                initial_value=0)

            self.add_parameter(
                name="amp_1",
                unit="A",
                parameter_class=ManualParameter,
                initial_value=0)

            self.add_parameter(
                name="offset",
                unit="A",
                parameter_class=ManualParameter,
                initial_value=0)

            self.add_parameter(
                name="adc",
                label="ADC input",
                unit="V",
                parameter_class=Parameter,
                get_cmd=self._get_dac_value)

        def _get_dac_value(self):
            s1 = np.exp(-3 * self.amp_0()) * np.sin(self.amp_0() * 2 * np.pi * 3)
            s2 = np.cos(self.amp_1() * 2 * np.pi * 2)
            return s1 + s2 + random.uniform(0, 0.2) + self.offset()


Instantiate the instruments
----------------------------


.. jupyter-execute::

    if "station" not in dir():
        station = Station()

    for instr in list(station.components):
        station.close_and_remove_instrument(instr)

    MC = MeasurementControl('MC')
    station.add_component(MC)
    plotmon = PlotMonitor_pyqt('Plot Monitor')
    station.add_component(plotmon)
    MC.instr_plotmon(plotmon.name)
    device = Device("Device")
    station.add_component(device)

Overview
----------

There are 3 parameters in the :class:`~quantify.visualization.pyqt_plotmon.PlotMonitor_pyqt` that control the datasets being displayed.

Two main parameters determine the datasets being displayed: *tuids* and *tuids_extra*.

.. jupyter-execute::

    plotmon.tuids()
    plotmon.tuids_extra()


The interface is the same for both. The parameters accept a list of tuids or an empty list to reset.

.. jupyter-execute::

    # Example of loading datasets onto the plot
    # plotmon.tuids(["20201124-184709-137-8a5112", "20201124-184716-237-918bee"])
    # plotmon.tuids_extra(["20201124-184722-988-0463d4", "20201124-184729-618-85970f"])

The difference is that the :class:`~quantify.measurement.MeasurementControl` uses `tuids` and overrides them when running measurements.


.. note::

    All the datasets must have matching data variables (settables and gettables).

The third relevant parameter is the *tuids_max_num*. It accepts an integer which determines the maximum number of dataset that will be stored in *tuids* when the :class:`~quantify.measurement.MeasurementControl` is running.

.. jupyter-execute::

    plotmon.tuids_max_num()


.. note::

    This parameter has no effect when setting the *tuids* manually.


Usage examples
---------------

.. jupyter-execute::

    device.offset(0.0)

    n_pnts = 50

    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    n_pnts = 20

    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    n_pnts = 30

    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot

Now the oldest dataset will vanish from the plot:

.. jupyter-execute::

    # Now the oldest dataset will vanish from the plot

    n_pnts = 40

    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot

We can accumulate more datasets on the plot if we want to:

.. jupyter-execute::

    # We can accumulate more datasets on the plot if we want to
    plotmon.tuids_max_num(4)
    n_pnts = 40

    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot

Or we can disable the accumulation and plot a single dataset:

.. jupyter-execute::

    # Or we can disable the accumulation and plot a single dataset
    plotmon.tuids_max_num(1)

    plotmon.main_QtPlot

This can also be reset:

.. jupyter-execute::

    # This can also be reset with
    plotmon.tuids([])

    plotmon.main_QtPlot # The plotting window will vanish, it is supposed to

For now, we will allow two datasets on the plot monitor.

.. jupyter-execute::

    # For now we will allow two datasets on the plot monitor
    plotmon.tuids_max_num(2)

Now let's imagine that something strange is happening with our setup...

.. jupyter-execute::

    # Now let's imagine that something strange is happening with our setup
    device.offset(1.5)

    n_pnts = 40
    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan problem')

    plotmon.main_QtPlot

We would like to compare if the current behavior matches for example what we got a few minutes ago:

.. jupyter-execute::

    # We would like to compare if the current behavior matches for example
    # what we got a few minutes ago

    reference_tuids = sorted(get_tuids_containing("ADC"))[0:2]

    plotmon.tuids_extra(reference_tuids)
    plotmon.main_QtPlot

OK... that cable was not connected in the right place...

.. jupyter-execute::

    # OK... that cable was not connected in the right place...
    device.offset(0.0)

    # Now let's run again our experiments while we compare it to the previous one in realtime

    n_pnts = 30
    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    n_pnts = 40
    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    print("Yey! We have recovered our setup!")
    plotmon.main_QtPlot

We do not need the reference datasets anymore

.. jupyter-execute::

    # We do not need the reference datasets anymore
    plotmon.tuids_extra([])
    plotmon.main_QtPlot


.. jupyter-execute::

    # Note: both plotmon.tuids_extra and plotmon.tuids can be used
    # but keep in mind that MC also uses the plotmon.tuids

    tuids = get_tuids_containing("problem")[0:1]
    tuids
    plotmon.tuids(tuids)

    n_pnts = 40
    MC.settables(device.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(device.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot

When we have 2D plots only the first dataset from `plotmon.tuids` or `plotmon.tuids_extra` will be plotted in the secondary window, in that order of priority.

.. jupyter-execute::

    # When we have 2D plots only the first dataset from plotmon.tuids or plotmon.tuids_extra, in that order of priority.
    # will be plotted in the secondary window

    MC.settables([device.amp_0, device.amp_1])
    MC.setpoints_grid([np.linspace(0, 1, 20), np.linspace(0, 0.5, 15)])
    MC.gettables(device.adc)
    dset = MC.run('ADC scan 2D')
    reference_tuid_2D = dset.attrs["tuid"]

    plotmon.main_QtPlot
    plotmon.secondary_QtPlot

We still have the persistence of the previous dataset on the main window:

.. jupyter-execute::

    MC.settables([device.amp_0, device.amp_1])
    MC.setpoints_grid([np.linspace(0, 1, 20), np.linspace(0, 0.5, 15)])
    MC.gettables(device.adc)
    dset = MC.run('ADC scan 2D')

    plotmon.main_QtPlot
    plotmon.secondary_QtPlot

We can still have a permanent dataset as a reference in the main window:

.. jupyter-execute::

    device.offset(2.03)
    plotmon.tuids_extra([reference_tuid_2D])

    MC.settables([device.amp_0, device.amp_1])
    MC.setpoints_grid([np.linspace(0, 1, 20), np.linspace(0, 0.5, 15)])
    MC.gettables(device.adc)
    dset = MC.run('ADC scan 2D')

    plotmon.main_QtPlot
    plotmon.secondary_QtPlot


But if we want to see the 2D plot we need to reset `plotmon.tuids`.

.. jupyter-execute::

    plotmon.tuids([])
    plotmon.tuids_extra([reference_tuid_2D])

    plotmon.main_QtPlot
    plotmon.secondary_QtPlot


Now your life will never be the same again ;)


.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 4. Plot monitor`

    :jupyter-download:script:`Tutorial 4. Plot monitor`
