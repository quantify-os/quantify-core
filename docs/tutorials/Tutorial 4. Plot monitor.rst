Tutorial 1. Controlling a basic experiment using MeasurementControl
=====================================================================


.. jupyter-execute::

    import random
    import numpy as np
    import time

    from qcodes import Station
    from qcodes import validators as vals
    from qcodes.instrument.base import Instrument
    from qcodes.instrument.parameter import Parameter, ManualParameter

    import quantify.measurement as mc
    from quantify.visualization.pyqt_plotmon import PlotMonitor_pyqt
    from quantify.visualization.instrument_monitor import InstrumentMonitor
    from quantify.data.handling import get_tuids_containing


.. include:: set_data_dir.rst


QCoDeS drivers for our instruments
-----------------------------------

.. jupyter-execute::

    # A source of pseudo-randomness
    random.seed(79)

    class CurrentSource(Instrument):
        """
        A QCoDeS intrument class that mimics the interaction with a physical (or virtual) intrument.
        """

        def __init__(self, name: str, physical_world: dict):
            """
            Creates an instance of this intrument

            Parameters
            ----------
            name : str
                name
            """
            # Run __init__ of parent class
            super().__init__(name=name)

            # Access to the emulated physical world (tutorial puposes only)
            self.physical_world = physical_world

            # internal driver variables
            self._amp_0 = 0
            self._amp_1 = 0

            # Parameters are attributes that are logged along with the dataset
            # and in realtime in the intrument monitor

            self.add_parameter(
                name="amp_0",
                label="Amplitude pin 0",
                unit="A",
                docstring="Controls the current on pin 0",
                parameter_class=Parameter,
                vals=vals.Numbers(min_value=0, max_value=1),
                # NB: usually required when there is some communication with a physical intrument
                # if we only want to store a value intended to be logged, then use ManualParameter instead
                set_cmd=self._set_amp_0,  # function to be executed when setting this parameter
                get_cmd=self._get_amp_0,  # function to be executed when getting this parameter
            )

            self.add_parameter(
                name="amp_1",
                label="Amplitude pin 1",
                unit="A",
                docstring="Controls the current on pin 1",
                parameter_class=Parameter,
                vals=vals.Numbers(min_value=0, max_value=1),
                # NB: usually required when there is some communication with a physical intrument
                # if we only want to store a value intended to be logged, then use ManualParameter instead
                set_cmd=self._set_amp_1,  # function to be executed when setting this parameter
                get_cmd=self._get_amp_1,  # function to be executed when getting this parameter
            )

        def _set_amp_0(self, val):
            # Mimic communication delay
            self._amp_0 = val
            # for tutorial purpose only
            self.physical_world["amp_0"] = val

        def _get_amp_0(self):
            return self._amp_0

        def _set_amp_1(self, val):
            self._amp_1 = val
            # for tutorial purpose only
            self.physical_world["amp_1"] = val

        def _get_amp_1(self):
            return self._amp_1


    class ADC(Instrument):
        """
        A QCoDeS intrument class that mimics the interaction with a physical (or virtual) intrument.
        """

        def __init__(self, name: str, physical_world: dict):
            """
            Creates an instance of this intrument

            Parameters
            ----------
            name : str
                name
            """
            # Run __init__ of parent class
            super().__init__(name=name)

            # Access to the emulated physical world (tutorial puposes only)
            self.physical_world = physical_world

            # Mimic a communication delay
            self._com_delay = 0.005

            # Parameters are attributes that are logged along with the dataset
            # and in realtime in the intrument monitor

            self.add_parameter(
                name="adc",
                label="ADC voltage",
                unit="V",
                docstring="Returns the voltage at the ADC input",
                parameter_class=Parameter,
                vals=vals.Numbers(min_value=0, max_value=1),
                get_cmd=self._get_dac_value,
            )

        def _get_dac_value(self):
            time.sleep(self._com_delay)
            # Mimic a voltage ouput
            amp_0 = self.physical_world["amp_0"]
            amp_1 = self.physical_world["amp_1"]
            offset = self.physical_world["offset"]

            return np.exp(-3 * amp_0) * np.sin(amp_0 * 2 * np.pi * 3) + np.cos(amp_1 * 2 * np.pi * 2) + random.uniform(0, 0.2) + offset

Instantiate the instruments
----------------------------


.. jupyter-execute::

    # Create a QCoDeS intruments Station (if not present)
    if "station" not in dir():
        station = Station()

    for instr in list(station.components):
        # This avoids exceptions when re-running this cell (in same python kernel)
        station.close_and_remove_instrument(instr)

    # Instantiate a measurement control
    MC = mc.MeasurementControl('MC')
    station.add_component(MC)

    # The instrument monitor will give an overview of all parameters of all instruments
    insmon = InstrumentMonitor("Instruments Monitor")
    station.add_component(insmon)

    # By connecting to the MC the parameters will be updated in real-time during an experiment.
    MC.instrument_monitor(insmon.name)

    # Instantiate a plot monitor
    plotmon = PlotMonitor_pyqt('Plot Monitor')
    station.add_component(plotmon)

    # Connect plotmon to measurement control
    MC.instr_plotmon(plotmon.name)

    # FOR TUTORIAL PURPOSES ONLY!
    # Global variables that mimics some physics unrelated to the intruments
    physical_world = dict(amp_0 = 0, amp_1 = 0, offset=0.1)

    # Instantiate our mesurement intruments
    adc_instr = ADC("ADC", physical_world)
    station.add_component(adc_instr)

    # Instantiate our mesurement intruments
    source = CurrentSource("Source", physical_world)
    station.add_component(source)


Run some experiment
----------------------

.. jupyter-execute::

    physical_world["offset"] = 0.0

    n_pnts = 50

    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    n_pnts = 20

    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    n_pnts = 30

    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    # Now the oldest dataset will vanish from the plot

    n_pnts = 40

    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    # We can accumulate more datasets on the plot if we want to
    plotmon.tuids_max_num(4)
    n_pnts = 40

    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    # Or we can disable the accumulation and plot a single dataset
    plotmon.tuids_max_num(1)

    plotmon.main_QtPlot


.. jupyter-execute::

    # This can also be reset with
    plotmon.tuids([])

    plotmon.main_QtPlot # The plotting window will vanish, it is supposed to


.. jupyter-execute::

    # For now we will allow two dataseta on the plot monitor
    plotmon.tuids_max_num(2)


.. jupyter-execute::

    # Now let's imagine that something strange is happanening with out setup
    physical_world["offset"] = 1.5

    n_pnts = 40
    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan problem')

    plotmon.main_QtPlot


.. jupyter-execute::

    # We would like to compare if the current behavior matches for example
    # what we got a few minutes ago

    reference_tuids = sorted(get_tuids_containing("ADC"))[0:2]

    plotmon.tuids_extra(reference_tuids)
    plotmon.main_QtPlot


.. jupyter-execute::

    # OK... that cable was not connected in the right place...
    physical_world["offset"] = 0.0

    # Now let's run again our experiments while we compare it to the previous one in realtime

    n_pnts = 30
    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    n_pnts = 40
    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    print("Yey! We have recovered our setup!")
    plotmon.main_QtPlot


.. jupyter-execute::

    # We do not need the reference datasets anymore
    plotmon.tuids_extra([])
    plotmon.main_QtPlot


.. jupyter-execute::


    # Note: both plotmon.tuids_extra and plotmon.tuids can be used
    # but keep in mind that MC also uses the plotmon.tuids

    tuids = get_tuids_containing("problem")[0:1]
    print(tuids)
    plotmon.tuids(tuids)

    n_pnts = 40
    MC.settables(source.amp_0)
    MC.setpoints(np.linspace(0, 1, n_pnts))
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan')

    plotmon.main_QtPlot


.. jupyter-execute::

    # When we have 2D plots only the first dataset from plotmon.tuids or plotmon.tuids_extra
    # will be plotted in the secondary window

    MC.settables([source.amp_0, source.amp_1])
    MC.setpoints_grid([np.linspace(0, 1, 20), np.linspace(0, 0.5, 15)])
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan 2D')
    plotmon.main_QtPlot

.. jupyter-execute::

    plotmon.secondary_QtPlot


.. jupyter-execute::

    MC.settables([source.amp_0, source.amp_1])
    MC.setpoints_grid([np.linspace(0, 1, 20), np.linspace(0, 0.5, 15)])
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan 2D')

    plotmon.main_QtPlot


.. jupyter-execute::


    plotmon.secondary_QtPlot


.. jupyter-execute::

    physical_world["offset"] = 2.03
    plotmon.tuids_extra([dset.attrs["tuid"]])

    MC.settables([source.amp_0, source.amp_1])
    MC.setpoints_grid([np.linspace(0, 1, 20), np.linspace(0, 0.5, 15)])
    MC.gettables(adc_instr.adc)
    dset = MC.run('ADC scan 2D')

    plotmon.main_QtPlot

.. jupyter-execute::


    plotmon.secondary_QtPlot


.. jupyter-execute::

    plotmon.tuids([])
    plotmon.tuids_extra(get_tuids_containing("2D")[0:1])

    plotmon.secondary_QtPlot
