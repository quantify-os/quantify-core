import time
import json
from os.path import join
import concurrent.futures
from threading import Event

import numpy as np
from qcodes import Instrument
from qcodes import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils.helpers import NumpyJSONEncoder
from quantify.data.handling import initialize_dataset, create_exp_folder, snapshot
from quantify.measurement.types import Settable, Gettable, is_software_controlled
from quantify.utilities.general import KeyboardFinish


class MeasurementControl(Instrument):
    """
    Instrument responsible for controling the data acquisition loop.

    MeasurementControl (MC) is based on the notion that every experiment consists of the following step:

        1. Set some parameter(s)            (settable_pars)
        2. Measure some other parameter(s)  (gettable_pars)
        3. Store the data.

    Example:

        .. code-block:: python

            MC.settables(mw_source1.freq)
            MC.setpoints(np.arange(5e9, 5.2e9, 100e3))
            MC.gettables(pulsar_AQM.signal)
            dataset = MC.run(name='Frequency sweep')


    MC exists to enforce structure on experiments. Enforcing this structure allows:

        - Standardization of data storage.
        - Providing basic real-time visualization.

    MC imposes minimal constraints and allows:

    - Soft loops, experiments in which MC controlled acquisition loop.
    - Hard loops, experiments in which MC is not in control of acquisition.
    - Adaptive loops, setpoints are determined based on measured values.

    """

    def __init__(self, name: str):
        """
        Creates an instance of the Measurement Control.

        Args:
            name (str): name
        """
        super().__init__(name=name)

        # Parameters are attributes that we include in logging and intend the user to change.

        self.add_parameter(
            "verbose",
            docstring="If set to True, prints to std_out during experiments.",
            parameter_class=ManualParameter,
            vals=vals.Bool(),
            initial_value=True,
        )

        self.add_parameter(
            "on_progress_callback",
            vals=vals.Callable(),
            docstring="A callback to communicate progress. This should be a "
            "Callable accepting ints between 0 and 100 indicating percdone.",
            parameter_class=ManualParameter,
            initial_value=None,
        )

        self.add_parameter(
            "soft_avg",
            label="Number of soft averages",
            parameter_class=ManualParameter,
            vals=vals.Ints(1, int(1e8)),
            initial_value=1,
        )

        self.add_parameter(
            'instr_plotmon',
            docstring='Instrument responsible for live plotting. '
            'Can be set to str(None) to disable live plotting.',
            parameter_class=InstrumentRefParameter)

        # TODO add update interval functionality.
        self.add_parameter(
            'update_interval',
            initial_value=0.1,
            docstring=(
                'Interval for updates during the data acquisition loop,' +
                ' everytime more than `update_interval` time has elapsed ' +
                'when acquiring new data points, data is written to file ' +
                'and the live monitoring is updated.'),
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=0)
        )

        # variables that are set before the start of any experiment.
        self._settable_pars = []
        self._setpoints = []
        self._gettable_pars = []

        # variables used for book keeping during acquisition loop.
        self._nr_acquired_values = 0
        self._loop_count = 0
        self._begintime = time.time()
        self._last_upd = time.time()

        # variables used for persistence and plotting
        self._dataset = None
        self._exp_folder = None
        self._plotmon_name = ''
        self._plot_info = {}

        self._GETTABLE_IDX = 0  # avoid magic numbers until/if we support multiple Gettables

        # early exit signal
        self._exit_event = Event()

    ############################################
    # Methods used to control the measurements #
    ############################################

    def run(self, name: str = ''):
        """
        Starts a data acquisition loop.

        Args:
            name (str): Name of the measurement. This name is included in the name of the data files.

        Returns:
            :class:`xarray.Dataset`: the dataset
        """

        # reset all variables that change during acquisition
        self._nr_acquired_values = 0
        self._loop_count = 0
        self._begintime = time.time()

        # initialize an empty dataset
        dataset = initialize_dataset(self._settable_pars, self._setpoints, self._gettable_pars)

        # cannot add it as a separate (nested) dict so make it flat.
        dataset.attrs['name'] = name
        dataset.attrs.update(self._plot_info)

        exp_folder = create_exp_folder(tuid=dataset.attrs['tuid'], name=dataset.attrs['name'])
        dataset.to_netcdf(join(exp_folder, 'dataset.hdf5'))  # Write the empty dataset
        snap = snapshot(update=False, clean=True)  # Save a snapshot of all
        with open(join(exp_folder, 'snapshot.json'), 'w') as file:
            json.dump(snap, file, cls=NumpyJSONEncoder, indent=4)

        plotmon_name = self.instr_plotmon()
        if plotmon_name is not None and plotmon_name != '':
            self.instr_plotmon.get_instr().tuid(dataset.attrs['tuid'])
            # if the timestamp has changed, this will initialize the monitor
            self.instr_plotmon.get_instr().update()

        self._prepare_settables()

        # spawn the measurement loop into a side thread and listen for a keyboard interrupt in the main
        # a keyboard interrupt will signal to the measurement loop that it should stop processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if self._is_soft:
                runner = self._run_soft
            else:
                runner = self._run_hard
            future = executor.submit(runner, dataset, plotmon_name, exp_folder)
            try:
                future.result()
            except KeyboardInterrupt as e:
                print('Interrupt signalled, exiting gracefully...')
                self._exit_event.set()
                future.result()

        dataset.to_netcdf(join(exp_folder, 'dataset.hdf5'))  # Wrap up experiment and store data
        self._finish()
        self._plot_info = {'2D-grid': False}  # reset the plot info for the next experiment.
        self.soft_avg(1)  # reset software averages back to 1

        return dataset

    def run_adapative(self):
        raise NotImplementedError()

    def _run_soft(self, dataset, plotmon_name, exp_folder):
        try:
            while self._get_fracdone() < 1.0:
                self._prepare_gettable()
                for idx, spts in enumerate(self._setpoints):
                    # set all individual setparams
                    for spar, spt in zip(self._settable_pars, spts):
                        # TODO add smartness to avoid setting if unchanged
                        spar.set(spt)
                    # acquire all data points
                    for j, gpar in enumerate(self._gettable_pars):
                        val = gpar.get()
                        old_val = dataset['y{}'.format(j)].values[idx]
                        if self.soft_avg() == 1 or np.isnan(old_val):
                            dataset['y{}'.format(j)].values[idx] = val
                        else:
                            averaged = (val + old_val * self._loop_count) / (1 + self._loop_count)
                            dataset['y{}'.format(j)].values[idx] = averaged
                    self._nr_acquired_values += 1
                    self._update(dataset, plotmon_name, exp_folder)
                self._loop_count += 1
        except KeyboardFinish as e:
            return

    def _run_hard(self, dataset, plotmon_name, exp_folder):
        try:
            while self._get_fracdone() < 1.0:
                setpoint_idx = self._curr_setpoint_idx()
                for i, spar in enumerate(self._settable_pars):
                    swf_setpoints = self._setpoints[:, i]
                    spar.set(swf_setpoints[setpoint_idx])
                self._prepare_gettable(self._setpoints[setpoint_idx:, self._GETTABLE_IDX])

                new_data = self._gettable_pars[self._GETTABLE_IDX].get()  # can return (N, M)
                # if we get a simple array, shape it to (1, M)
                if len(np.shape(new_data)) == 1:
                    new_data = new_data.reshape(1, (len(new_data)))

                for i, row in enumerate(new_data):
                    slice_len = setpoint_idx + len(row)  # the slice we will be updating
                    old_vals = dataset['y{}'.format(i)].values[setpoint_idx:slice_len]
                    old_vals[np.isnan(old_vals)] = 0  # will be full of NaNs on the first iteration, change to 0
                    dataset['y{}'.format(i)].values[setpoint_idx:slice_len] = self._build_data(row, old_vals)
                self._nr_acquired_values += np.shape(new_data)[1]
                self._update(dataset, plotmon_name, exp_folder)
        except KeyboardFinish as e:
            return

    def _build_data(self, new_data, old_data):
        if self.soft_avg() == 1:
            return old_data + new_data
        else:
            return (new_data + old_data * self._loop_count) / (1 + self._loop_count)

    ############################################
    # Methods used to control the measurements #
    ############################################

    def _update(self, dataset, plotmon_name, exp_folder):
        """
        Do any updates to/from external systems, such as saving, plotting, checking for interrupts etc.

        Args:
            dataset (:class:`xarray.Dataset`): the dataset
            plotmon_name (str): the plotmon identifier
            exp_folder (str): persistence directory

        Raises:
            (:class:`quantify.utilities.general.KeyboardFinish`): if the main thread has signalled to exit early
        """
        update = time.time() - self._last_upd > self.update_interval() \
            or self._nr_acquired_values == self._max_setpoints
        if update:
            self.print_progress()
            dataset.to_netcdf(join(exp_folder, 'dataset.hdf5'))
            if plotmon_name is not None and plotmon_name != '':
                self.instr_plotmon.get_instr().update()
            self._last_upd = time.time()
        if self._exit_event.is_set():
            raise KeyboardFinish()

    def _prepare_gettable(self, setpoints=None):
        """
        Call prepare() on the Gettable, if prepare() exists

        Args:
            setpoints (:class:`numpy.ndarray`): The values to pass to the Gettable
        """
        try:
            if setpoints is not None:
                self._gettable_pars[self._GETTABLE_IDX].prepare(setpoints)
            else:
                self._gettable_pars[self._GETTABLE_IDX].prepare()
        # it's fine if the gettable does not have a prepare function
        except AttributeError:
            pass

    def _prepare_settables(self):
        """
        Call prepare() on all Settable, if prepare() exists
        """
        for setpar in self._settable_pars:
            try:
                setpar.prepare()
            # it's fine if the settable does not have a prepare function
            except AttributeError:
                pass

    def _finish(self):
        """
        Call finish() on all Settables and Gettables, if finish() exists
        """
        for p in self._gettable_pars and self._settable_pars:
            try:
                p.finish()
            # it's fine if the parameter does not have a finish function
            except AttributeError:
                pass

    @property
    def _is_soft(self):
        """
        Whether this MeasurementControl controls data stepping
        """
        if is_software_controlled(self._settable_pars[0]) and is_software_controlled(self._gettable_pars[0]):
            return True
        elif not is_software_controlled(self._gettable_pars[0]):
            return False
        else:
            raise Exception("Control mismatch")  # todo improve message

    @property
    def _max_setpoints(self):
        """
        The total number of setpoints to examine
        """
        return len(self._setpoints) * self.soft_avg()

    def _curr_setpoint_idx(self):
        """
        Returns the current position through the sweep
        Updates the _soft_iterations_completed counter as it may have rolled over

        Returns:
            int: setpoint_idx
        """
        acquired = self._nr_acquired_values
        setpoint_idx = acquired % len(self._setpoints)
        self._loop_count = acquired // len(self._setpoints)
        return setpoint_idx

    def _get_fracdone(self):
        """
        Returns the fraction of the experiment that is completed.
        """
        return self._nr_acquired_values / self._max_setpoints

    def print_progress(self):
        percdone = self._get_fracdone()*100
        elapsed_time = time.time() - self._begintime
        progress_message = (
            "\r {percdone}% completed \telapsed time: "
            "{t_elapsed}s \ttime left: {t_left}s".format(
                percdone=int(percdone),
                t_elapsed=round(elapsed_time, 1),
                t_left=round((100.0 - percdone) / percdone * elapsed_time, 1)
                if percdone != 0
                else "",
            )
        )
        if self.on_progress_callback() is not None:
            self.on_progress_callback()(percdone)
        if percdone != 100:
            end_char = ""
        else:
            end_char = "\n"
        if self.verbose():
            print("\r", progress_message, end=end_char)

    ####################################
    # Non-parameter get/set functions  #
    ####################################

    def settables(self, settable_pars):
        """
        Define the settable parameters for the acquisition loop.

        Args:
            settable_pars: parameter(s) to be set during the acquisition loop, accepts:
                - list or tuple of multiple Settable objects
                - a single Settable object.

        The :class:`~quantify.measurement.Settable` helper class defines the requirements for a Settable object.
        """
        # for native nD compatibility we treat this like a list of settables.
        if not isinstance(settable_pars, (list, tuple)):
            settable_pars = [settable_pars]

        self._settable_pars = []
        for _, settable in enumerate(settable_pars):
            self._settable_pars.append(Settable(settable))

    def setpoints(self, setpoints):
        """
        Set setpoints that determine values to be set in acquisition loop.

        Args: setpoints (:class:`numpy.ndarray`) : An array that defines the values to loop over in the experiment.
        The shape of the array has to be either (N,) (N,1) for a 1D loop or (N, M) in the case of an MD loop.

        The setpoints are softly reshaped to (N, M) to be natively compatible with M-dimensional loops.

        .. tip::

            Use :code:`np.colstack((x0, x1))` to reshape multiple
            1D arrays when setting multiple setables.
        """
        if len(np.shape(setpoints)) == 1:
            setpoints = setpoints.reshape((len(setpoints), 1))
        self._setpoints = setpoints

        # set to False whenever new setpoints are defined.
        # this gets updated after calling setpoints_2D.
        self._plot_info['2D-grid'] = False

    def setpoints_grid(self, setpoints):
        """
        Set a setpoint grid that determine values to be set in the acquisition loop. Updates the setpoints in a grid
        by repeating the setpoints M times and filling the second column with tiled values.

        Args: setpoints (list(:class:`numpy.ndarray`)) : The values to loop over in the experiment. The grid is
        reshaped in this order.

        Example

            .. code-block:: python

                MC.settables([t, amp])
                MC.setpoints_grid([times, amplitudes])
                MC.gettables(sig)
                dataset = MC.run('2D grid')
        """
        if len(setpoints) == 2:
            self._plot_info['xlen'] = len(setpoints[0])
            self._plot_info['ylen'] = len(setpoints[1])
            self._plot_info['2D-grid'] = True
        self._setpoints = tile_setpoints_grid(setpoints)

    def gettables(self, gettable_par):
        """
        Define the parameters to be acquired during the acquisition loop.

        Args:
            gettable_pars: parameter(s) to be get during the acquisition loop, accepts:
                 - list or tuple of multiple Gettable objects
                 - a single Gettable object

        The :class:`~quantify.measurement.Gettable` helper class defines the requirements for a Gettable object.

        TODO: support fancier getables, i.e. ones that return
            - more than one quantity
            - multiple points at once (hard loop)

        """
        self._gettable_pars = [Gettable(gettable_par)]


def tile_setpoints_grid(setpoints):
    """
    Tile setpoints into an n-dimensional grid.

    Args: setpoints (list(:class:`numpy.ndarray`)): A list of arrays that defines the values to loop over in the
    experiment. The grid is reshaped in this order.

    Returns:
        :class:`numpy.ndarray`: an array with repeated x-values and tiled xn-values.

    .. warning ::

        using this method typecasts all values into the same type. This may lead to validator errors when setting
        e.g., a float instead of an int.
    """
    xn = setpoints[0].reshape((len(setpoints[0]), 1))
    for setpoints_n in setpoints[1:]:
        curr_l = len(xn)
        new_l = len(setpoints_n)
        col_stack = []
        for i in range(0, np.size(xn, 1)):
            col_stack.append(np.tile(xn[:, i], new_l))
        col_stack.append(np.repeat(setpoints_n, curr_l))
        xn = np.column_stack(col_stack)
    return xn
