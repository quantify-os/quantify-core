# -----------------------------------------------------------------------------
# Description:    Module containing the MeasurementControl.
# Repository:     https://gitlab.com/qblox/packages/software/quantify/
# Copyright (C) Qblox BV (2020)
# -----------------------------------------------------------------------------
import time
import json
import types
from os.path import join

import numpy as np
import adaptive
from qcodes import Instrument
from qcodes import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils.helpers import NumpyJSONEncoder
from quantify.data.handling import initialize_dataset, create_exp_folder, snapshot, grow_dataset, trim_dataset
from quantify.measurement.types import Settable, Gettable, is_software_controlled


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
        self._plot_info = {'2D-grid': False}

    ############################################
    # Methods used to control the measurements #
    ############################################

    def _reset(self):
        """
        Resets all experiment specific variables for a new run.
        """
        self._nr_acquired_values = 0
        self._loop_count = 0
        self._begintime = time.time()

    def _init(self, name):
        """
        Initializes MC, such as creating the Dataset, experiment folder and such.
        """
        # initialize an empty dataset
        self._dataset = initialize_dataset(self._settable_pars, self._setpoints, self._gettable_pars)

        # cannot add it as a separate (nested) dict so make it flat.
        self._dataset.attrs['name'] = name
        self._dataset.attrs.update(self._plot_info)

        self._exp_folder = create_exp_folder(tuid=self._dataset.attrs['tuid'], name=self._dataset.attrs['name'])
        self._dataset.to_netcdf(join(self._exp_folder, 'dataset.hdf5'))  # Write the empty dataset
        snap = snapshot(update=False, clean=True)  # Save a snapshot of all
        with open(join(self._exp_folder, 'snapshot.json'), 'w') as file:
            json.dump(snap, file, cls=NumpyJSONEncoder, indent=4)

        self._plotmon_name = self.instr_plotmon()
        if self._plotmon_name is not None and self._plotmon_name != '':
            self.instr_plotmon.get_instr().tuid(self._dataset.attrs['tuid'])
            # if the timestamp has changed, this will initialize the monitor
            self.instr_plotmon.get_instr().update()

    def run(self, name: str = ''):
        """
        Starts a data acquisition loop.

        Args:
            name (str): Name of the measurement. This name is included in the name of the data files.

        Returns:
            :class:`xarray.Dataset`: the dataset
        """

        self._reset()
        self._init(name)

        self._prepare_settables()

        try:
            if self._is_soft:
                self._run_soft()
            else:
                self._run_hard()
        except KeyboardInterrupt as e:
            print()
            print("Interrupt signalled, exiting gracefully...")

        self._dataset.to_netcdf(join(self._exp_folder, 'dataset.hdf5'))  # Wrap up experiment and store data
        self._finish()
        self._plot_info = {'2D-grid': False}  # reset the plot info for the next experiment.
        self.soft_avg(1)  # reset software averages back to 1

        return self._dataset

    def run_adaptive(self, name, params):
        """
        Starts a data acquisition loop using an adaptive function.

        .. warning ::
            The functionality of this mode can be complex - it is recommended to read the relevant long form
            documentation.

        Args:
            name (str): Name of the measurement. This name is included in the name of the data files.
            params (dict): Key value parameters describe the adaptive function to use, and any further parameters for
            that function.

        Returns:
            :class:`xarray.Dataset`: the dataset
        """
        def measure(vec) -> float:
            if len(self._dataset['y0']) == self._nr_acquired_values:
                self._dataset = grow_dataset(self._dataset)

            #  1D sweeps return single values, wrap in a list
            if np.isscalar(vec):
                vec = [vec]

            self._soft_set_and_get(vec, self._nr_acquired_values)
            ret = self._dataset['y0'].values[self._nr_acquired_values]
            self._nr_acquired_values += 1
            self._update("Running adaptively")
            return ret

        def subroutine():
            self._prepare_settables()
            self._prepare_gettable()

            adaptive_function = params.get("adaptive_function")
            af_pars_copy = dict(params)

            # leveraging the adaptive library
            if isinstance(adaptive_function, type) and issubclass(adaptive_function, adaptive.learner.BaseLearner):
                goal = af_pars_copy['goal']
                unusued_pars = ["adaptive_function", "goal"]
                for unusued_par in unusued_pars:
                    af_pars_copy.pop(unusued_par, None)
                learner = adaptive_function(measure, **af_pars_copy)
                adaptive.runner.simple(learner, goal)

            # free function
            if isinstance(adaptive_function, types.FunctionType):
                unused_pars = ["adaptive_function"]
                for unused_par in unused_pars:
                    af_pars_copy.pop(unused_par, None)
                adaptive_function(measure, **af_pars_copy)

        if self.soft_avg() != 1:
            raise ValueError("software averaging not allowed in adaptive loops; currently set to {}."
                             .format(self.soft_avg()))

        self._reset()
        self.setpoints(np.empty((64, len(self._settable_pars))))  # block out some space in the dataset
        self._init(name)
        try:
            subroutine()
        except KeyboardInterrupt as e:
            print()
            print('Interrupt signalled, exiting gracefully...')

        self._finish()
        self._dataset = trim_dataset(self._dataset)
        self._dataset.to_netcdf(join(self._exp_folder, 'dataset.hdf5'))  # Wrap up experiment and store data
        return self._dataset

    def _run_soft(self):
        while self._get_fracdone() < 1.0:
            self._prepare_gettable()
            for row in self._setpoints:
                self._soft_set_and_get(row, self._curr_setpoint_idx())
                self._nr_acquired_values += 1
                self._update()
            self._loop_count += 1

    def _run_hard(self):
        while self._get_fracdone() < 1.0:
            setpoint_idx = self._curr_setpoint_idx()
            for i, spar in enumerate(self._settable_pars):
                spar.set(self._setpoints[setpoint_idx:, i])
            self._prepare_gettable()

            y_offset = 0
            for gpar in self._gettable_pars:
                new_data = gpar.get()  # can return (N, M)
                # if we get a simple array, shape it to (1, M)
                if len(np.shape(new_data)) == 1:
                    new_data = new_data.reshape(1, (len(new_data)))

                for row in new_data:
                    slice_len = setpoint_idx + len(row)  # the slice we will be updating
                    old_vals = self._dataset['y{}'.format(y_offset)].values[setpoint_idx:slice_len]
                    old_vals[np.isnan(old_vals)] = 0  # will be full of NaNs on the first iteration, change to 0
                    self._dataset['y{}'.format(y_offset)].values[setpoint_idx:slice_len] = self._build_data(row, old_vals)
                    y_offset += 1
                self._nr_acquired_values += np.shape(new_data)[1]
            self._update()

    def _build_data(self, new_data, old_data):
        if self.soft_avg() == 1:
            return old_data + new_data
        else:
            return (new_data + old_data * self._loop_count) / (1 + self._loop_count)

    def _soft_set_and_get(self, setpoints: np.ndarray, idx: int):
        """
        Processes one row of setpoints. Sets all settables, gets all gettables, encodes new data in dataset

        Note: some lines in this function are redundant depending on mode (sweep vs adaptive). Specifically
        - in sweep, the x dimensions are already filled
        - in adaptive, soft_avg is always 1
        """
        # set all individual setparams
        for setpar_idx, (spar, spt) in enumerate(zip(self._settable_pars, setpoints)):
            self._dataset['x{}'.format(setpar_idx)].values[idx] = spt
            spar.set(spt)  # TODO add smartness to avoid setting if unchanged
        # get all data points
        y_offset = 0
        for gpar in self._gettable_pars:
            new_data = gpar.get()
            # if the gettable returned a float, cast to list
            if np.isscalar(new_data):
                new_data = [new_data]
            # iterate through the data list, each element is different y for these x coordinates
            for val in new_data:
                old_val = self._dataset['y{}'.format(y_offset)].values[idx]
                if self.soft_avg() == 1 or np.isnan(old_val):
                    self._dataset['y{}'.format(y_offset)].values[idx] = val
                else:
                    averaged = (val + old_val * self._loop_count) / (1 + self._loop_count)
                    self._dataset['y{}'.format(y_offset)].values[idx] = averaged
                y_offset += 1

    ############################################
    # Methods used to control the measurements #
    ############################################

    def _update(self, print_message: str = None):
        """
        Do any updates to/from external systems, such as saving, plotting, checking for interrupts etc.
        """
        update = time.time() - self._last_upd > self.update_interval() \
            or self._nr_acquired_values == self._max_setpoints
        if update:
            self.print_progress(print_message)
            self._dataset.to_netcdf(join(self._exp_folder, 'dataset.hdf5'))
            if self._plotmon_name is not None and self._plotmon_name != '':
                self.instr_plotmon.get_instr().update()
            self._last_upd = time.time()

    def _prepare_gettable(self):
        """
        Call prepare() on the Gettable, if prepare() exists
        """
        for getpar in self._gettable_pars:
            try:
                getpar.prepare()
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
        if any(is_software_controlled(gpar) for gpar in self._gettable_pars):
            if not all(is_software_controlled(gpar) for gpar in self._gettable_pars):
                raise Exception("Control mismatch; all Gettables must have the same Control Mode")
            return True
        return False

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

    def print_progress(self, progress_message: str = None):
        percdone = self._get_fracdone()*100
        elapsed_time = time.time() - self._begintime
        if not progress_message:
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
        for settable in settable_pars:
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

    def gettables(self, gettable_pars):
        """
        Define the parameters to be acquired during the acquisition loop.

        Args:
            gettable_pars: parameter(s) to be get during the acquisition loop, accepts:
                 - list or tuple of multiple Gettable objects
                 - a single Gettable object

        The :class:`~quantify.measurement.Gettable` helper class defines the requirements for a Gettable object.
        """
        if not isinstance(gettable_pars, (list, tuple)):
            gettable_pars = [gettable_pars]

        self._gettable_pars = []
        for gpar in gettable_pars:
            self._gettable_pars.append(Gettable(gpar))


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
