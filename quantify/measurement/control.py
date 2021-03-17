# -----------------------------------------------------------------------------
# Description:    Module containing the MeasurementControl.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations
import time
import json
import types
import tempfile
from os.path import join
from collections.abc import Iterable
from collections import OrderedDict
import itertools
from itertools import chain
import signal
import threading

import xarray as xr
import numpy as np
import adaptive
from filelock import FileLock
from qcodes import Instrument
from qcodes import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils.helpers import NumpyJSONEncoder
from quantify.data.handling import (
    initialize_dataset,
    create_exp_folder,
    snapshot,
    grow_dataset,
    trim_dataset,
    DATASET_NAME,
    write_dataset,
)
from quantify.utilities.general import call_if_has_method
from quantify.measurement.types import Settable, Gettable, is_batched

# Intended for plotting monitors that run in separate processes
_DATASET_LOCKS_DIR = tempfile.gettempdir()


class MeasurementControl(Instrument):
    """
    Instrument responsible for controlling the data acquisition loop.

    MeasurementControl (MC) is based on the notion that every experiment consists of
    the following steps:

        1. Set some parameter(s)            (settable_pars)
        2. Measure some other parameter(s)  (gettable_pars)
        3. Store the data.

    Example:

        .. code-block:: python

            MC.settables(mw_source1.freq)
            MC.setpoints(np.arange(5e9, 5.2e9, 100e3))
            MC.gettables(pulsar_QRM.signal)
            dataset = MC.run(name='Frequency sweep')


    MC exists to enforce structure on experiments. Enforcing this structure allows:

        - Standardization of data storage.
        - Providing basic real-time visualization.

    MC imposes minimal constraints and allows:

    - Iterative loops, experiments in which setpoints are processed step by step.
    - Batched loops, experiments in which setpoints are processed in batches.
    - Adaptive loops, setpoints are determined based on measured values.

    """

    def __init__(self, name: str):
        """
        Creates an instance of the Measurement Control.

        Parameters
        ----------
        name : str
            name of this instrument
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
            "Callable accepting floats between 0 and 100 indicating %% done.",
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
            "instr_plotmon",
            docstring="Instrument responsible for live plotting. "
            "Can be set to str(None) to disable live plotting.",
            parameter_class=InstrumentRefParameter,
            vals=vals.MultiType(vals.Strings(), vals.Enum(None)),
        )

        self.add_parameter(
            "instrument_monitor",
            docstring="Instrument responsible for live monitoring summarized snapshot. "
            "Can be set to str(None) to disable monitoring of snapshot.",
            parameter_class=InstrumentRefParameter,
            vals=vals.MultiType(vals.Strings(), vals.Enum(None)),
        )

        self.add_parameter(
            "update_interval",
            initial_value=0.5,
            docstring=(
                "Interval for updates during the data acquisition loop,"
                " everytime more than `update_interval` time has elapsed "
                "when acquiring new data points, data is written to file "
                "and the live monitoring is updated."
            ),
            parameter_class=ManualParameter,
            # minimum value set to avoid performance issues
            vals=vals.Numbers(min_value=0.1),
        )

        # variables that are set before the start of any experiment.
        self._settable_pars = []
        self._setpoints = []
        self._setpoints_input = []
        self._gettable_pars = []

        # variables used for book keeping during acquisition loop.
        self._nr_acquired_values = 0
        self._loop_count = 0
        self._begintime = time.time()
        self._last_upd = time.time()
        self._batch_size_last = None

        # variables used for persistence and plotting
        self._dataset = None
        self._exp_folder = None
        self._plotmon_name = ""
        self._plot_info = {"2D-grid": False}

        # properly handling KeyboardInterrupts
        self._thread_data = threading.local()
        # counter for KeyboardInterrupts to allow forced interrupt
        self._thread_data.events_num = 0

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
        self._batch_size_last = None
        # Reset KeyboardInterrupt counter
        self._thread_data.events_num = 0
        # Assign handler to interrupt signal
        signal.signal(signal.SIGINT, self._interrupt_handler)

    def _reset_post(self):
        """
        Resets specific variables that can change before `.run()`
        """
        self._plot_info = {"2D-grid": False}
        self.soft_avg(1)
        # Reset to default interrupt handler
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def _init(self, name):
        """
        Initializes MC, such as creating the Dataset, experiment folder and such.
        """
        # needs to be calculated here because we need the settables' `.batched`
        if self._setpoints is None:
            self._setpoints = grid_setpoints(self._setpoints_input, self._settable_pars)

        # initialize an empty dataset
        self._dataset = initialize_dataset(
            self._settable_pars, self._setpoints, self._gettable_pars
        )

        # cannot add it as a separate (nested) dict so make it flat.
        self._dataset.attrs["name"] = name
        self._dataset.attrs.update(self._plot_info)

        self._exp_folder = create_exp_folder(
            tuid=self._dataset.attrs["tuid"], name=self._dataset.attrs["name"]
        )
        self._safe_write_dataset()  # Write the empty dataset

        snap = snapshot(update=False, clean=True)  # Save a snapshot of all
        with open(join(self._exp_folder, "snapshot.json"), "w") as file:
            json.dump(snap, file, cls=NumpyJSONEncoder, indent=4)

    def run(self, name: str = ""):
        """
        Starts a data acquisition loop.

        Parameters
        ----------
        name : str
            Name of the measurement. This name is included in the name of the data files.
        Returns
        -------
        :class:`xarray.Dataset`
            the dataset
        """

        self._reset()
        self._init(name)

        self._prepare_settables()

        try:
            if self._get_is_batched():
                if self.verbose():
                    print("Starting batched measurement...")
                self._run_batched()
            else:
                if self.verbose():
                    print("Starting iterative measurement...")
                self._run_iterative()
        except KeyboardInterrupt:
            print("\nInterrupt signaled, exiting gracefully...")

        self._safe_write_dataset()  # Wrap up experiment and store data
        self._finish()
        self._reset_post()

        self._check_interrupt()  # Propagate interruption

        return self._dataset

    def run_adaptive(self, name, params):
        """
        Starts a data acquisition loop using an adaptive function.

        .. warning ::
            The functionality of this mode can be complex - it is recommended to read the relevant long form
            documentation.

        Parameters
        ----------
        name : str
            Name of the measurement. This name is included in the name of the data files.
        params : dict
            Key value parameters describe the adaptive function to use, and any further parameters for that function.
        Returns
        -------
        :class:`xarray.Dataset`
            the dataset
        """

        def measure(vec) -> float:
            if len(self._dataset["y0"]) == self._nr_acquired_values:
                self._dataset = grow_dataset(self._dataset)

            #  1D sweeps return single values, wrap in a list
            if np.isscalar(vec):
                vec = [vec]

            self._iterative_set_and_get(vec, self._nr_acquired_values)
            ret = self._dataset["y0"].values[self._nr_acquired_values]
            self._nr_acquired_values += 1
            self._update(".")
            self._check_interrupt()
            return ret

        def subroutine():
            self._prepare_settables()
            self._prepare_gettables()

            adaptive_function = params.get("adaptive_function")
            af_pars_copy = dict(params)

            # leveraging the adaptive library
            if isinstance(adaptive_function, type) and issubclass(
                adaptive_function, adaptive.learner.BaseLearner
            ):
                goal = af_pars_copy["goal"]
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
            raise ValueError(
                "software averaging not allowed in adaptive loops; "
                f"currently set to {self.soft_avg()}."
            )

        self._reset()
        self.setpoints(
            np.empty((64, len(self._settable_pars)))
        )  # block out some space in the dataset
        self._init(name)
        try:
            print("Running adaptively...")
            subroutine()
        except KeyboardInterrupt:
            print("\nInterrupt signaled, exiting gracefully...")

        self._finish()
        self._dataset = trim_dataset(self._dataset)
        self._safe_write_dataset()  # Wrap up experiment and store data

        self._check_interrupt()  # Propagate interruption

        return self._dataset

    def _run_iterative(self):
        while self._get_fracdone() < 1.0:
            self._prepare_gettables()
            for row in self._setpoints:
                self._iterative_set_and_get(row, self._curr_setpoint_idx())
                self._nr_acquired_values += 1
                self._update()
                self._check_interrupt()
            self._loop_count += 1

    def _run_batched(self):
        batch_size = self._get_batch_size()
        where_batched = self._get_where_batched()
        where_iterative = self._get_where_iterative()
        batched_settables = self._get_batched_settables()
        iterative_settables = self._get_iterative_settables()

        if self.verbose():
            print(
                "Iterative settable(s) [outer loop(s)]:\n\t",
                ", ".join(par.name for par in iterative_settables) or "--- (None) ---",
                "\nBatched settable(s):\n\t",
                ", ".join(par.name for par in batched_settables),
                f"\nBatch size limit: {batch_size:d}\n",
            )
        while self._get_fracdone() < 1.0:
            setpoint_idx = self._curr_setpoint_idx()
            self._batch_size_last = batch_size
            slice_len = setpoint_idx + self._batch_size_last
            for i, spar in enumerate(iterative_settables):
                # Here ensure that all setpoints of each iterative settable are the same
                # within each batch
                val, iterator = next(
                    itertools.groupby(
                        self._setpoints[setpoint_idx:slice_len, where_iterative[i]]
                    )
                )
                spar.set(val)
                # We also determine the size of each next batch
                self._batch_size_last = min(self._batch_size_last, len(tuple(iterator)))

            slice_len = setpoint_idx + self._batch_size_last
            for i, spar in enumerate(batched_settables):
                pnts = self._setpoints[setpoint_idx:slice_len, where_batched[i]]
                spar.set(pnts)
            # Update for `print_progress`
            self._batch_size_last = min(self._batch_size_last, len(pnts))

            self._prepare_gettables()

            y_off = 0
            for gpar in self._gettable_pars:
                new_data = gpar.get()  # can return (N, M)
                # if we get a simple array, shape it to (1, M)
                if len(np.shape(new_data)) == 1:
                    new_data = new_data.reshape(1, (len(new_data)))

                for row in new_data:
                    yi_name = f"y{y_off}"
                    slice_len = setpoint_idx + len(row)  # the slice we will be updating
                    old_vals = self._dataset[yi_name].values[setpoint_idx:slice_len]
                    old_vals[
                        np.isnan(old_vals)
                    ] = 0  # will be full of NaNs on the first iteration, change to 0
                    self._dataset[yi_name].values[
                        setpoint_idx:slice_len
                    ] = self._build_data(row, old_vals)
                    y_off += 1

            self._nr_acquired_values += np.shape(new_data)[1]
            self._update()
            self._check_interrupt()

    def _build_data(self, new_data, old_data):
        if self.soft_avg() == 1:
            return old_data + new_data

        return (new_data + old_data * self._loop_count) / (1 + self._loop_count)

    def _iterative_set_and_get(self, setpoints: np.ndarray, idx: int):
        """
        Processes one row of setpoints. Sets all settables, gets all gettables, encodes new data in dataset

        .. note ::
            Note: some lines in this function are redundant depending on mode (sweep vs adaptive). Specifically
                - in sweep, the x dimensions are already filled
                - in adaptive, soft_avg is always 1
        """
        # set all individual setparams
        for setpar_idx, (spar, spt) in enumerate(zip(self._settable_pars, setpoints)):
            self._dataset[f"x{setpar_idx}"].values[idx] = spt
            spar.set(spt)
        # get all data points
        y_offset = 0
        for gpar in self._gettable_pars:
            new_data = gpar.get()
            # if the gettable returned a float, cast to list
            if np.isscalar(new_data):
                new_data = [new_data]
            # iterate through the data list, each element is different y for these x coordinates
            for val in new_data:
                yi_name = f"y{y_offset}"
                old_val = self._dataset[yi_name].values[idx]
                if self.soft_avg() == 1 or np.isnan(old_val):
                    self._dataset[yi_name].values[idx] = val
                else:
                    averaged = (val + old_val * self._loop_count) / (
                        1 + self._loop_count
                    )
                    self._dataset[yi_name].values[idx] = averaged
                y_offset += 1

    ############################################
    # Methods used to control the measurements #
    ############################################

    def _interrupt_handler(self, signal, frame):
        """
        A proper handler for signal.SIGINT (a.k.a. KeyboardInterrupt).

        Used to avoid affecting any other code, e.g. instruments drivers, and be able
        safely stop the MC after an iteration finishes.
        """
        self._thread_data.events_num += 1
        if self._thread_data.events_num >= 5:
            raise KeyboardInterrupt
        print(
            f"\n\n[!!!] {self._thread_data.events_num} interruption(s) signaled. "
            "Stopping after this iteration/batch.\n"
            f"[Send more {5 -  self._thread_data.events_num} interruptions to force stop (not safe!)].\n"
        )

    def _check_interrupt(self):
        """
        Verifies if the user has signaled the interruption of the experiment.

        Intended to be used after each iteration or after each batch of data.
        """
        if self._thread_data.events_num >= 1:
            # It is safe to raise the KeyboardInterrupt here because we are guaranteed
            # To be running MC code. The exception can be handled in a try-except
            raise KeyboardInterrupt("Measurement interrupted")

    def _update(self, print_message: str = None):
        """
        Do any updates to/from external systems, such as saving, plotting, etc.
        """
        update = (
            time.time() - self._last_upd > self.update_interval()
            or self._nr_acquired_values == self._get_max_setpoints()
        )
        if update:
            self.print_progress(print_message)

            self._safe_write_dataset()

            if self.instr_plotmon():
                # Plotmon requires to know which dataset was modified
                self.instr_plotmon.get_instr().update(tuid=self._dataset.attrs["tuid"])

            if self.instrument_monitor():
                self.instrument_monitor.get_instr().update()

            self._last_upd = time.time()

    def _prepare_gettables(self) -> None:
        """
        Call prepare() on the Gettable, if prepare() exists
        """
        for getpar in self._gettable_pars:
            call_if_has_method(getpar, "prepare")

    def _prepare_settables(self) -> None:
        """
        Call prepare() on all Settable, if prepare() exists
        """
        for setpar in self._settable_pars:
            call_if_has_method(setpar, "prepare")

    def _finish(self) -> None:
        """
        Call finish() on all Settables and Gettables, if finish() exists
        """
        for par in self._gettable_pars + self._settable_pars:
            call_if_has_method(par, "finish")

    def _get_batched_mask(self):
        return tuple(is_batched(spar) for spar in self._settable_pars)

    def _get_where_batched(self):
        # Indices to select correct entries in results data
        return np.where(self._get_batched_mask())[0]

    def _get_where_iterative(self):
        return np.where(tuple(not m for m in self._get_batched_mask()))[0]

    def _get_iterative_settables(self):
        return tuple(spar for spar in self._settable_pars if not is_batched(spar))

    def _get_batched_settables(self):
        return tuple(spar for spar in self._settable_pars if is_batched(spar))

    def _get_batch_size(self):
        # np.inf is not supported by the JSON schema, but we keep the code robust
        min_with_inf = min(
            getattr(par, "batch_size", np.inf)
            for par in chain.from_iterable((self._settable_pars, self._gettable_pars))
        )
        return min(min_with_inf, len(self._setpoints))

    def _get_is_batched(self) -> bool:
        if any(
            is_batched(gpar) for gpar in chain(self._gettable_pars, self._settable_pars)
        ):
            if not all(is_batched(gpar) for gpar in self._gettable_pars):
                raise RuntimeError(
                    "Control mismatch; all Gettables must have batched Control Mode, "
                    "i.e. all gettables must have `.batched=True`."
                )
            if not any(is_batched(spar) for spar in self._settable_pars):
                raise RuntimeError(
                    "Control mismatch; At least one settable must have "
                    "`settable.batched=True`, if the gettables are batched."
                )
            return True

        return False

    def _get_max_setpoints(self) -> int:
        """
        The total number of setpoints to examine
        """
        return len(self._setpoints) * self.soft_avg()

    def _curr_setpoint_idx(self) -> int:
        """
        Current position through the sweep

        Returns
        -------
        int
            setpoint_idx
        """
        acquired = self._nr_acquired_values
        setpoint_idx = acquired % len(self._setpoints)
        self._loop_count = acquired // len(self._setpoints)
        return setpoint_idx

    def _get_fracdone(self) -> float:
        """
        Returns the fraction of the experiment that is completed.
        """
        return self._nr_acquired_values / self._get_max_setpoints()

    def print_progress(self, progress_message: str = None):
        """
        Prints the provided `progress_messages` or a default one; and calls the
        callback specified by `on_progress_callback`.
        Printing can be suppressed with `.verbose(False)`.
        """
        progress_percent = self._get_fracdone() * 100
        elapsed_time = time.time() - self._begintime
        if self.verbose() and progress_message is None:
            t_left = (
                round((100.0 - progress_percent) / progress_percent * elapsed_time, 1)
                if progress_percent != 0
                else ""
            )
            progress_message = (
                f"\r{int(progress_percent):#3d}% completed  elapsed time: "
                f"{int(round(elapsed_time, 1)):#6d}s  time left: {int(round(t_left, 1)):#6d}s  "
            )
            if self._batch_size_last is not None:
                progress_message += f"last batch size: {self._batch_size_last:#6d}  "

            print(progress_message, end="" if progress_percent < 100 else "\n")

        if self.on_progress_callback() is not None:
            self.on_progress_callback()(progress_percent)

        if self.verbose() and progress_message is not None:
            print(progress_message, end="")

    def _safe_write_dataset(self):
        """
        Uses a lock when writing the file to stay safe for multiprocessing.
        Locking files are written into a temporary dir to avoid polluting
        the experiment container.
        """
        filename = join(self._exp_folder, DATASET_NAME)
        # Multiprocess safe
        lockfile = join(
            _DATASET_LOCKS_DIR,
            self._dataset.attrs["tuid"] + "-" + DATASET_NAME + ".lock",
        )
        with FileLock(lockfile, 5):
            write_dataset(path=filename, dataset=self._dataset)

    ####################################
    # Non-parameter get/set functions  #
    ####################################

    def settables(self, settable_pars):
        """
        Define the settable parameters for the acquisition loop.

        The :class:`~quantify.measurement.Settable` helper class defines the requirements for a Settable object.

        Parameters
        ---------
        settable_pars
            parameter(s) to be set during the acquisition loop, accepts a list or tuple of multiple Settable objects
            or a single Settable object.
        """
        # for native nD compatibility we treat this like a list of settables.
        if not isinstance(settable_pars, (list, tuple)):
            settable_pars = [settable_pars]

        self._settable_pars = []
        for settable in settable_pars:
            self._settable_pars.append(Settable(settable))

    def setpoints(self, setpoints: np.ndarray):
        """
        Set setpoints that determine values to be set in acquisition loop.

        .. tip::

            Use :func:`~numpy.column_stack` to reshape multiple
            1D arrays when setting multiple settables.

        Parameters
        ----------
        setpoints :
            An array that defines the values to loop over in the experiment.
            The shape of the array has to be either (N,) or (N,1) for a 1D loop;
            or (N, M) in the case of an MD loop.
        """
        if len(np.shape(setpoints)) == 1:
            setpoints = setpoints.reshape((len(setpoints), 1))
        self._setpoints = setpoints

    def setpoints_grid(self, setpoints: Iterable[np.array]):
        """
        Makes a grid from the provided `setpoints` assuming each array element
        corresponds to an orthogonal dimension.
        The resulting gridded points determine values to be set in the acquisition loop.

        The gridding is such that the inner most loop corresponds to the batched settable
        with the smallest `.batch_size`.

        Parameters
        ----------
        setpoints : list
            The values to loop over in the experiment. The grid is reshaped in the same order.


        .. include:: ./docstring_examples/quantify.measurement.control.setpoints_grid.rst.txt
        """
        self._setpoints = None  # assigned later in the `._init()`
        self._setpoints_input = setpoints

        if len(setpoints) == 2:
            self._plot_info["xlen"] = len(setpoints[0])
            self._plot_info["ylen"] = len(setpoints[1])
            self._plot_info["2D-grid"] = True

    def gettables(self, gettable_pars):
        """
        Define the parameters to be acquired during the acquisition loop.

        The :class:`~quantify.measurement.Gettable` helper class defines the requirements for a Gettable object.

        Parameters
        ----------
        gettable_pars
            parameter(s) to be get during the acquisition loop, accepts:
                 - list or tuple of multiple Gettable objects
                 - a single Gettable object
        """
        if not isinstance(gettable_pars, (list, tuple)):
            gettable_pars = [gettable_pars]

        self._gettable_pars = []
        for gpar in gettable_pars:
            self._gettable_pars.append(Gettable(gpar))


def grid_setpoints(setpoints: Iterable, settables: Iterable = None) -> np.ndarray:
    """
    Makes gridded setpoints. If `settables` is provided, the gridding is such that the
    inner most loop corresponds to the batched settable with the smallest `.batch_size`.

    .. warning ::

        Using this method typecasts all values into the same type.
        This may lead to validator errors when setting
        e.g., a `float` instead of an `int`.

    Parameters
    ----------
    setpoints
        A list of arrays that defines the values to loop over in the experiment for each
        orthogonal dimension. The grid is reshaped in the same order.
    settables
        A list of settable objects to which the elements in the `setpoints` correspond to.
        Used to correctly grid data when mixing batched and iterative settables.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array where the first numpy axis correspond to individual setpoints.
    """

    if settables is None:
        settables = [None] * len(setpoints)

    coordinates_names = [f"x{i}" for i, pnts in enumerate(setpoints)]
    dataset_coordinates = xr.Dataset(
        coords=OrderedDict(zip(coordinates_names, setpoints))
    )
    coordinates_batched = [
        name for name, spar in zip(coordinates_names, settables) if is_batched(spar)
    ]
    coordinates_iterative = sorted(
        (
            name
            for name, spar in zip(coordinates_names, settables)
            if not is_batched(spar)
        ),
        reverse=True,
    )

    if len(coordinates_batched):
        batch_sizes = [
            getattr(spar, "batch_size", np.inf)
            for spar in settables
            if is_batched(spar)
        ]
        coordinates_batched_set = set(coordinates_batched)
        inner_coord_name = coordinates_batched[np.argmin(batch_sizes)]
        coordinates_batched_set.remove(inner_coord_name)
        # The inner most coordinate must correspond to the batched settable with min `.batch_size`
        stack_order = (
            coordinates_iterative
            + sorted(coordinates_batched_set, reverse=True)
            + [inner_coord_name]
        )
    else:
        stack_order = coordinates_iterative + sorted(coordinates_batched, reverse=True)

    # Internally the xarray's stack mechanism is used to achieve the desired grid
    stacked_dset = dataset_coordinates.stack(dim_0=stack_order)

    # Return numpy array in the original order
    return np.column_stack([stacked_dset[name].values for name in coordinates_names])
