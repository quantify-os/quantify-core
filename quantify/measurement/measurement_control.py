import numpy as np
import time
from os.path import join
from qcodes import Instrument
from quantify.measurement.data_handling import initialize_dataset, \
    create_exp_folder
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals


class MeasurementControl(Instrument):
    """
    Instrument responsible for controling the data acquisition loop.

    MeasurementControl (MC) is based on the notion that every experiment
    consists of the following step.

        1. Set some parameter(s)            (setable_pars)
        2. Measure some other parameter(s)  (getable_pars)
        3. Store the data.

    MC exists to enforce structure on experiments.
    Enforcing this structure allows
        - Standardization of data storage.
        - Providing basic real-time visualization.

    MC imposes minimal constraints and allows
    - Soft loops, experiments in which MC controlled acquisition loop.
    - Hard loops, experiments in which MC is not in control of acquisition.
    - Adaptive loops, experiments in which data points to sample are based
        on the previously measured results through a function provided by
        the user.

    """

    def __init__(
            self,
            name: str):  # verbose: bool = True
        """
        Creates an instance of the Measurement Control.

        Args:
            name (str): name
        """
        super().__init__(name=name)

        # Paramaters are attributes that we include in logging
        # and intend the user to change.

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

        # variables that are set before the start
        # of any experiment.
        self._setable_pars = []
        self._setpoints = []
        self._getable_pars = []

        # Variables used for book keeping during acquisition loop.
        self._nr_acquired_values = 0
        self._begintime = time.time()

    ############################################
    # Methods used to control the measurements #
    ############################################

    def run(self, name: str = '',
            mode: str = '1D'):
        """
        Starts a data acquisition loop.

        Args:
            name (string):
                Name of the measurement. This name is included in the
                name of the data files.
            mode (str):
                Measurement mode. Can '1D', '2D', or 'adaptive'.

        Returns:
            dataset : an xarray Dataset object.
        """

        # Reset all variables that change during acquisition
        self._nr_acquired_values = 0
        self._begintime = time.time()

        # initialize an empty dataset

        dataset = initialize_dataset(self._setable_pars,
                                     self._setpoints,
                                     self._getable_pars)

        exp_folder = create_exp_folder(tuid=dataset.attrs['tuid'],
                                       name=name)
        # TODO: Prepare statements

        # TODO percdone

        # Iterate over all points to set
        for idx, spts in enumerate(self._setpoints):
            # set all individual setparams
            for spar, spt in zip(self._setable_pars, spts):
                # TODO add smartness to avoid setting if unchanged
                spar.set(spt)
            # acquire all data points
            for j, gpar in enumerate(self._getable_pars):
                val = gpar.get()
                dataset['y{}'.format(j)].values[idx] = val

            self._nr_acquired_values += 1

            # Here we do saving, plotting, checking for interupts etc.
            self.print_progress()

        # Wrap up experiment and store data
        # FIXME: this needs to happen in the loop.
        dataset.to_netcdf(join(exp_folder, 'dataset.hdf5'))

        return dataset

    ############################################
    # Methods used to control the measurements #
    ############################################

    def _get_fracdone(self):
        """
        Returns the fraction of the experiment that is completed.
        """
        fracdone = self._nr_acquired_values / (
            len(self._setpoints)*self.soft_avg())
        return fracdone

    def print_progress(self):
        percdone = self._get_fracdone()*100
        elapsed_time = time.time() - self._begintime
        progress_message = (
            "\r {percdone}% completed \telapsed time: "
            "{t_elapsed}s \ttime left: {t_left}s".format(
                percdone=int(percdone),
                t_elapsed=round(elapsed_time, 1),
                t_left=round((100.0 - percdone) / (percdone) * elapsed_time, 1)
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

    def set_setpars(self, setable_pars):
        """
        Define the setable parameters for the acquisition loop.

        Args:
            setable_pars: parameter(s) to be set during the acquisition loop.
                accepts:
                - list or tuple of multiple setable objects
                - a single setable object.
            A setable object is an object that has at least
                - unit (str) attribute
                - name (str) attribute
                - set method
                See also `is_getable`
        """
        # for native nD compatibility we treat this like a list of
        # setables.
        if not isinstance(setable_pars, (list, tuple)):
            setable_pars = [setable_pars]

        self._setable_pars = []
        for i, setable in enumerate(setable_pars):
            if is_setable(setable):
                self._setable_pars.append(setable)

    def set_setpoints(self, setpoints):
        """
        Set setpoints that determine values to be set in acquisition loop.

        Args:
            setpoints (np.array) : An array that defines the values to loop
                over in the experiment. The shape of the the array has to be
                either (N,) (N,1) for a 1D loop or (N, M) in the case of
                an MD loop.

        The setpoints are internally reshaped to (N, M) to be natively
        compatible with M-dimensional loops.

        """
        sp_shape = np.shape(setpoints)
        if len(sp_shape) == 1:
            setpoints = setpoints.reshape((len(setpoints), 1))
        self._setpoints = setpoints

    def set_setpoints_2D(self):
        pass

    def set_getpars(self, getable_par):
        """
        Define the parameters to be acquired during the acquisition loop.

        Args:
            getable_pars: parameter(s) to be get during the acquisition loop.
                accepts:
                - list or tuple of multiple getable objects
                - a single getable object.
            A getable object is an object that has at least
                - unit (str) attribute
                - name (str) attribute
                - get method
                See also `is_getable`

        TODO: support fancier getables, i.e. ones that return
            - more than one quantity
            - multiple points at once (hard loop)

        """
        if is_getable(getable_par):
            self._getable_pars = [getable_par]


def is_setable(setable):
    """
    Test if an object is a valid setable.

    Args:
        setable (object)

    Return:
        is_setable (bool)

    TODO: Add desicription of what a valid setable is. (unit, get etc.)
    """
    if not hasattr(setable, 'set'):
        raise AttributeError("{} does not have 'set'.".format(setable))
    if not hasattr(setable, 'name'):
        raise AttributeError("{} does not have 'name'.".format(setable))
    if not hasattr(setable, 'unit'):
        raise AttributeError("{} does not have 'unit'.".format(setable))

    return True


def is_getable(getable):
    """
    Test if an object is a valid getable.

    Args:
        getable (object)

    Return:
        is_getable (bool)

    TODO: Add desicription of what a valid getable is. (unit, get etc.)
    """
    if not hasattr(getable, 'get'):
        raise AttributeError("{} does not have 'get'.".format(getable))
    if not hasattr(getable, 'name'):
        raise AttributeError("{} does not have 'name'.".format(getable))
    if not hasattr(getable, 'unit'):
        raise AttributeError("{} does not have 'unit'.".format(getable))

    return True
