import numpy as np
import xarray as xr
from qcodes import Instrument
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
            datadir (str): directory where datafiles are stored.
        """
        super().__init__(name=name)

        self.add_parameter(
            "datadir",
            initial_value='',
            vals=vals.Strings(),
            parameter_class=ManualParameter,
        )

        # variables
        self._setable_pars = []
        self._setpoints = []
        self._getable_pars = []

    ############################################
    # Methods used to control the measurements #
    ############################################

    def run(self, name: str = '',
            exp_metadata: dict = None,
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
        pass

    ############################################
    # Methods used to control the measurements #
    ############################################

    def set_setpars(self, setable_pars):
        """
        Define the setable_pars for the acquisition loop.

        Args:
            setable_pars: something to be set in the data-acquisition loop.
                a valid setable has a "set" method and the
                an example of a setable is a qcodes Parameter.
        """
        self._setable_pars = []
        for i, setable in enumerate(setable_pars):
            if is_setable(setable):
                self._setable_pars.append(setable)

    def _initialize_dataset(self):
        """
        Initialize an empty dataset based on
            mode, setables, getable_pars and _setable_pars

        """
        darrs = []
        for i, setpar in enumerate(self._setable_pars):
            darrs.append(xr.DataArray(
                data=self._setpoints[:, i],
                name='x{}'.format(i),
                attrs={'name': setpar.name, 'long_name': setpar.label,
                       'unit': setpar.unit}))

        numpoints = len(self._setpoints[:, 0])
        for j, getpar in enumerate(self._getable_pars):
            darrs.append(xr.DataArray(
                data=np.zeros(numpoints),
                name='y{}'.format(i),
                attrs={'name': getpar.name, 'long_name': getpar.label,
                       'unit': getpar.unit}))

        self._dataset = xr.merge(darrs)

    ####################################
    # Non-parameter get/set functions  #
    ####################################

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
        if is_getable(getable_par):
            self._getable_pars = [getable_par]


def is_setable(setable):
    """Test if object is a valid setable."""
    if not hasattr(setable, 'set'):
        raise AttributeError("{} does not have 'set'.".format(setable))
    if not hasattr(setable, 'name'):
        raise AttributeError("{} does not have 'name'.".format(setable))
    if not hasattr(setable, 'unit'):
        raise AttributeError("{} does not have 'unit'.".format(setable))

    return True


def is_getable(getable):
    """Test if object is a valid getable."""
    if not hasattr(getable, 'get'):
        raise AttributeError("{} does not have 'get'.".format(getable))
    if not hasattr(getable, 'name'):
        raise AttributeError("{} does not have 'name'.".format(getable))
    if not hasattr(getable, 'unit'):
        raise AttributeError("{} does not have 'unit'.".format(getable))

    return True
