from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals


class MeasurementControl(Instrument):
    """
    Instrument responsible for controling the data acquisition loop.

    MeasurementControl (MC) is based on the notion that every experiment
    consists of the following step.

        1. Set some parameter(s)            (setables)
        2. Measure some other parameter(s)  (getables)
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

        self.add_parameter(
            "setables",
            docstring='Object to be set in data acquisition loop',
            set_cmd=self._set_setables,
            get_cmd=self._get_setables,
            )
        self.add_parameter(
            "getables",
            docstring='Object to be set in data acquisition loop',
            set_cmd=self._set_getable,
            get_cmd=self._get_getable)

        # variables
        self._setables = []
        self._setpoints = []
        self._getables = []

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

    def _set_setables(self, *setables):
        """
        Define the setables for the acquisition loop.

        Args:
            setables: something to be set in the data-acquisition loop.
                a valid setable has a "set" method and the
                an example of a setable is a qcodes Parameter.
        """
        self._setables = []
        for i, setable in enumerate(setables):
            if is_setable(setable):
                self._setables.append(setable)

    def _get_setables(self):
        return self._setables


    def set_setpoints(self):
        pass

    def set_setpoints_2D(self):
        pass

    def _set_getable(self, getable):
        if is_getable(getable):
            self._getable = getable

    def _get_getable(self):
        pass

    def _initialize_dataset(self):
        """
        Initializeempty dataset based on mode, setables, getables and setpoints
        """

        pass


def is_setable(setable):
    """Test if object is a valid setable."""
    if not hasattr(setable, 'set'):
        raise AttributeError
    if not hasattr(setable, 'name'):
        raise AttributeError
    if not hasattr(setable, 'unit'):
        raise AttributeError

    return True


def is_getable(getable):
    """Test if object is a valid getable."""
    if not hasattr(getable, 'get'):
        raise AttributeError
    if not hasattr(getable, 'name'):
        raise AttributeError
    if not hasattr(getable, 'unit'):
        raise AttributeError

    return True
