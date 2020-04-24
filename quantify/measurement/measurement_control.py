from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals


class MeasurementControl(Instrument):
    """
    Instrument responsible for controling the data acquisition loop.

    MeasurementControl (MC) is based on the notion that every experiment
    consists of the following step.

        1. Set some parameter(s)
        2. Measure some other parameter(s)
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
            name: str,
            datadir: str = ''):  # verbose: bool = True
        """
        Creates an instance of the Measurement Control.

        Args:
            name (str): name
            datadir (str): directory where datafiles are stored.
        """
        super().__init__(name=name)

        self.add_parameter(
            "datadir",
            initial_value=datadir,
            vals=vals.Strings(),
            parameter_class=ManualParameter,
        )

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
