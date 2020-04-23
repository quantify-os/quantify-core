from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals


class MeasurementControl(Instrument):
    """
    Instrument responsible for controling the data acquisition loop.

    The MeasurementControl is based on the notion that every experiment
    consists of the following step.

        1. Set some parameter(s)
        2. Measure some other parameter(s)
        3. Store the data.

    The MeasurementControl

        - Enforces structure.
        - Standardizes data storage.
        - Providesl live vizualization.
        - Supports "advanced" experiments
            - Software controlled loops (soft)
            - Hardware con

    """

    def __init__(
            self,
            name: str,
            plotting_interval: float = 3,
            datadir: str = '',
            verbose: bool = True):
        super().__init__(name=name)

        self.add_parameter(
            "datadir",
            initial_value=datadir,
            vals=vals.Strings(),
            parameter_class=ManualParameter,
        )
