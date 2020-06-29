"""
Library containing common resources for use with the quantify sequencer.
"""

import numpy as np
from .types import Resource


class QubitResource(Resource):
    """
    A qubit resource.
    """

    def __init__(self, name: str):
        super().__init__()

        self.data = {'name': name,
                     'type': str(self.__class__.__name__)}


class CompositeResource(Resource):
    """
    A channel composed of multiple resources.

    The compiler backend is responsible for using this resource to map
    operations to the relevant sub-channels.

    .. tip::

        A relevant use-case of this class is when making use of sequencer units in
        the Pulsar_QCM. The user can make specify this composite channel to
        play pulses, while the backend compiler ensures the pulses get distributed
        to the relevant sequencer resources.

    """

    def __init__(self, name: str, resource_names: list):
        """
        A channel composed of multiple sub-channels.

        Parameters
        -------------
        name : str
            the name of this resource
        resource_names : list
            a list of the resources referenced within this composite


        """
        super().__init__()
        for rn in resource_names:
            if not isinstance(rn, str):
                raise TypeError(
                    'resource_names "{}"must be strings'.format(resource_names))

        self.data = {'name': name,
                     'type': str(self.__class__.__name__),
                     'resources': resource_names}


class Pulsar_QCM_sequencer(Resource):
    """
    A single sequencer unit contained in a Pulsar_QCM module.

    For pulse-sequencing purposes, the Pulsar_QCM_sequencer can be considered
    a channel capabable of outputting complex valued signals (I, and Q).
    """

    def __init__(self, name: str, instrument_name: str,
                 seq_idx: int, nco_freq: float = None, mod_enable: bool = False):
        """
        A channel composed of multiple sub-channels.

        Parameters
        -------------
        name : str
            the name of this resource.
        instrument_name: str
            name of the Pulsar_QCM instrument.
        seq_idx: int
            index of the sequencer unit to use.
        nco_freq: float
            modulation frequency.
        mod_enable: bool
            enable pulse modulation.
        """
        super().__init__()

        self._timing_tuples = []
        self._pulse_dict = {}

        self.data = {'name': name,
                     'type': str(self.__class__.__name__),
                     'instrument_name': instrument_name,
                     'seq_idx': seq_idx,
                     'nco_freq': nco_freq,
                     'mod_enable': mod_enable}

    @property
    def timing_tuples(self):
        """
        A list of timing tuples con
        """
        return self._timing_tuples

    @property
    def pulse_dict(self):
        return self._pulse_dict

