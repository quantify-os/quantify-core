"""
Library containing common resources for use with the quantify sequencer.
"""

import numpy as np
from .types import Resource


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
                raise TypeError('resource_names "{}"must be strings'.format(resource_names))

        self.data = {'name': name,
                     'type': str(self.__class__.__name__),
                     'resources': resource_names}


class Pulsar_QCM_sequencer(Resource):
    """
    A single sequencer unit contained in a Pulsar_QCM module.
    """

    def __init__(self, name: str, instrument_name: str,
                 seq_idx: int, nco_freq: float = None, mod_enable: bool = False):
        """
        A channel composed of multiple sub-channels.

        Parameters
        -------------
        name : str
            the name of this resource



        """
        super().__init__()

        self.data = {'name': name,
                     'type': str(self.__class__.__name__),
                     'instrument_name': instrument_name,
                     'seq_idx': seq_idx,
                     'nco_freq': nco_freq,
                     'mod_enable': mod_enable}
