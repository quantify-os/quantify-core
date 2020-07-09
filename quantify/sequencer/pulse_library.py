"""
Library standard pulses for use with the quantify sequencer.
"""

from .types import Operation


class IdlePulse(Operation):
    """
    An idle pulse performing no actions for a certain duration.
    """

    def __init__(self, duration):
        """
        An idle pulse performing no actions for a certain duration.

        Parameters
        ------------
        duration : float
            Duration of the idling in seconds.
        """
        data = {'name': 'Idle', 'pulse_info': [{
            'wf_func': None,
            't0': 0,
            'duration': duration,
            'freq_mod': 0,
            'channel': None}]}
        super().__init__(name=data['name'], data=data)


class NumericPulse(Operation):
    """
    """

    def __init__(self, t0):
        raise NotImplementedError


class SquarePulse(Operation):

    def __init__(self, amp: float, duration: float, ch, t0: float = 0):
        """
        A single-channel square pulse.

        Parameters
        ------------
        amp : float
            Amplitude of the Gaussian envelope.
        duration : float
            Duration of the pulse in seconds.
        ch : str
            channel of the pulse.
        """

        data = {'name': 'SquarePulse', 'pulse_info': [{
            'wf_func': 'quantify.sequencer.waveforms.square',
            'amp': amp, 'duration': duration,
            't0': t0,
            'channel': ch}]}
        super().__init__(name=data['name'], data=data)


class ModSquarePulse(Operation):

    def __init__(self, amp: float, duration: float, ch: str, phase: float = 0, freq_mod: float = 0, t0: float = 0):
        """
        A two-channel square pulse.

        Parameters
        ------------
        amp : float
            Amplitude of the envelope.
        duration : float
            Duration of the pulse in seconds.
        ch : str
            channel of the pulse, must be capable of playing a complex waveform.
        phase : float
            Phase of the pulse in degrees.
        freq_mod :
            Modulation frequency in Hz.

        """

        data = {'name': 'ModSquarePulse', 'pulse_info': [{
            'wf_func': 'quantify.sequencer.waveforms.square',
            'amp': amp, 'duration': duration,
            't0': t0,
            'freq_mod': freq_mod,
            'channel': ch}]}
        super().__init__(name=data['name'], data=data)


class SoftSquarePulse(Operation):
    """
    Place holder pulse for mocking the CZ pulse until proper implementation. Replicates parameters.
    """

    def __init__(self, amp: float, duration: float, ch, t0: float = 0):
        data = {'name': 'SoftSquarePulse', 'pulse_info': [{
            'wf_func': 'quantify.sequencer.waveforms.soft_square',
            'amp': amp, 'duration': duration,
            't0': t0,
            'channel': ch}]}
        super().__init__(name=data['name'], data=data)


class DRAGPulse(Operation):
    """
    DRAG pulse inteded for single qubit gates in transmon based systems.

    A DRAG pulse is a gaussian pulse with a derivative component added to the out-of-phase channel to
    reduce unwanted excitations of the :math:`|1\\rangle - |2\\rangle` transition.


    The waveform is generated using :func:`.waveforms.drag` .

    References:
        1. |citation1|_

        .. _citation1: https://link.aps.org/doi/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).*

        2. |citation2|_

        .. _citation2: https://link.aps.org/doi/10.1103/PhysRevLett.103.110501

        .. |citation2| replace:: *F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
           Phys. Rev. Lett. 103, 110501 (2009).*
    """

    def __init__(self, G_amp: float, D_amp: float, phase: float, freq_mod: float,
                 duration: float, ch: str,
                 t0: float = 0):
        """
        Parameters
        ------------
        G_amp : float
            Amplitude of the Gaussian envelope.
        D_amp : float
            Amplitude of the derivative component, the DRAG-pulse parameter.
        duration : float
            Duration of the pulse in seconds.
        nr_sigma : int
            After how many sigma the Gaussian is cut off.
        phase : float
            Phase of the pulse in degrees.
        freq_mod :
            Modulation frequency in Hz.
        ch : str
            channel of the pulse, must be capable of playing a complex waveform.
        """

        data = {'name': "DRAG", 'pulse_info': [{
            'wf_func': 'quantify.sequencer.waveforms.drag',
            'G_amp': G_amp, 'D_amp': D_amp, 'duration': duration,
            'phase': phase, 'nr_sigma': 4, 'freq_mod': freq_mod,
            'channel': ch,  't0': t0}]}

        super().__init__(name=data['name'], data=data)
