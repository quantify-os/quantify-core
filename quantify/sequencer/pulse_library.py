"""
Library standard pulses for use with the quantify sequencer.
"""

from .types import Operation


class SquareModPulse(Operation):
    """
    """

    def __init__(self, amp: float, duration: float,
                 ch_I: str, ch_Q: str,
                 t0: float = 0):

        data = {}
        data['name'] = 'SquareModPulse'
        data['pulse_info'] = [{
            'wf_func': 'quantify.sequencer.waveforms.drag',
            'amp': amp, 'duration': duration,
            't0': t0,
            'channels': [ch_I, ch_Q]}]
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
    pass

    def __init__(self, G_amp: float, D_amp: float, phase: float,
                 duration: float, ch_I: str, ch_Q: str, t0: float = 0):
        """
        Args:
            G_amp (float):
                Amplitude of the Gaussian envelope.
            D_amp (float):
                Amplitude of the derivative component, the DRAG-pulse parameter.
            duration (float):
                Duration of the pulse in seconds.
            nr_sigma (int):
                After how many sigma the Gaussian is cut off.
            phase (float):
                Phase of the pulse in degrees.
            ch_I (str): channel for the in-phase component
            ch_Q (str): channel for the quadrature component

        .. note::

            For many hardware backends the I and Q channel have to be set
            in a pair.

        """

        data = {}
        data['name'] = "DRAG"
        data['pulse_info'] = [{
            'wf_func': 'quantify.sequencer.waveforms.drag',
            'G_amp': G_amp, 'D_amp': D_amp, 'duration': duration,
            'phase': phase, 'nr_sigma': 4,
            'channels': [ch_I, ch_Q], 't0': t0}]

        super().__init__(name=data['name'], data=data)

    # amp: float, sigma_length: float, nr_sigma: int=4,
    #             sampling_rate: float=2e8, axis: str='x', phase: float=0,
    #             phase_unit: str='deg',
    #             motzoi: float=0, delay: float=0,
