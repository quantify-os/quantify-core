"""
Contains function to generate most basic waveforms, basic means having a
few parameters and a straightforward translation to AWG amplitude, i.e.,
no knowledge of qubit parameters, channels, etc.

These functions are intened to be used to generate waveforms defined in the :mod:`.pulse_library`.


Examples of waveforms that are too advanced are flux pulses that require
knowledge of the flux sensitivity and interaction strengths and qubit
frequencies.
"""
import numpy as np


def square(t, amp):
    """
    A square pulse.
    """
    return amp*np.ones(len(t))


def drag(t,
         G_amp: float,
         D_amp: float,
         duration: float,
         nr_sigma: int = 3,
         phase: float = 0,
         subtract_offset: str = 'average'):
    '''
    All inputs are in s and Hz.
    phases are in degree.

    Args:
        t (:py:class:`numpy.ndarray`): times at which to evaluate the function
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
        subtract_offset (str):
            Instruction on how to subtract the offset in order to avoid jumps
            in the waveform due to the cut-off.
            'average': subtract the average of the first and last point.
            'first': subtract the value of the waveform at the first sample.
            'last': subtract the value of the waveform at the last sample.
            'none', None: don't subtract any offset.

    :returns:
        - rot_drag_wave (:py:class:`numpy.ndarray`) - complex waveform.

    '''

    mu = t[0] + duration/2

    sigma = duration/(2*nr_sigma)

    gauss_env = G_amp*np.exp(-(0.5 * ((t-mu)**2) / sigma**2))
    deriv_gauss_env = - D_amp * (t-mu)/(sigma**1) * gauss_env

    # Subtract offsets
    if subtract_offset.lower() == 'none' or subtract_offset is None:
        # Do not subtract offset
        pass
    elif subtract_offset.lower() == 'average':
        gauss_env -= (gauss_env[0]+gauss_env[-1])/2.
        deriv_gauss_env -= (deriv_gauss_env[0]+deriv_gauss_env[-1])/2.
    elif subtract_offset.lower() == 'first':
        gauss_env -= gauss_env[0]
        deriv_gauss_env -= deriv_gauss_env[0]
    elif subtract_offset.lower() == 'last':
        gauss_env -= gauss_env[-1]
        deriv_gauss_env -= deriv_gauss_env[-1]
    else:
        raise ValueError(
            'Unknown value "{}" for keyword argument subtract_offset".'.format(subtract_offset))

    # generate pulses
    drag_wave = gauss_env + 1j * deriv_gauss_env

    # Apply phase rotation
    rot_drag_wave = rotate_wave(drag_wave, phase=phase)

    return rot_drag_wave


def rotate_wave(wave, phase: float):
    """
    Rotate a wave in the complex plane.


    Parameters
    -------------
    wave : :py:class:`numpy.ndarray`
        complex waveform, real component corresponds to I, imag component to Q.
    phase : float
        rotation angle in degrees

    Returns
    -----------
    rot_wave : :class:`numpy.ndarray`
        rotated waveform.
    rot_Q : :class:`numpy.ndarray`
        rotated quadrature component of the waveform.

    """

    angle = np.deg2rad(phase)

    rot_I = np.cos(angle)*wave.real - np.sin(angle)*wave.imag
    rot_Q = np.sin(angle)*wave.real + np.cos(angle)*wave.imag
    return rot_I + 1j * rot_Q


def modulate_wave(t, wave, freq_mod):
    """
    Apply single sideband (SSB) modulation to a waveform.

    The frequency convention we adhere to is:

        freq_base + freq_mod = freq_signal

    Parameters
    ------------
    t : :py:class:`numpy.ndarray`
        times at which to determine the modulation.
    wave : :py:class:`numpy.ndarray`
        complex waveform, real component corresponds to I, imag component to Q.
    freq_mod: float
        modulation frequency in Hz.


    Returns
    -----------
    mod_wave : :py:class:`numpy.ndarray`
        modulated waveform.

    .. note::

        Pulse modulation is generally not included when specifying waveform envelopes
        as there are many hardware backends include this capability.

    """
    cos_mod = np.cos(2*np.pi*freq_mod*t)
    sin_mod = np.sin(2*np.pi*freq_mod*t)
    mod_I = cos_mod*wave.real + sin_mod*wave.imag
    mod_Q = - sin_mod*wave.real + cos_mod*wave.imag

    return mod_I + 1j*mod_Q
