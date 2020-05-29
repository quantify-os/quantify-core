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


def drag(t,
         G_amp: float,
         D_amp: float,
         duration: float,
         nr_sigma: int = 4,
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
        - pulse_I (:py:class:`numpy.ndarray`) - in phase component of waveform
        - pulse_Q (:py:class:`numpy.ndarray`) - quadrature component of waveform

    '''

    mu = t[0] + duration/2
    sigma = duration/nr_sigma

    gauss_env = G_amp*np.exp(-(0.5 * ((t-mu)**2) / sigma**2))
    deriv_gauss_env = D_amp * -1 * (t-mu)/(sigma**1) * gauss_env

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
        raise ValueError('Unknown value "{}" for keyword argument '
                         '"subtract_offset".'.format(subtract_offset))

    # generate pulses
    G = gauss_env
    D = deriv_gauss_env

    # Apply phase rotation
    pulse_I, pulse_Q = rotate_wave(G, D, phase=phase)

    return pulse_I, pulse_Q


def rotate_wave(wave_I, wave_Q, phase: float):
    """
    Rotate a wave in the complex plane.


    Parameters
    -------------
    wave_I : (:py:class:`numpy.ndarray`)
        in phase component of waveform
    wave_Q : (:py:class:`numpy.ndarray`)
        quadrature component of waveform
    phase : float
        rotation angle in degrees

    Returns
    -----------
    rot_I : :class:`numpy.ndarray`
        rotated in phase component of waveform
    rot_Q : :class:`numpy.ndarray`
        rotated quadrature component of waveform

    """

    angle = np.deg2rad(phase)

    rot_I = np.cos(angle)*wave_I - np.sin(angle)*wave_Q
    rot_Q = np.sin(angle)*wave_I + np.cos(angle)*wave_Q
    return rot_I, rot_Q


def modulate_wave(t, wave_I, wave_Q, freq_mod):
    """
    Apply single sideband (SSB) modulation to a waveform.

    The frequency convention we adhere to is:

        freq_base + freq_mod = freq_signal

    Parameters
    ------------
    t : (:py:class:`numpy.ndarray`)
        times at which to determine the modulation
    wave_I : (:py:class:`numpy.ndarray`)
        in phase component of waveform
    wave_Q : (:py:class:`numpy.ndarray`)
        quadrature component of waveform
    freq_mod: float

    Returns
    -----------
    mod_I : (:py:class:`numpy.ndarray`)
        modulated in phase component of waveform
    mod_Q : (:py:class:`numpy.ndarray`)
        modulated quadrature component of waveform

    """
    cos_mod = np.cos(2*np.pi*freq_mod*t)
    sin_mod = np.sin(2*np.pi*freq_mod*t)
    mod_I = cos_mod*wave_I + sin_mod*wave_Q
    mod_Q = - sin_mod*wave_I + cos_mod*wave_Q

    return mod_I, mod_Q


