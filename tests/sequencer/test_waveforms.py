import numpy as np
import numpy.testing as npt
import pytest
from quantify.sequencer.waveforms import square, drag, modulate_wave, rotate_wave


def test_square_wave():
    amped_sq = square(np.arange(50), 2.44)
    npt.assert_array_equal(amped_sq, np.linspace(2.44, 2.44, 50))

    amped_sq_iq = square(np.arange(20), 6.88)
    npt.assert_array_equal(amped_sq_iq.real, np.linspace(6.88, 6.88, 20))
    npt.assert_array_equal(amped_sq_iq.imag, np.linspace(0, 0, 20))


def test_drag():
    duration = 25
    sigma = 4
    amp = 0.5j
    beta = 1
    # formulaic
    times = np.arange(duration)
    times = times - (duration / 2) + times[0]
    gauss = amp * np.exp(-(times / sigma) ** 2 / 2)
    gauss_deriv = -(times / sigma ** 2) * gauss
    formula_der_comp = gauss + 1j * beta * gauss_deriv

    # quantify
    waveform = drag(np.arange(duration), 0.5, beta, duration, subtract_offset='none')

    with pytest.raises(ValueError):
        drag(np.arange(duration), 0.5, beta, duration, subtract_offset='bad!')

    np.testing.assert_array_almost_equal(waveform.imag, np.real(formula_der_comp), decimal=3)



def test_rotate_wave():

    I = np.ones(10)  # Q component is zero
    Q = np.zeros(10) # not used as input, only used for testing

    rot_wf = rotate_wave(I, 0)

    npt.assert_array_almost_equal(I, rot_wf.real)
    npt.assert_array_almost_equal(I.imag, rot_wf.imag)

    rot_wf = rotate_wave(I, 90)

    npt.assert_array_almost_equal(I, rot_wf.imag)
    npt.assert_array_almost_equal(Q, -rot_wf.real)

    rot_wf = rotate_wave(I, 180)

    npt.assert_array_almost_equal(I, -rot_wf.real)
    npt.assert_array_almost_equal(Q, -rot_wf.imag)

    rot_wf = rotate_wave(I, 360)

    npt.assert_array_almost_equal(I, rot_wf.real)
    npt.assert_array_almost_equal(Q, rot_wf.imag)


def test_modulate():
    fs = 100
    f = 4
    t = np.arange(fs)
    I = np.sin(2 * np.pi * f * (t/fs))
    Q = np.sin(2 * np.pi * f * (t/fs) + (np.pi/2))
    wf = I + 1j*Q

    mod_wf = modulate_wave(np.linspace(0, 1, fs), wf, 2)
    npt.assert_array_almost_equal(mod_wf.real, np.sin(2 * np.pi * (f+2) * (t/fs)), decimal=1)

    mod_wf = modulate_wave(np.linspace(0, 1, fs), wf, -2)
    npt.assert_array_almost_equal(mod_wf.imag, np.sin(2 * np.pi * (f-2) * (t/fs) + (np.pi/2)), decimal=1)
