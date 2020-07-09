import numpy as np
import numpy.testing as npt
import pytest
from quantify.sequencer.waveforms import square, soft_square, drag, modulate_wave, rotate_wave


def test_square_wave():
    amped_sq = square(np.arange(50), 2.44)
    npt.assert_array_equal(amped_sq, np.linspace(2.44, 2.44, 50))

    amped_sq_iq = square(np.arange(20), 6.88)
    npt.assert_array_equal(amped_sq_iq.real, np.linspace(6.88, 6.88, 20))
    npt.assert_array_equal(amped_sq_iq.imag, np.linspace(0, 0, 20))


def test_drag_ns():
    duration = 20e-9
    nr_sigma = 3
    G_amp = 0.5
    D_amp = 1

    times = np.arange(0, duration, 1e-9)  # sampling rate set to 1 GSPs
    mu = times[0] + duration/2
    sigma = duration/(2*nr_sigma)
    gauss_env = G_amp*np.exp(-(0.5 * ((times-mu)**2) / sigma**2))
    deriv_gauss_env = D_amp * -1 * (times-mu)/(sigma**1) * gauss_env
    exp_waveform = gauss_env + 1j * deriv_gauss_env

    # quantify
    waveform = drag(times, G_amp=G_amp, D_amp=D_amp, duration=duration, nr_sigma=nr_sigma, subtract_offset='none')

    np.testing.assert_array_almost_equal(waveform, exp_waveform, decimal=3)
    assert pytest.approx(np.max(waveform), .5)

    with pytest.raises(ValueError):
        drag(times, 0.5, D_amp, duration, subtract_offset='bad!')

    waveform = drag(times, G_amp=G_amp, D_amp=D_amp, duration=duration, nr_sigma=nr_sigma, subtract_offset='average')
    exp_waveform.real -= np.mean([exp_waveform.real[0], exp_waveform.real[-1]])
    exp_waveform.imag -= np.mean([exp_waveform.imag[0], exp_waveform.imag[-1]])
    np.testing.assert_array_almost_equal(waveform, exp_waveform, decimal=3)


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
