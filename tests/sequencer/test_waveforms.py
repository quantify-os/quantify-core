import numpy as np
import numpy.testing as npt
import pytest
from quantify.sequencer.waveforms import square, square_IQ, drag, modulate_wave, rotate_wave


def test_square_wave():
    amped_sq = square(np.arange(50), 2.44)
    npt.assert_array_equal(amped_sq, np.linspace(2.44, 2.44, 50))

    amped_sq_iq = square_IQ(np.arange(20), 6.88)
    npt.assert_array_equal(amped_sq_iq[0], np.linspace(6.88, 6.88, 20))
    npt.assert_array_equal(amped_sq_iq[1], np.linspace(0, 0, 20))


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
    formula = gauss + 1j * beta * gauss_deriv

    # quantify
    waveform = drag(np.arange(duration), 0.5, beta, duration, subtract_offset='none')

    with pytest.raises(ValueError):
        drag(np.arange(duration), 0.5, beta, duration, subtract_offset='bad!')

    import matplotlib.pyplot as plt
    plt.plot(np.arange(duration), formula)
    plt.plot(np.arange(duration), waveform[1])
    plt.legend(['formula', 'quantify'])
    #plt.show()

    np.testing.assert_array_almost_equal(waveform[1], np.real(formula), decimal=3)


def test_rotate_wave():

    I = np.ones(10)
    Q = np.zeros(10)

    Ir, Qr = rotate_wave(I, Q, 0)

    npt.assert_array_almost_equal(I, Ir)
    npt.assert_array_almost_equal(Q, Qr)

    Ir, Qr = rotate_wave(I, Q, 90)

    npt.assert_array_almost_equal(I, Qr)
    npt.assert_array_almost_equal(Q, -Ir)

    Ir, Qr = rotate_wave(I, Q, 180)

    npt.assert_array_almost_equal(I, -Ir)
    npt.assert_array_almost_equal(Q, -Qr)

    Ir, Qr = rotate_wave(I, Q, 360)

    npt.assert_array_almost_equal(I, Ir)
    npt.assert_array_almost_equal(Q, Qr)


def test_modulate():
    fs = 100
    f = 4
    t = np.arange(fs)
    I = np.sin(2 * np.pi * f * (t/fs))
    Q = np.sin(2 * np.pi * f * (t/fs) + (np.pi/2))
    mod_I, _ = modulate_wave(np.linspace(0, 1, fs), I, Q, 2)
    npt.assert_array_almost_equal(mod_I, np.sin(2 * np.pi * (f+2) * (t/fs)), decimal=1)

    _, mod_Q = modulate_wave(np.linspace(0, 1, fs), I, Q, -2)
    npt.assert_array_almost_equal(mod_Q, np.sin(2 * np.pi * (f-2) * (t/fs) + (np.pi/2)), decimal=1)
