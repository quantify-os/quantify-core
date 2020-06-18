import numpy as np
import numpy.testing as npt
from quantify.sequencer.waveforms import square, square_IQ, drag, modulate_wave, rotate_wave


def test_square_wave():
    amped_sq = square(np.arange(50), 2.44)
    npt.assert_array_equal(amped_sq, np.linspace(2.44, 2.44, 50))

    amped_sq_iq = square_IQ(np.arange(20), 6.88)
    npt.assert_array_equal(amped_sq_iq[0], np.linspace(6.88, 6.88, 20))
    npt.assert_array_equal(amped_sq_iq[1], np.linspace(0, 0, 20))


def test_drag():
    pass  # todo by someone who knows what it should look like


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
