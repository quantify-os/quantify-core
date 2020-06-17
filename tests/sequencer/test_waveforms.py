import numpy as np
import numpy.testing as npt
from quantify.sequencer.waveforms import drag, modulate_wave, rotate_wave


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
