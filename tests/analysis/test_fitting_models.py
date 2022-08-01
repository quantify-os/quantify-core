"""Tests for analysis fitting models"""
import numpy as np
import pytest

import quantify_core.data.handling as dh
from quantify_core.analysis import fitting_models as fm


def test_resonator_phase_guess(tmp_test_data_dir):
    """Test for resonator_phase_guess function"""
    dh.set_datadir(tmp_test_data_dir)
    tuid_list = dh.get_tuids_containing(
        "Resonator_id", t_start="20210305", t_stop="20210306"
    )
    real_phi_vs = [-3.7774e-07, -3.7619e-07, -3.7742e-07, -3.7251e-07]

    _ = fm.ResonatorModel()

    # Go through all the test datasets
    for idx, _ in enumerate(tuid_list):
        dataset = dh.load_dataset(tuid=tuid_list[idx])
        freq = np.array(dataset["x0"])
        s21 = np.array(
            dataset["y0"] * np.cos(np.deg2rad(dataset["y1"]))
            + 1j * dataset["y0"] * np.sin(np.deg2rad(dataset["y1"]))
        )
        (_, phi_v) = fm.resonator_phase_guess(s21, freq)

        # We allow a certain tolerance on the accuracy of the guess, as this is only an
        # initial input for our fit
        guess_tolerance = 0.3

        assert phi_v == pytest.approx(real_phi_vs[idx], rel=guess_tolerance)


def test_fft_freq_phase_guess(tmp_test_data_dir):
    """Test for fft_freq_phase_guess function"""
    dh.set_datadir(tmp_test_data_dir)
    tuid_list = ["20210419-153127-883-fa4508"]
    real_freqs = [1 / (2 * 498.8e-3)]

    _ = fm.ResonatorModel()

    # Go through all the test datasets
    for idx, _ in enumerate(tuid_list):
        dataset = dh.load_dataset(tuid=tuid_list[idx])
        time = np.array(dataset["x0"])
        magnitude = dataset["y0"]
        (freq_guess, _) = fm.fft_freq_phase_guess(magnitude, time)

        # We allow a certain tolerance on the accuracy of the guess, as this is only an
        # initial input for our fit
        guess_tolerance = 0.3

        assert freq_guess == pytest.approx(real_freqs[idx], rel=guess_tolerance)


def test_cosine_model():
    """
    Test for CosineModel guessing and fitting
    """

    # Generate some random cosine data
    x_data = np.linspace(0, 4, 100)

    test_freq = np.random.uniform(low=0.5, high=4, size=1)
    test_amp = np.random.uniform(low=1, high=4, size=1)
    test_phs = np.random.uniform(low=0, high=2 * np.pi, size=1)
    test_cos = test_amp * np.cos(2 * np.pi * test_freq * x_data + test_phs)
    test_noise = np.random.normal(loc=0, scale=0.1, size=len(x_data))

    y_data = test_cos + test_noise

    # Fit a cosine to it
    model = fm.CosineModel()
    guess = model.guess(data=y_data, x=x_data)
    fit = model.fit(data=y_data, x=x_data, params=guess)

    # Test guessing, freq and phs already tested for
    assert guess["offset"] == pytest.approx(np.average(y_data))
    assert guess["amplitude"] == pytest.approx((y_data.max() - y_data.min()) / 2)

    # Test fitting
    fit_tolerance = 0.1
    assert fit.best_values["offset"] == pytest.approx(0, abs=fit_tolerance)
    assert fit.best_values["amplitude"] == pytest.approx(test_amp, rel=fit_tolerance)
    assert fit.best_values["frequency"] == pytest.approx(test_freq, rel=fit_tolerance)
    assert fit.best_values["phase"] == pytest.approx(test_phs, rel=fit_tolerance)
