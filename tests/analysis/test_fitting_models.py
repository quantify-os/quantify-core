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
