from pathlib import Path

from pytest import approx
import quantify.data.handling as dh
from quantify.analysis import fitting_models as fm
from quantify.utilities._tests_helpers import get_test_data_dir
import numpy as np


dh.set_datadir(get_test_data_dir())
tuid_list = dh.get_tuids_containing("resonator_analysis_test")


def test_phase_guess():
    mod = fm.ResonatorModel()
    dataset = dh.load_dataset(tuid=tuid_list[3])
    freq = np.array(dataset["x0"])
    S21 = np.array(dataset["y0"] * np.cos(np.deg2rad(dataset["y1"])) + 1j * dataset["y0"] * np.sin(np.deg2rad(dataset["y1"])))
    (phi_0, phi_v) = mod.phase_guess(S21, freq)

    # We allow a certain tolerance on the accuracy of the guess, as this is only an intial imput for our fit
    guess_tolerance = 0.2

    assert phi_v == approx(-3.7681e-07, rel=guess_tolerance) 
