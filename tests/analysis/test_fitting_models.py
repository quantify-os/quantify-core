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
	dataset = dh.load_dataset(tuid=tuid_list[2])
	freq = np.array(dataset["x0"])
	S21 = np.array(dataset["y0"] * np.cos(np.deg2rad(dataset["y1"])) + 1j * dataset["y0"] * np.sin(np.deg2rad(dataset["y1"])))
	(phi_0, phi_v) = mod.phase_guess(S21, freq)

	assert (phi_0 + np.pi)%2*np.pi == approx((354.631574 + np.pi)%2*np.pi)
	assert phi_v == approx(-3.7840e-07) 
