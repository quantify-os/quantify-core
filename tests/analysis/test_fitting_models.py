from pytest import approx
import quantify.data.handling as dh
from quantify.analysis import fitting_models as fm
from quantify.utilities._tests_helpers import get_test_data_dir
import numpy as np


def test_phase_guess():
    dh.set_datadir(get_test_data_dir())
    tuid_list = dh.get_tuids_containing(
        "Resonator_id", t_start="20210305", t_stop="20210306"
    )
    real_phi_vs = [-3.7774e-07, -3.7619e-07, -3.7742e-07, -3.7251e-07]

    _ = fm.ResonatorModel()

    # Go through all the test datasets
    for idx, _ in enumerate(tuid_list):
        dataset = dh.load_dataset(tuid=tuid_list[idx])
        freq = np.array(dataset["x0"])
        S21 = np.array(
            dataset["y0"] * np.cos(np.deg2rad(dataset["y1"]))
            + 1j * dataset["y0"] * np.sin(np.deg2rad(dataset["y1"]))
        )
        (_, phi_v) = fm.phase_guess(S21, freq)

        # We allow a certain tolerance on the accuracy of the guess, as this is only an intial input for our fit
        guess_tolerance = 0.3

        assert phi_v == approx(real_phi_vs[idx], rel=guess_tolerance)
