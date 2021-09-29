# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file

import pytest
import numpy as np
from quantify_core.data.handling import set_datadir
from quantify_core.analysis.calibration import (
    rotate_to_calibrated_axis,
    has_calibration_points,
)
from quantify_core.analysis.single_qubit_timedomain import SingleQubitTimedomainAnalysis
from quantify_core.utilities.examples_support import mk_iq_shots


@pytest.fixture(scope="module", name="angles")
def fixture_angles():
    return np.arange(0, 2 * np.pi, 2 * np.pi / 17)


def test_rotate_to_calibrated_axis():

    ref_val_0 = 0.24 + 324 * 1j
    ref_val_1 = 0.89 + 0.324 * 1j
    data = np.array([ref_val_0, ref_val_1])

    corrected_data = np.real(
        rotate_to_calibrated_axis(data, ref_val_0=ref_val_0, ref_val_1=ref_val_1)
    )

    np.testing.assert_array_equal(corrected_data, np.array([0.0, 1.0]))


def test_has_calibration_points_points_outside_centers(angles):
    probabilities = np.array([0] * 20 + list(np.linspace(0, 1, 30)) + [1] * 20)
    center_0 = 0.6 + 1.2j
    center_1 = -0.2 + 0.5j
    data = np.array(
        tuple(
            mk_iq_shots(
                50,
                sigmas=[0.3] * 2,
                centers=[center_0, center_1],
                probabilities=[prob, 1 - prob],
            ).mean()
            for prob in probabilities
        )
    )

    for points, has_cal in zip(
        [data, np.concatenate((data, [center_0, center_1]))], [False, True]
    ):
        for angle in angles:
            assert has_calibration_points(points * np.exp(1j * angle)) == has_cal


def test_has_calibration_points_datasets(tmp_test_data_dir, angles):
    set_datadir(tmp_test_data_dir)
    tuids = [
        "20210322-205253-758-6689ca-T1 at 4.715GHz",
        "20210827-174946-357-70a986-T1 experiment q0",
        "20210422-104958-297-7d6034-Ramsey oscillation at 4.715GHz",
        "20210827-175004-087-ab1aab-Ramsey oscillation q0 at 4.9969 GHz",
        "20210901-132357-561-5c3ef7-Ramsey oscillation q0 at 6.1400 GHz",
        "20210827-175021-521-251f28-Echo experiment q0",
        "20210420-001339-580-97bdef-Echo at 4.715GHz",
    ]

    for tuid, has_cal in zip(tuids, [False, True, False, True, True, True, False]):
        a_obj = SingleQubitTimedomainAnalysis(tuid=tuid).run_until(
            interrupt_before="run_fitting",  # avoid writing to disk
            calibration_points="auto",
        )
        # test many rotations on IQ plane
        for angle in angles:
            assert (
                has_calibration_points(a_obj.dataset_processed.S21 * np.exp(1j * angle))
                == has_cal
            )
