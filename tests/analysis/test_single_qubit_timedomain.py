# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from pytest import approx
from uncertainties.core import Variable
from quantify_core.data.handling import set_datadir


import numpy as np
from quantify_core.analysis.single_qubit_timedomain import (
    rotate_to_calibrated_axis,
    T1Analysis,
)


def test_rotate_to_calibrated_axis():

    ref_val_0 = 0.24 + 324 * 1j
    ref_val_1 = 0.89 + 0.324 * 1j
    data = np.array([ref_val_0, ref_val_1])

    corrected_data = rotate_to_calibrated_axis(
        data, ref_val_0=ref_val_0, ref_val_1=ref_val_1
    )

    np.testing.assert_array_equal(corrected_data, np.array([0.0, 1.0]))


@pytest.fixture(scope="module", autouse=True)
def t1_analysis_no_cal_points(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    tuid = "20210322-205253-758-6689"
    set_datadir(tmp_test_data_dir)
    return T1Analysis(tuid=tuid).run(calibration_points=False)


def test_figures_generated(t1_analysis_no_cal_points):
    """
    Test that the right figures get created.
    """
    assert set(t1_analysis_no_cal_points.figs_mpl.keys()) == {
        "T1_decay",
    }


def test_quantities_of_interest(t1_analysis_no_cal_points):
    """
    Test that the fit returns the correct values
    """

    assert set(t1_analysis_no_cal_points.quantities_of_interest.keys()) == {
        "T1",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    exp_t1 = 1.07e-5
    assert isinstance(t1_analysis_no_cal_points.quantities_of_interest["T1"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert t1_analysis_no_cal_points.quantities_of_interest[
        "T1"
    ].nominal_value == approx(
        exp_t1, abs=5 * t1_analysis_no_cal_points.quantities_of_interest["T1"].std_dev
    )


@pytest.fixture(scope="module", autouse=True)
def t1_analysis_with_cal_points(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    tuid = "20210827-174946-357-70a986"
    set_datadir(tmp_test_data_dir)
    return T1Analysis(tuid=tuid).run(calibration_points=True)


def test_quantities_of_interest_cal_pts(t1_analysis_with_cal_points):
    """
    Test that the fit returns the correct values
    """

    assert set(t1_analysis_with_cal_points.quantities_of_interest.keys()) == {
        "T1",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    exp_t1 = 7.716e-6
    assert isinstance(
        t1_analysis_with_cal_points.quantities_of_interest["T1"], Variable
    )
    # Tests that the fitted values are correct (to within 5 standard deviations)
    meas_t1 = t1_analysis_with_cal_points.quantities_of_interest["T1"].nominal_value

    # accurate to < 1 %
    assert meas_t1 == approx(exp_t1, rel=0.01)
