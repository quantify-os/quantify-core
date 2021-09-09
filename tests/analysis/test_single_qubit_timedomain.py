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


tuid_list = ["20210322-205253-758-6689"]
t1_list = [1.07e-5]


@pytest.fixture(scope="session", autouse=True)
def analyses(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    set_datadir(tmp_test_data_dir)
    analyses = [T1Analysis(tuid=tuid).run() for tuid in tuid_list]

    return analyses


def test_figures_generated(analyses):
    """
    Test that the right figures get created.
    """
    for analysis in analyses:
        assert set(analysis.figs_mpl.keys()) == {
            "T1_decay",
        }


def test_quantities_of_interest(analyses):
    """
    Test that the fit returns the correct values
    """
    for analysis, t_1 in zip(analyses, t1_list):
        assert set(analysis.quantities_of_interest.keys()) == {
            "T1",
            "fit_msg",
            "fit_result",
            "fit_success",
        }

        assert isinstance(analysis.quantities_of_interest["T1"], Variable)
        # Tests that the fitted values are correct (to within 5 standard deviations)
        assert analysis.quantities_of_interest["T1"].nominal_value == approx(
            t_1, abs=5 * analysis.quantities_of_interest["T1"].std_dev
        )
