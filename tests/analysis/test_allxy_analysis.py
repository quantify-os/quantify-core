# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest

import quantify.data.handling as dh
from quantify.analysis import allxy_analysis as aa


@pytest.fixture(scope="session", autouse=True)
def analysis_obj(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    a_obj = aa.AllXYAnalysis(tuid="20210419-173649-456-23c5f3").run()
    return a_obj


def test_figures_generated(analysis_obj):
    """test that the right figures get created"""
    assert set(analysis_obj.figs_mpl.keys()) == {
        "AllXY",
    }


def test_quantities_of_interest(analysis_obj):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis_obj.quantities_of_interest.keys()) == {
        "deviation",
    }

    qois_values = {
        "deviation": 0.027,
    }

    assert isinstance(analysis_obj.quantities_of_interest["deviation"], float)

    # Tests that the fitted values are correct
    assert analysis_obj.quantities_of_interest["deviation"] == pytest.approx(
        qois_values["deviation"],
        rel=0.05,
    )


# Test that the analysis returns an error when the number of datapoints
# is not a multiple of 21
def test_analysis_obj_invalid_data(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    with pytest.raises(
        ValueError,
        match=(
            "Invalid dataset. The number of calibration points in an "
            "AllXY experiment must be a multiple of 21"
        ),
    ):
        aa.AllXYAnalysis(tuid="20210422-104958-297-7d6034").run()
