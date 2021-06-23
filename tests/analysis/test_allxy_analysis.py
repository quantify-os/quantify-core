# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest

import quantify.data.handling as dh
from quantify.analysis import allxy_analysis as aa


@pytest.fixture(scope="session", autouse=True)
def analysis(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis = aa.AllXYAnalysis(tuid="20210419-173649-456-23c5f3").run()
    return analysis


def test_figures_generated(analysis):
    """test that the right figures get created"""
    assert set(analysis.figs_mpl.keys()) == {
        "AllXY",
    }


def test_quantities_of_interest(analysis):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis.quantities_of_interest.keys()) == {
        "deviation",
    }

    values = {
        "deviation": 0.027,
    }

    assert isinstance(analysis.quantities_of_interest["deviation"], float)

    # Tests that the fitted values are correct
    assert analysis.quantities_of_interest["deviation"] == pytest.approx(
        values["deviation"],
        rel=0.05,
    )


def test_dataset_processed(analysis):
    """some analysis results for the figure are stored in the processed dataset"""
    assert len(analysis.dataset_processed.experiment_numbers)
    assert len(analysis.dataset_processed.ideal_data)
    assert len(analysis.dataset_processed.normalized_data)


# Test that the analysis returns an error when the number of datapoints
# is not a multiple of 21
def test_analysis_invalid_data(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    with pytest.raises(
        ValueError,
        match=(
            "Invalid dataset. The number of calibration points in an "
            "AllXY experiment must be a multiple of 21"
        ),
    ):
        aa.AllXYAnalysis(tuid="20210422-104958-297-7d6034").run()
