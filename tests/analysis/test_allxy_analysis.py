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
