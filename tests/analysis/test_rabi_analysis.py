# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
from pathlib import Path
import pytest
from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import rabi_analysis as ra


@pytest.fixture(scope="session", autouse=True)
def analysis_obj(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    a_obj = ra.RabiAnalysis(tuid="20210419-153127-883-fa4508").run()
    return a_obj


def test_raw_data_not_in_processed_dataset(analysis_obj):
    """Check that the required data is in the dataset"""
    container = Path(dh.locate_experiment_container(analysis_obj.tuid))
    file_path = container / "analysis_RabiAnalysis" / "processed_dataset.hdf5"
    dataset = dh.load_dataset_from_path(file_path)

    assert "x0" in dataset.dims.keys()
    assert "x0" in dataset.coords.keys()
    assert "y0" not in dataset.data_vars.keys()
    assert "y1" not in dataset.data_vars.keys()
    assert "Magnitude" in dataset.data_vars.keys()


def test_figures_generated(analysis_obj):
    """test that the right figures get created"""
    assert set(analysis_obj.figs_mpl.keys()) == {
        "Rabi_oscillation",
    }


def test_quantities_of_interest(analysis_obj):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis_obj.quantities_of_interest.keys()) == {
        "Pi-pulse amp",
        "fit_msg",
        "fit_res",
    }

    qois_values = {"Pi-pulse amp": 498.8e-3}
    assert isinstance(analysis_obj.quantities_of_interest["Pi-pulse amp"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert analysis_obj.quantities_of_interest["Pi-pulse amp"].nominal_value == approx(
        qois_values["Pi-pulse amp"],
        abs=5 * analysis_obj.quantities_of_interest["Pi-pulse amp"].std_dev,
    )
