# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from pytest import approx, warns
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import rabi_analysis as ra


@pytest.fixture(scope="session", autouse=True)
def analysis_obj(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    a_obj = ra.RabiAnalysis(tuid="20210419-153127-883-fa4508").run()
    return a_obj


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
        "fit_success",
    }

    qois_values = {"Pi-pulse amp": 498.8e-3}
    assert isinstance(analysis_obj.quantities_of_interest["Pi-pulse amp"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert analysis_obj.quantities_of_interest["Pi-pulse amp"].nominal_value == approx(
        qois_values["Pi-pulse amp"],
        abs=5 * analysis_obj.quantities_of_interest["Pi-pulse amp"].std_dev,
    )
    assert analysis_obj.quantities_of_interest["fit_success"] is True


@pytest.fixture(scope="session", autouse=True)
def analysis_obj_bad_fit(tmp_test_data_dir):
    """
    Used for (Rabi) Analysis class that gives the correct warning when a lmfit
    cannot find a good fit.
    """
    dh.set_datadir(tmp_test_data_dir)
    with warns(
        UserWarning,
        match="lmfit could not find a good fit."
        " Fitted parameters may not be accurate.",
    ):
        a_obj = ra.RabiAnalysis(tuid="20210424-191802-994-f16eb3").run()
    return a_obj


def test_figures_generated_bad_fit(analysis_obj_bad_fit):
    """test that the right figures get created despite failed fit"""
    assert set(analysis_obj_bad_fit.figs_mpl.keys()) == {
        "Rabi_oscillation",
    }


def test_quantities_of_interest_bad_fit(analysis_obj_bad_fit):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis_obj_bad_fit.quantities_of_interest.keys()) == {
        "Pi-pulse amp",
        "fit_msg",
        "fit_res",
        "fit_success",
    }

    assert analysis_obj_bad_fit.quantities_of_interest["fit_success"] is False
    assert (
        analysis_obj_bad_fit.quantities_of_interest["fit_msg"]
        == "Warning: lmfit could not find a good fit. Fitted parameters"
        " may not be accurate."
    )
