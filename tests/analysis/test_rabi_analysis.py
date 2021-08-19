# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from pytest import approx, warns
import xarray as xr
import numpy as np
from uncertainties.core import Variable

import quantify_core.data.handling as dh
from quantify_core.analysis import rabi_analysis as ra


@pytest.fixture(scope="session", autouse=True)
def analysis(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis = ra.RabiAnalysis(tuid="20210419-153127-883-fa4508").run()
    return analysis


def test_figures_generated(analysis):
    """test that the right figures get created"""
    assert set(analysis.figs_mpl.keys()) == {
        "Rabi_oscillation",
    }


def test_quantities_of_interest(analysis):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis.quantities_of_interest.keys()) == {
        "Pi-pulse amplitude",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    assert isinstance(analysis.quantities_of_interest["Pi-pulse amplitude"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert analysis.quantities_of_interest[
        "Pi-pulse amplitude"
    ].nominal_value == approx(
        498.8e-3,
        abs=5e-3,
    )
    assert analysis.quantities_of_interest["fit_success"] is True


def test_quantities_of_interest_negative_amp(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)

    dh.set_datadir(tmp_test_data_dir)
    x_data = np.linspace(-0.5, 0.5, 31)
    y_data = np.cos(x_data * 4 * np.pi) * 0.31 + 0.05
    # add some noise
    y_data += 0.05 * np.random.randn(len(x_data))

    x_array = xr.DataArray(
        x_data,
        name="x0",
        attrs={
            "batched": False,
            "long_name": "Qubit drive amp",
            "name": "drive_amp",
            "units": "V",
        },
    )
    y_array = xr.DataArray(
        y_data,
        name="y0",
        attrs={
            "batched": False,
            "long_name": "Signal level",
            "name": "sig",
            "units": "V",
        },
    )
    dataset = xr.Dataset(
        {"x0": x_array, "y0": y_array},
        attrs={
            "name": "Mock_Rabi_power_scan_bad_fit",
            "tuid": "20210424-191802-994-f16eb3",
        },
    )
    dataset = dataset.set_coords(["x0"])

    analysis = ra.RabiAnalysis(dataset=dataset).run()
    assert analysis.quantities_of_interest[
        "Pi-pulse amplitude"
    ].nominal_value == approx(
        250e-3,
        abs=5e-3,
    )


@pytest.fixture(scope="session", autouse=True)
def analysis_bad_fit(tmp_test_data_dir):
    """
    Used for (Rabi) Analysis class that gives the correct warning when a lmfit
    cannot find a good fit.
    """
    dh.set_datadir(tmp_test_data_dir)
    x_data = np.linspace(-0.5, 0.5, 100)
    y_data = np.cos(x_data * 4 * np.pi + np.pi / 2) * 0.1 + 0.05

    y_data = np.random.randn(len(x_data))
    x_array = xr.DataArray(
        x_data,
        name="x0",
        attrs={
            "batched": False,
            "long_name": "Qubit drive amp",
            "name": "drive_amp",
            "units": "V",
        },
    )
    y_array = xr.DataArray(
        y_data,
        name="y0",
        attrs={
            "batched": False,
            "long_name": "Signal level",
            "name": "sig",
            "units": "V",
        },
    )
    dataset = xr.Dataset(
        {"x0": x_array, "y0": y_array},
        attrs={
            "name": "Mock_Rabi_power_scan_bad_fit",
            "tuid": "20210424-191802-994-f16eb3",
        },
    )
    dataset = dataset.set_coords(["x0"])

    # this check is suppressed as it is not a reliable indicator for a bad fit.
    # with warns(
    #     UserWarning,
    #     match="lmfit could not find a good fit."
    #     " Fitted parameters may not be accurate.",
    # ):
    analysis = ra.RabiAnalysis(
        dataset=dataset, settings_overwrite={"mpl_fig_formats": []}
    ).run()

    return analysis


def test_figures_generated_bad_fit(analysis_bad_fit):
    """test that the right figures get created despite failed fit"""
    assert set(analysis_bad_fit.figs_mpl.keys()) == {
        "Rabi_oscillation",
    }


def test_quantities_of_interest_bad_fit(analysis_bad_fit):
    """Test that the quantities of interest exist for a bad fit"""
    assert set(analysis_bad_fit.quantities_of_interest.keys()) == {
        "Pi-pulse amplitude",
        "fit_msg",
        "fit_result",
        "fit_success",
    }


@pytest.mark.xfail(reason="known parser issue")
def test_quantities_of_interest_bad_fit_warning_raised(analysis_bad_fit):
    """Test that the quantities of interest exist for a bad fit"""
    assert set(analysis_bad_fit.quantities_of_interest.keys()) == {
        "Pi-pulse amplitude",
        "fit_msg",
        "fit_result",
        "fit_success",
    }
    assert analysis_bad_fit.quantities_of_interest["fit_success"] is False
    assert (
        analysis_bad_fit.quantities_of_interest["fit_msg"]
        == "Warning: lmfit could not find a\ngood fit. Fitted parameters"
        " may not\nbe accurate."
    )
