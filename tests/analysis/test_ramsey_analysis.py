# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from pytest import approx
from uncertainties.core import Variable

import quantify.data.handling as dh
from quantify.analysis import ramsey_analysis as ra


@pytest.fixture(scope="session", autouse=True)
def analysis_obj(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    a_obj = ra.RamseyAnalysis(
        tuid="20210422-104958-297-7d6034", artificial_detuning=250e3
    ).run()
    return a_obj


def test_figures_generated(analysis_obj):
    """test that the right figures get created"""
    assert set(analysis_obj.figs_mpl.keys()) == {
        "Ramsey_decay",
    }


def test_quantities_of_interest(analysis_obj):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis_obj.quantities_of_interest.keys()) == {
        "T2*",
        "fitted_detuning",
        "detuning",
        "fit_msg",
        "fit_res",
        "fit_success",
    }

    qois_values = {
        "T2*": 9.029460824594437e-06,
        "fitted_detuning": 260217.48366305148,
        "detuning": 10.217e3,
    }

    assert isinstance(analysis_obj.quantities_of_interest["T2*"], Variable)
    assert isinstance(analysis_obj.quantities_of_interest["fitted_detuning"], Variable)
    assert isinstance(analysis_obj.quantities_of_interest["detuning"], Variable)

    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert analysis_obj.quantities_of_interest["T2*"].nominal_value == approx(
        qois_values["T2*"],
        abs=5 * analysis_obj.quantities_of_interest["T2*"].std_dev,
    )
    assert analysis_obj.quantities_of_interest[
        "fitted_detuning"
    ].nominal_value == approx(
        qois_values["fitted_detuning"],
        abs=5 * analysis_obj.quantities_of_interest["fitted_detuning"].std_dev,
    )
    assert analysis_obj.quantities_of_interest["detuning"].nominal_value == approx(
        qois_values["detuning"],
        abs=5 * analysis_obj.quantities_of_interest["detuning"].std_dev,
    )
    assert analysis_obj.quantities_of_interest["fit_success"] is True


# Also test for the case where the user inputs a qubit frequency
@pytest.fixture(scope="session", autouse=True)
def analysis_obj_qubit_freq(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    a_obj = ra.RamseyAnalysis(
        tuid="20210422-104958-297-7d6034",
        artificial_detuning=250e3,
        qubit_frequency=4.7149e9,
    ).run()
    return a_obj


def test_figures_generated_qubit_freq_qubit_freq(analysis_obj_qubit_freq):
    """test that the right figures get created"""
    assert set(analysis_obj_qubit_freq.figs_mpl.keys()) == {
        "Ramsey_decay",
    }


def test_quantities_of_interest_qubit_freq(analysis_obj_qubit_freq):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis_obj_qubit_freq.quantities_of_interest.keys()) == {
        "T2*",
        "fitted_detuning",
        "detuning",
        "qubit_frequency",
        "fit_msg",
        "fit_res",
        "fit_success",
    }

    qois_values = {
        "T2*": 9.029460824594437e-06,
        "fitted_detuning": 260217.48366305148,
        "detuning": 10.217e3,
        "qubit_frequency": 4.7149e9,
    }

    assert isinstance(analysis_obj_qubit_freq.quantities_of_interest["T2*"], Variable)
    assert isinstance(
        analysis_obj_qubit_freq.quantities_of_interest["fitted_detuning"], Variable
    )
    assert isinstance(
        analysis_obj_qubit_freq.quantities_of_interest["detuning"], Variable
    )
    assert isinstance(
        analysis_obj_qubit_freq.quantities_of_interest["qubit_frequency"], Variable
    )

    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert analysis_obj_qubit_freq.quantities_of_interest[
        "T2*"
    ].nominal_value == approx(
        qois_values["T2*"],
        abs=5 * analysis_obj_qubit_freq.quantities_of_interest["T2*"].std_dev,
    )
    assert analysis_obj_qubit_freq.quantities_of_interest[
        "fitted_detuning"
    ].nominal_value == approx(
        qois_values["fitted_detuning"],
        abs=5
        * analysis_obj_qubit_freq.quantities_of_interest["fitted_detuning"].std_dev,
    )
    assert analysis_obj_qubit_freq.quantities_of_interest[
        "detuning"
    ].nominal_value == approx(
        qois_values["detuning"],
        abs=5 * analysis_obj_qubit_freq.quantities_of_interest["detuning"].std_dev,
    )
    assert analysis_obj_qubit_freq.quantities_of_interest[
        "qubit_frequency"
    ].nominal_value == approx(
        qois_values["qubit_frequency"],
        abs=5
        * analysis_obj_qubit_freq.quantities_of_interest["qubit_frequency"].std_dev,
    )
    assert analysis_obj_qubit_freq.quantities_of_interest["fit_success"] is True
