# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from uncertainties.core import Variable, AffineScalarFunc

import quantify.data.handling as dh
from quantify.analysis import ramsey_analysis as ra


@pytest.fixture(scope="session", autouse=True)
def analysis(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis = ra.RamseyAnalysis(tuid="20210422-104958-297-7d6034").run(
        artificial_detuning=250e3
    )
    return analysis


def test_figures_generated(analysis):
    """test that the right figures get created"""
    assert set(analysis.figs_mpl.keys()) == {
        "Ramsey_decay",
    }


def test_quantities_of_interest(analysis):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis.quantities_of_interest.keys()) == {
        "T2*",
        "fitted_detuning",
        "detuning",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    values = {
        "T2*": 9.029460824594437e-06,
        "fitted_detuning": 260217.48366305148,
        "detuning": 10.217e3,
    }

    assert isinstance(analysis.quantities_of_interest["T2*"], Variable)
    assert isinstance(analysis.quantities_of_interest["fitted_detuning"], Variable)
    assert isinstance(analysis.quantities_of_interest["detuning"], AffineScalarFunc)

    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert analysis.quantities_of_interest["T2*"].nominal_value == pytest.approx(
        values["T2*"],
        abs=5 * analysis.quantities_of_interest["T2*"].std_dev,
    )
    assert analysis.quantities_of_interest[
        "fitted_detuning"
    ].nominal_value == pytest.approx(
        values["fitted_detuning"],
        abs=5 * analysis.quantities_of_interest["fitted_detuning"].std_dev,
    )
    assert analysis.quantities_of_interest["detuning"].nominal_value == pytest.approx(
        values["detuning"],
        abs=5 * analysis.quantities_of_interest["detuning"].std_dev,
    )
    assert analysis.quantities_of_interest["fit_success"] is True


# Also test for the case where the user inputs a qubit frequency
@pytest.fixture(scope="session", autouse=True)
def analysis_qubit_freq(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis = ra.RamseyAnalysis(tuid="20210422-104958-297-7d6034").run(
        artificial_detuning=250e3, qubit_frequency=4.7149e9
    )
    return analysis


def test_figures_generated_qubit_freq_qubit_freq(analysis_qubit_freq):
    """test that the right figures get created"""
    assert set(analysis_qubit_freq.figs_mpl.keys()) == {
        "Ramsey_decay",
    }


def test_quantities_of_interest_qubit_freq(analysis_qubit_freq):
    """Test that the quantities of interest have the correct values"""
    assert set(analysis_qubit_freq.quantities_of_interest.keys()) == {
        "T2*",
        "fitted_detuning",
        "detuning",
        "qubit_frequency",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    values = {
        "T2*": 9.029460824594437e-06,
        "fitted_detuning": 260217.48366305148,
        "detuning": 10.217e3,
        "qubit_frequency": 4.7149e9,
    }

    assert isinstance(analysis_qubit_freq.quantities_of_interest["T2*"], Variable)
    assert isinstance(
        analysis_qubit_freq.quantities_of_interest["fitted_detuning"], Variable
    )
    assert isinstance(
        analysis_qubit_freq.quantities_of_interest["detuning"], AffineScalarFunc
    )
    assert isinstance(
        analysis_qubit_freq.quantities_of_interest["qubit_frequency"],
        AffineScalarFunc,
    )

    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert analysis_qubit_freq.quantities_of_interest[
        "T2*"
    ].nominal_value == pytest.approx(
        values["T2*"],
        abs=5 * analysis_qubit_freq.quantities_of_interest["T2*"].std_dev,
    )
    assert analysis_qubit_freq.quantities_of_interest[
        "fitted_detuning"
    ].nominal_value == pytest.approx(
        values["fitted_detuning"],
        abs=5 * analysis_qubit_freq.quantities_of_interest["fitted_detuning"].std_dev,
    )
    assert analysis_qubit_freq.quantities_of_interest[
        "detuning"
    ].nominal_value == pytest.approx(
        values["detuning"],
        abs=5 * analysis_qubit_freq.quantities_of_interest["detuning"].std_dev,
    )
    assert analysis_qubit_freq.quantities_of_interest[
        "qubit_frequency"
    ].nominal_value == pytest.approx(
        values["qubit_frequency"],
        abs=5 * analysis_qubit_freq.quantities_of_interest["qubit_frequency"].std_dev,
    )
    assert analysis_qubit_freq.quantities_of_interest["fit_success"] is True
