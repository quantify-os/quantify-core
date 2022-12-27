# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import numpy as np
import pytest
import xarray as xr
from pytest import approx
from uncertainties.core import AffineScalarFunc, Variable

from quantify_core.analysis.single_qubit_timedomain import (
    AllXYAnalysis,
    EchoAnalysis,
    RabiAnalysis,
    RamseyAnalysis,
    T1Analysis,
)
from quantify_core.data.handling import set_datadir


@pytest.fixture(scope="module", autouse=True)
def t1_analysis_no_cal_points(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    tuid = "20210322-205253-758-6689"
    set_datadir(tmp_test_data_dir)
    return T1Analysis(tuid=tuid).run()


def test_t1_load_fit_results(t1_analysis_no_cal_points, tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    for fit_name, fit_result in t1_analysis_no_cal_points.fit_results.items():

        loaded_fit_result = T1Analysis.load_fit_result(
            tuid=t1_analysis_no_cal_points.tuid, fit_name=fit_name
        )

        assert loaded_fit_result.params == fit_result.params


def test_t1_figures_generated(t1_analysis_no_cal_points):
    """
    Test that the right figures get created.
    """
    assert set(t1_analysis_no_cal_points.figs_mpl.keys()) == {
        "T1_decay",
    }


def test_t1_quantities_of_interest(t1_analysis_no_cal_points):
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


def test_t1_analysis_with_cal_points(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    tuid = "20210827-174946-357-70a986"
    set_datadir(tmp_test_data_dir)
    analysis_obj = T1Analysis(tuid=tuid).run()

    assert set(analysis_obj.quantities_of_interest.keys()) == {
        "T1",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    exp_t1 = 7.716e-6
    assert isinstance(analysis_obj.quantities_of_interest["T1"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    meas_t1 = analysis_obj.quantities_of_interest["T1"].nominal_value

    # accurate to < 1 %
    assert meas_t1 == approx(exp_t1, rel=0.01)


def test_echo_analysis_no_cal(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)

    analysis_obj = EchoAnalysis(tuid="20210420-001339-580-97bdef").run()
    assert set(analysis_obj.figs_mpl.keys()) == {
        "Echo_decay",
    }

    exp_t2_echo = 10.00e-6
    assert set(analysis_obj.quantities_of_interest.keys()) == {
        "t2_echo",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    assert isinstance(analysis_obj.quantities_of_interest["t2_echo"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    meas_echo = analysis_obj.quantities_of_interest["t2_echo"].nominal_value

    # accurate to < 1 %
    assert meas_echo == approx(exp_t2_echo, rel=0.01)

    # Test loading and saving fit result object
    for fit_name, fit_result in analysis_obj.fit_results.items():

        loaded_fit_result = EchoAnalysis.load_fit_result(
            tuid=analysis_obj.tuid, fit_name=fit_name
        )

        assert loaded_fit_result.params == fit_result.params


def test_echo_analysis_with_cal(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)

    analysis_obj = EchoAnalysis(tuid="20210827-175021-521-251f28").run()
    assert set(analysis_obj.figs_mpl.keys()) == {
        "Echo_decay",
    }

    exp_t2_echo = 13.61e-6
    assert set(analysis_obj.quantities_of_interest.keys()) == {
        "t2_echo",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    assert isinstance(analysis_obj.quantities_of_interest["t2_echo"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    meas_echo = analysis_obj.quantities_of_interest["t2_echo"].nominal_value

    # accurate to < 1 %
    assert meas_echo == approx(exp_t2_echo, rel=0.01)


def test_ramsey_no_cal_generated(tmp_test_data_dir):
    """test that the right figures get created"""
    set_datadir(tmp_test_data_dir)
    analysis = RamseyAnalysis(tuid="20210422-104958-297-7d6034").run(
        artificial_detuning=250e3
    )
    assert set(analysis.figs_mpl.keys()) == {
        "Ramsey_decay",
    }

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
@pytest.fixture(scope="module", autouse=True)
def ramsey_analysis_qubit_freq(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    analysis = RamseyAnalysis(tuid="20210422-104958-297-7d6034").run(
        artificial_detuning=250e3,
        qubit_frequency=4.7149e9,
    )
    return analysis


def test_ramsey_load_fit_results(ramsey_analysis_qubit_freq, tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    for fit_name, fit_result in ramsey_analysis_qubit_freq.fit_results.items():

        loaded_fit_result = RamseyAnalysis.load_fit_result(
            tuid=ramsey_analysis_qubit_freq.tuid, fit_name=fit_name
        )

        assert loaded_fit_result.params == fit_result.params


def test_figures_generated_qubit_freq_qubit_freq(ramsey_analysis_qubit_freq):
    """test that the right figures get created"""
    assert set(ramsey_analysis_qubit_freq.figs_mpl.keys()) == {
        "Ramsey_decay",
    }


def test_quantities_of_interest_qubit_freq(ramsey_analysis_qubit_freq):
    """Test that the quantities of interest have the correct values"""
    assert set(ramsey_analysis_qubit_freq.quantities_of_interest.keys()) == {
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

    assert isinstance(
        ramsey_analysis_qubit_freq.quantities_of_interest["T2*"], Variable
    )
    assert isinstance(
        ramsey_analysis_qubit_freq.quantities_of_interest["fitted_detuning"], Variable
    )
    assert isinstance(
        ramsey_analysis_qubit_freq.quantities_of_interest["detuning"], AffineScalarFunc
    )
    assert isinstance(
        ramsey_analysis_qubit_freq.quantities_of_interest["qubit_frequency"],
        AffineScalarFunc,
    )

    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert ramsey_analysis_qubit_freq.quantities_of_interest[
        "T2*"
    ].nominal_value == pytest.approx(
        values["T2*"],
        rel=0.01,
    )
    assert ramsey_analysis_qubit_freq.quantities_of_interest[
        "fitted_detuning"
    ].nominal_value == pytest.approx(
        values["fitted_detuning"],
        rel=0.01,
    )
    assert ramsey_analysis_qubit_freq.quantities_of_interest[
        "detuning"
    ].nominal_value == pytest.approx(
        values["detuning"],
        rel=0.01,
    )
    assert ramsey_analysis_qubit_freq.quantities_of_interest[
        "qubit_frequency"
    ].nominal_value == pytest.approx(
        values["qubit_frequency"],
        rel=0.01,
    )
    assert ramsey_analysis_qubit_freq.quantities_of_interest["fit_success"] is True


def test_ramsey_analysis_with_cal(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)

    analysis_obj = RamseyAnalysis(tuid="20210827-175004-087-ab1aab").run()
    assert set(analysis_obj.figs_mpl.keys()) == {
        "Ramsey_decay",
    }

    exp_t2_ramsey = 10.43e-06
    assert set(
        {"T2*", "detuning", "fit_msg", "fit_result", "fit_success", "fitted_detuning"}
    ) == set(analysis_obj.quantities_of_interest.keys())

    assert isinstance(analysis_obj.quantities_of_interest["T2*"], Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    meas_t2_ramsey = analysis_obj.quantities_of_interest["T2*"].nominal_value

    # accurate to < 1 %
    assert meas_t2_ramsey == approx(exp_t2_ramsey, rel=0.01)
    meas_detuning = analysis_obj.quantities_of_interest["detuning"].nominal_value
    assert meas_detuning == approx(166557, rel=0.01)


def test_ramsey_analysis_with_cal_qubit_freq_reporting(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    tuid = "20210901-132357-561-5c3ef7"
    qubit_frequency = 6140002015.621445

    a_obj = RamseyAnalysis(tuid=tuid)
    a_obj.run(artificial_detuning=150e3, qubit_frequency=qubit_frequency)

    exp_t2_ramsey = 7.239e-6
    exp_detuning = -244.65
    exp_fitted_detuning = 149609
    exp_qubit_frequency = 6.140002406e9

    t2_ramsey = a_obj.quantities_of_interest["T2*"].nominal_value
    detuning = a_obj.quantities_of_interest["detuning"].nominal_value
    fitted_detuning = a_obj.quantities_of_interest["fitted_detuning"].nominal_value
    qubit_frequency = a_obj.quantities_of_interest["qubit_frequency"].nominal_value

    assert t2_ramsey == approx(exp_t2_ramsey, rel=0.01)
    assert detuning == approx(exp_detuning, rel=0.01)
    assert fitted_detuning == approx(exp_fitted_detuning, rel=0.01)
    assert qubit_frequency == approx(exp_qubit_frequency, rel=0.01)


@pytest.fixture(scope="session", autouse=True)
def allxy_analysis_obj(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    allxy_analysis_obj = AllXYAnalysis(tuid="20210419-173649-456-23c5f3").run()
    return allxy_analysis_obj


def test_allxy_figures_generated(allxy_analysis_obj):
    """test that the right figures get created"""
    assert set(allxy_analysis_obj.figs_mpl.keys()) == {
        "AllXY",
    }


def test_allxy_quantities_of_interest(allxy_analysis_obj):
    """Test that the quantities of interest have the correct values"""
    assert set(allxy_analysis_obj.quantities_of_interest.keys()) == {
        "deviation",
    }

    exp_deviation = 0.022

    assert isinstance(allxy_analysis_obj.quantities_of_interest["deviation"], float)

    # Tests that the fitted values are correct
    assert allxy_analysis_obj.quantities_of_interest["deviation"] == pytest.approx(
        exp_deviation,
        rel=0.01,
    )


def test_allxy_dataset_processed(allxy_analysis_obj):
    assert len(allxy_analysis_obj.dataset_processed.ideal_data)
    assert len(allxy_analysis_obj.dataset_processed.pop_exc)


# Test that the analysis returns an error when the number of datapoints
# is not a multiple of 21
def test_allxy_analysis_invalid_data(caplog, tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    analysis = AllXYAnalysis(tuid="20210422-104958-297-7d6034").run()
    assert isinstance(analysis, AllXYAnalysis)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert (
        'Exception was raised while executing analysis step 1 ("<bound method '
        "AllXYAnalysis.process_data of "
        "<quantify_core.analysis.single_qubit_timedomain.AllXYAnalysis object at"
        in record.msg
    )
    exception = record.exc_info[1]
    assert isinstance(exception, ValueError)
    assert exception.args[0].startswith(
        "Invalid dataset. The number of calibration points"
    )


def test_allxy_load_fit_results_missing(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)

    with pytest.raises(
        FileNotFoundError, match="No fit results found for this analysis."
    ):
        AllXYAnalysis.load_fit_result(
            tuid="20210419-173649-456-23c5f3", fit_name="fit_name"
        )


@pytest.fixture(scope="session", autouse=True)
def rabi_analysis_obj(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    rabi_analysis_obj = RabiAnalysis(tuid="20210419-153127-883-fa4508").run()
    return rabi_analysis_obj


def test_rabi_load_fit_results(rabi_analysis_obj, tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    for fit_name, fit_result in rabi_analysis_obj.fit_results.items():

        loaded_fit_result = RabiAnalysis.load_fit_result(
            tuid=rabi_analysis_obj.tuid, fit_name=fit_name
        )

        assert loaded_fit_result.params == fit_result.params


def test_figures_generated(rabi_analysis_obj):
    """test that the right figures get created"""
    assert set(rabi_analysis_obj.figs_mpl.keys()) == {
        "Rabi_oscillation",
    }


def test_quantities_of_interest(rabi_analysis_obj):
    """Test that the quantities of interest have the correct values"""
    assert set(rabi_analysis_obj.quantities_of_interest.keys()) == {
        "Pi-pulse amplitude",
        "fit_msg",
        "fit_result",
        "fit_success",
    }

    assert isinstance(
        rabi_analysis_obj.quantities_of_interest["Pi-pulse amplitude"], Variable
    )
    # Tests that the fitted values are correct (to within 5 standard deviations)
    assert rabi_analysis_obj.quantities_of_interest[
        "Pi-pulse amplitude"
    ].nominal_value == approx(
        498.8e-3,
        abs=5e-3,
    )
    assert rabi_analysis_obj.quantities_of_interest["fit_success"] is True


def test_quantities_of_interest_negative_amp(tmp_test_data_dir):

    set_datadir(tmp_test_data_dir)
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

    analysis = RabiAnalysis(dataset=dataset).run()
    assert analysis.quantities_of_interest[
        "Pi-pulse amplitude"
    ].nominal_value == approx(
        250e-3,
        abs=10e-3,
    )


@pytest.fixture(scope="session", autouse=True)
def rabi_analysis_obj_bad_fit(tmp_test_data_dir):
    """
    Used for (Rabi) Analysis class that gives the correct warning when a lmfit
    cannot find a good fit.
    """
    set_datadir(tmp_test_data_dir)
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

    # this check is suppressed as it is not a reliable indicator for a bad fit. #245

    # with warns(
    #     UserWarning,
    #     match="lmfit could not find a good fit."
    #     " Fitted parameters may not be accurate.",
    # ):
    analysis = RabiAnalysis(
        dataset=dataset, settings_overwrite={"mpl_fig_formats": []}
    ).run()

    return analysis


def test_figures_generated_bad_fit(rabi_analysis_obj_bad_fit):
    """Test that the right figures get created despite failed fit."""
    assert set(rabi_analysis_obj_bad_fit.figs_mpl.keys()) == {
        "Rabi_oscillation",
    }


def test_quantities_of_interest_bad_fit(rabi_analysis_obj_bad_fit):
    """Test that the quantities of interest exist for a bad fit."""
    assert set(rabi_analysis_obj_bad_fit.quantities_of_interest.keys()) == {
        "Pi-pulse amplitude",
        "fit_msg",
        "fit_result",
        "fit_success",
    }
