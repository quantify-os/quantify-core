# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

import numpy as np
import pytest

import quantify_core.data.handling as dh
from quantify_core.analysis.spectroscopy_analysis import (
    ResonatorFluxSpectroscopyAnalysis,
)


@pytest.fixture(
    scope="session",
    autouse=True,
    params=[
        (
            "20230308-235659-059-cf471e",
            9.463e5,  # amplitude (unitless)
            971.9e-3,  # frequency of flux sinusoid (V)
            1.8854,  # shift of flux sinusoid (V)
            7.831203e9,  # center of flux sinusoid (Hz)
            -0.0515,  # sweetspot_0 (V)
        ),
        (
            "20230309-235228-129-1a58f5",
            9.455e5,  # amplitude (unitless)
            972.1e-3,  # frequency of flux sinusoid (V)
            1.8844,  # shift of flux sinusoid (V)
            7.8312e9,  # center of flux sinusoid (Hz)
            -0.0513,  # sweetspot_0 (V)
        ),
    ],
)
def analysis_and_ref(tmp_test_data_dir, request):
    dh.set_datadir(tmp_test_data_dir)

    tuid, amplitude, frequency, shift, center, sweetspot_0 = request.param

    analysis = ResonatorFluxSpectroscopyAnalysis(
        tuid=tuid, dataset=dh.load_dataset(tuid)
    ).run()

    return analysis, (amplitude, frequency, shift, center, sweetspot_0)


def test_load_fit_results(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref
    for fit_name, fit_result in analysis.fit_results.items():
        loaded_fit_result = ResonatorFluxSpectroscopyAnalysis.load_fit_result(
            tuid=analysis.tuid, fit_name=fit_name
        )
        assert loaded_fit_result.params == fit_result.params


# pylint: disable=invalid-name
def test_quantities_of_interest(analysis_and_ref):
    analysis, fixed_qois = analysis_and_ref
    assert {
        "amplitude",
        "fit_msg",
        "fit_result",
        "fit_success",
        "frequency",
        "center",
        "shift",
    } <= set(analysis.quantities_of_interest.keys())

    for fixed_qoi, name in zip(
        fixed_qois, ("amplitude", "frequency", "shift", "center", "sweetspot_0")
    ):
        fitted_qoi = analysis.quantities_of_interest[name]
        assert abs(fixed_qoi - fitted_qoi.nominal_value) < 5 * fitted_qoi.std_dev


# pylint: disable=invalid-name
def test_print_error_without_crash(analysis_and_ref, capsys):
    analysis, _ = analysis_and_ref

    # re-run analysis with nan values
    bad_ds = analysis.dataset
    bad_ds.y0.data[100:] = np.asarray([float("nan")] * (bad_ds.y0.data.size - 100))

    _ = ResonatorFluxSpectroscopyAnalysis(dataset=bad_ds).run()

    # Capture the printed output
    captured = capsys.readouterr()

    assert "Error during fit:" in captured.out
