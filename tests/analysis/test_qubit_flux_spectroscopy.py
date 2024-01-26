# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
# pylint: disable=invalid-name
import warnings
from pathlib import Path

import pytest

import quantify_core.data.handling as dh
from quantify_core.analysis.spectroscopy_analysis import QubitFluxSpectroscopyAnalysis


@pytest.fixture(
    scope="session",
    autouse=True,
    params=[
        (
            "20230309-235354-353-9c94c5",
            -0.049842003104,  # 'sweetspot'
            6355322856.14,  # 'sweetspot_freq'
        ),
        (
            "20230523-110858-666-7a7dbb",
            -0.033165989078738814,
            5646531633.665411,
        ),
    ],
)
def analysis_and_ref(tmp_test_data_dir, request):
    dh.set_datadir(tmp_test_data_dir)

    tuid, offset_0, frequency_0 = request.param
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        analysis = QubitFluxSpectroscopyAnalysis(
            tuid=tuid, dataset=dh.load_dataset(tuid)
        ).run()

    return analysis, (offset_0, frequency_0)


def test_load_fit_results(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref
    for fit_name, fit_result in analysis.fit_results.items():
        loaded_fit_result = QubitFluxSpectroscopyAnalysis.load_fit_result(
            tuid=analysis.tuid, fit_name=fit_name
        )
        assert loaded_fit_result.params == fit_result.params


def test_processed_dataset(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref

    container = Path(dh.locate_experiment_container(analysis.tuid))
    file_path = (
        container / "analysis_QubitFluxSpectroscopyAnalysis" / "dataset_processed.hdf5"
    )
    _ = dh.load_dataset_from_path(file_path)


# pylint: disable=invalid-name
def test_quantities_of_interest(analysis_and_ref):
    analysis, (offset_0, frequency_0) = analysis_and_ref
    assert set(analysis.quantities_of_interest.keys()) == {
        "fit_msg",
        "fit_result",
        "fit_success",
        "sweetspot_freq",
        "sweetspot",
        "a",
        "b",
        "c",
    }
    fitted_freq = analysis.quantities_of_interest["sweetspot_freq"]
    fitted_offs = analysis.quantities_of_interest["sweetspot"]

    # Tests that the fitted values are approximately correct.
    assert abs(frequency_0 - fitted_freq) < 5 * fitted_freq.std_dev
    assert abs(offset_0 - fitted_offs.nominal_value) < 5 * fitted_offs.std_dev
