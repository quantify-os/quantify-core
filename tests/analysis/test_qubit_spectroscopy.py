# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
# pylint: disable=invalid-name
import warnings
from pathlib import Path

import pytest

import quantify_core.data.handling as dh
from quantify_core.analysis.spectroscopy_analysis import QubitSpectroscopyAnalysis


@pytest.fixture(
    scope="session",
    autouse=True,
    params=[
        ("20230523-175716-868-8746ad", 5635062937.11554),
        ("20230523-114322-399-dacb68", 4544928307.951747),
        ("20230508-122129-027-38881c", 6230583873.24426),
    ],
)
def analysis_and_ref(tmp_test_data_dir, request):
    dh.set_datadir(tmp_test_data_dir)

    tuid, frequency_0 = request.param
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        analysis = QubitSpectroscopyAnalysis(
            tuid=tuid, dataset=dh.load_dataset(tuid)
        ).run()

    return analysis, (frequency_0,)


def test_load_fit_results(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref
    for fit_name, fit_result in analysis.fit_results.items():
        loaded_fit_result = QubitSpectroscopyAnalysis.load_fit_result(
            tuid=analysis.tuid, fit_name=fit_name
        )
        assert loaded_fit_result.params == fit_result.params


def test_processed_dataset(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref

    container = Path(dh.locate_experiment_container(analysis.tuid))
    file_path = (
        container / "analysis_QubitSpectroscopyAnalysis" / "dataset_processed.hdf5"
    )
    dataset_processed = dh.load_dataset_from_path(file_path)

    assert hasattr(dataset_processed, "Magnitude")


# pylint: disable=invalid-name
def test_quantities_of_interest(analysis_and_ref):
    analysis, (frequency_0,) = analysis_and_ref

    fitted_freq = analysis.quantities_of_interest["frequency_01"]

    # Tests that the fitted values are approximately correct.
    assert abs(frequency_0 - fitted_freq) < 5 * fitted_freq.std_dev
