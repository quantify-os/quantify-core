# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
# pylint: disable=invalid-name  # to allow fr, Ql and Qe names
import warnings
from pathlib import Path

import pytest
from pytest import approx
from uncertainties.core import Variable

import quantify_core.data.handling as dh
from quantify_core.analysis import spectroscopy_analysis as sa


@pytest.fixture(
    scope="session",
    autouse=True,
    params=[
        (
            "20210305-152943-497-ad8670",
            4482667411.329845,
            4949.6320391632225,
            6571.519330056671,
        ),
        (
            "20210305-154735-413-142768",
            4482704759.647413,
            4148.4196790792685,
            7288.027397061653,
        ),
        (
            "20210305-160157-184-3c17e9",
            4540265849.626052,
            4917.415560754831,
            5221.513279590303,
        ),
        (
            "20210305-161550-671-982c6d",
            4576407330.09009,
            6475.137375175341,
            7809.138979502401,
        ),
    ],
)
def analysis_and_ref(tmp_test_data_dir, request):
    dh.set_datadir(tmp_test_data_dir)

    tuid, fr, Ql, Qe = request.param
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        analysis = sa.ResonatorSpectroscopyAnalysis(tuid=tuid).run()

    return analysis, (fr, Ql, Qe)


def test_load_fit_results(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref
    for fit_name, fit_result in analysis.fit_results.items():
        loaded_fit_result = sa.ResonatorSpectroscopyAnalysis.load_fit_result(
            tuid=analysis.tuid, fit_name=fit_name
        )
        assert loaded_fit_result.params == fit_result.params


def test_raw_data_not_in_processed_dataset(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref

    container = Path(dh.locate_experiment_container(analysis.tuid))
    file_path = (
        container / "analysis_ResonatorSpectroscopyAnalysis" / "dataset_processed.hdf5"
    )
    dataset_processed = dh.load_dataset_from_path(file_path)

    assert "x0" in dataset_processed.dims.keys()
    assert "x0" in dataset_processed.coords.keys()
    assert "y0" not in dataset_processed.data_vars.keys()
    assert "y1" not in dataset_processed.data_vars.keys()
    assert "S21" in dataset_processed.data_vars.keys()


def test_figures_generated(analysis_and_ref):
    """Test that the right figures get created."""
    analysis, _ = analysis_and_ref
    assert set(analysis.figs_mpl.keys()) == {
        "S21-RealImag",
        "S21-MagnPhase",
        "S21-complex",
    }


# pylint: disable=invalid-name
def test_quantities_of_interest(analysis_and_ref):
    analysis, (fr, Ql, Qe) = analysis_and_ref
    assert set(analysis.quantities_of_interest.keys()) == {
        "Qi",
        "Qe",
        "Ql",
        "Qc",
        "fr",
        "fit_result",
        "fit_msg",
        "fit_success",
    }
    fitted_freq = analysis.quantities_of_interest["fr"]
    assert isinstance(fitted_freq, Variable)
    # Tests that the fitted values are correct (to within 5 standard deviations)
    qoi = analysis.quantities_of_interest
    assert fr == approx(qoi["fr"].nominal_value, abs=5 * qoi["fr"].std_dev)
    assert Ql == approx(qoi["Ql"].nominal_value, abs=5 * qoi["Ql"].std_dev)
    assert Qe == approx(qoi["Qe"].nominal_value, abs=5 * qoi["Qe"].std_dev)
