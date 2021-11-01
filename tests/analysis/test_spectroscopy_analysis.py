# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
from pathlib import Path

import pytest
from pytest import approx
from uncertainties.core import Variable

import quantify_core.data.handling as dh
from quantify_core.analysis import spectroscopy_analysis as sa

fr_list = [
    4482627786.933104,
    4482670162.566818,
    4540287828.70407,
    4576421867.293702,
]
Ql_list = [
    4983.385483402395,
    4192.005581230714,
    4910.617635185228,
    6437.377871269456,
]
Qe_list = [
    6606.202849302761,
    7317.398211359418,
    5216.566199947343,
    7759.742961176549,
]


@pytest.fixture(scope="session", autouse=True)
def analyses(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid_list = dh.get_tuids_containing(
        "Resonator_id", t_start="20210305", t_stop="20210306"
    )

    with pytest.warns(None) as warning_list:
        analysis = [
            sa.ResonatorSpectroscopyAnalysis(tuid=tuid).run() for tuid in tuid_list
        ]

    # Check that there are no warnings raised
    assert len(warning_list) == 0

    return analysis


def test_raw_data_not_in_processed_dataset(analyses):
    for analysis in analyses:
        container = Path(dh.locate_experiment_container(analysis.tuid))
        file_path = (
            container
            / "analysis_ResonatorSpectroscopyAnalysis"
            / "dataset_processed.hdf5"
        )
        dataset_processed = dh.load_dataset_from_path(file_path)

        assert "x0" in dataset_processed.dims.keys()
        assert "x0" in dataset_processed.coords.keys()
        assert "y0" not in dataset_processed.data_vars.keys()
        assert "y1" not in dataset_processed.data_vars.keys()
        assert "S21" in dataset_processed.data_vars.keys()


def test_figures_generated(analyses):
    # test that the right figures get created.
    for analysis in analyses:
        assert set(analysis.figs_mpl.keys()) == {
            "S21-RealImag",
            "S21-MagnPhase",
            "S21-complex",
        }


# pylint: disable=invalid-name
def test_quantities_of_interest(analyses):
    for analysis, fr, Ql, Qe in zip(analyses, fr_list, Ql_list, Qe_list):
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
        assert analysis.quantities_of_interest["fr"].nominal_value == approx(
            fr, abs=5 * analysis.quantities_of_interest["fr"].std_dev
        )
        assert analysis.quantities_of_interest["Ql"].nominal_value == approx(
            Ql, abs=5 * analysis.quantities_of_interest["Ql"].std_dev
        )
        assert analysis.quantities_of_interest["Qe"].nominal_value == approx(
            Qe, abs=5 * analysis.quantities_of_interest["Qe"].std_dev
        )
