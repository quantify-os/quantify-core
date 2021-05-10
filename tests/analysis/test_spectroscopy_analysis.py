# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
from pathlib import Path
import pytest
from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import spectroscopy_analysis as sa

frs = [
    4482627786.933104,
    4482670162.566818,
    4540287828.70407,
    4576421867.293702,
]
Qls = [
    4983.385483402395,
    4192.005581230714,
    4910.617635185228,
    6437.377871269456,
]
Qes = [
    6606.202849302761,
    7317.398211359418,
    5216.566199947343,
    7759.742961176549,
]


@pytest.fixture(scope="session", autouse=True)
def analysis_objs(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuids = dh.get_tuids_containing(
        "Resonator_id", t_start="20210305", t_stop="20210306"
    )

    a_objs = [sa.ResonatorSpectroscopyAnalysis(tuid=tuid).run() for tuid in tuids]

    return a_objs


def test_raw_data_not_in_processed_dataset(analysis_objs):
    for a_obj in analysis_objs:
        container = Path(dh.locate_experiment_container(a_obj.tuid))
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


def test_figures_generated(analysis_objs):
    # test that the right figures get created.
    for a_obj in analysis_objs:
        assert set(a_obj.figs_mpl.keys()) == {
            "S21-RealImag",
            "S21-MagnPhase",
            "S21-complex",
        }


# pylint: disable=invalid-name
def test_quantities_of_interest(analysis_objs):
    for a_obj, fr, Ql, Qe in zip(analysis_objs, frs, Qls, Qes):
        assert set(a_obj.quantities_of_interest.keys()) == {
            "Qi",
            "Qe",
            "Ql",
            "Qc",
            "fr",
            "fit_res",
            "fit_msg",
            "fit_success",
        }

        fitted_freq = a_obj.quantities_of_interest["fr"]
        assert isinstance(fitted_freq, Variable)
        # Tests that the fitted values are correct (to within 5 standard deviations)
        assert a_obj.quantities_of_interest["fr"].nominal_value == approx(
            fr, abs=5 * a_obj.quantities_of_interest["fr"].std_dev
        )
        assert a_obj.quantities_of_interest["Ql"].nominal_value == approx(
            Ql, abs=5 * a_obj.quantities_of_interest["Ql"].std_dev
        )
        assert a_obj.quantities_of_interest["Qe"].nominal_value == approx(
            Qe, abs=5 * a_obj.quantities_of_interest["Qe"].std_dev
        )
