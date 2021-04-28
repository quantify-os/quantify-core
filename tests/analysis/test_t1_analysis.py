# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
from pathlib import Path
import pytest
from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import t1_analysis as ta

tuids = ["20210322-205253-758-6689"]
t1s = [1.07e-5]


@pytest.fixture(scope="session", autouse=True)
def analysis_objs(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    dh.set_datadir(tmp_test_data_dir)
    a_objs = [ta.T1Analysis(tuid=tuid).run() for tuid in tuids]

    return a_objs


def test_raw_data_not_in_processed_dataset(analysis_objs):
    """
    Test that all the relevant quantities are in the processed dataset
    """
    for a_obj in analysis_objs:
        container = Path(dh.locate_experiment_container(a_obj.tuid))
        file_path = container / "analysis_T1Analysis" / "processed_dataset.hdf5"
        dataset = dh.load_dataset_from_path(file_path)

        assert "x0" in dataset.dims.keys()
        assert "x0" in dataset.coords.keys()
        assert "y0" not in dataset.data_vars.keys()
        assert "y1" not in dataset.data_vars.keys()
        assert "Magnitude" in dataset.data_vars.keys()


def test_figures_generated(analysis_objs):
    """
    Test that the right figures get created.
    """
    for a_obj in analysis_objs:
        assert set(a_obj.figs_mpl.keys()) == {
            "T1_decay",
        }


def test_quantities_of_interest(analysis_objs):
    """
    Test that the fit returns the correct values
    """
    for a_obj, t_1 in zip(analysis_objs, t1s):
        assert set(a_obj.quantities_of_interest.keys()) == {
            "T1",
            "fit_msg",
            "fit_res",
        }

        assert isinstance(a_obj.quantities_of_interest["T1"], Variable)
        # Tests that the fitted values are correct (to within 5 standard deviations)
        assert a_obj.quantities_of_interest["T1"].nominal_value == approx(
            t_1, abs=5 * a_obj.quantities_of_interest["T1"].std_dev
        )
