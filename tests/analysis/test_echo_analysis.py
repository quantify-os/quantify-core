# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import echo_analysis as ea

tuids = ["20210420-001339-580-97bdef"]
t2echos = [10.00e-6]


@pytest.fixture(scope="session", autouse=True)
def analysis_objs(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    dh.set_datadir(tmp_test_data_dir)
    a_objs = [ea.EchoAnalysis(tuid=tuid).run() for tuid in tuids]

    return a_objs


def test_figures_generated(analysis_objs):
    """
    Test that the right figures get created.
    """
    for a_obj in analysis_objs:
        assert set(a_obj.figs_mpl.keys()) == {
            "Echo_decay",
        }


def test_quantities_of_interest(analysis_objs):
    """
    Test that the fit returns the correct values
    """
    for a_obj, t2_echo in zip(analysis_objs, t2echos):
        assert set(a_obj.quantities_of_interest.keys()) == {
            "t2_echo",
            "fit_msg",
            "fit_result",
            "fit_success",
        }

        assert isinstance(a_obj.quantities_of_interest["t2_echo"], Variable)
        # Tests that the fitted values are correct (to within 5 standard deviations)
        assert a_obj.quantities_of_interest["t2_echo"].nominal_value == pytest.approx(
            t2_echo, abs=5 * a_obj.quantities_of_interest["t2_echo"].std_dev
        )
