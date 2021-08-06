# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from uncertainties.core import Variable
import quantify_core.data.handling as dh
from quantify_core.analysis import echo_analysis as ea

tuid_list = ["20210420-001339-580-97bdef"]
t2_echo_list = [10.00e-6]


@pytest.fixture(scope="session", autouse=True)
def analyses(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    dh.set_datadir(tmp_test_data_dir)
    analyses = [ea.EchoAnalysis(tuid=tuid).run() for tuid in tuid_list]

    return analyses


def test_figures_generated(analyses):
    """
    Test that the right figures get created.
    """
    for analysis in analyses:
        assert set(analysis.figs_mpl.keys()) == {
            "Echo_decay",
        }


def test_quantities_of_interest(analyses):
    """
    Test that the fit returns the correct values
    """
    for analysis, t2_echo in zip(analyses, t2_echo_list):
        assert set(analysis.quantities_of_interest.keys()) == {
            "t2_echo",
            "fit_msg",
            "fit_result",
            "fit_success",
        }

        assert isinstance(analysis.quantities_of_interest["t2_echo"], Variable)
        # Tests that the fitted values are correct (to within 5 standard deviations)
        assert analysis.quantities_of_interest[
            "t2_echo"
        ].nominal_value == pytest.approx(
            t2_echo, abs=5 * analysis.quantities_of_interest["t2_echo"].std_dev
        )
