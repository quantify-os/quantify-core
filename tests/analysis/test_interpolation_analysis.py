# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
import quantify.data.handling as dh
from quantify.analysis import interpolation_analysis as ia

tuids = ["20210419-170747-902-9c5a05"]
offsets = [[0.0008868002631485698, 0.006586920009126688]]


@pytest.fixture(scope="session", autouse=True)
def analysis_objs_interpolation(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    dh.set_datadir(tmp_test_data_dir)
    a_objs = [ia.InterpolationAnalysis2D(tuid=tuid).run() for tuid in tuids]

    return a_objs


def test_figures_generated_interpolation(analysis_objs_interpolation):
    """
    Test that the right figures get created.
    """
    for a_obj in analysis_objs_interpolation:
        assert set(a_obj.figs_mpl.keys()) == {
            "SignalHound_fixed_frequency interpolating",
        }
