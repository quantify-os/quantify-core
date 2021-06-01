# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import pytest
from pytest import approx
import quantify.data.handling as dh
from quantify.analysis import optimization_analysis as oa

tuids = ["20210419-170747-902-9c5a05"]
offsets = [[0.0008868002631485698, 0.006586920009126688]]
POWER = -118.08462066650391


@pytest.fixture(scope="session", autouse=True)
def analysis_objs(tmp_test_data_dir):
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    dh.set_datadir(tmp_test_data_dir)
    a_objs = [oa.OptimizationAnalysis(tuid=tuid).run() for tuid in tuids]

    return a_objs


def test_figures_generated(analysis_objs):
    """
    Test that the right figures get created.
    """
    for a_obj in analysis_objs:
        assert set(a_obj.figs_mpl.keys()) == {
            "Line plot sequencer0_offset_awg_path0 vs SignalHound_fixed_frequency",
            "Line plot sequencer0_offset_awg_path1 vs SignalHound_fixed_frequency",
            "Line plot sequencer0_offset_awg_path0 vs iteration",
            "Line plot sequencer0_offset_awg_path1 vs iteration",
            "Line plot SignalHound_fixed_frequency vs iteration",
        }


def test_quantities_of_interest(analysis_objs):
    """
    Test that the optimization returns the correct values
    """
    for a_obj, offset in zip(analysis_objs, offsets):
        assert set(a_obj.quantities_of_interest.keys()) == {
            "sequencer0_offset_awg_path0",
            "sequencer0_offset_awg_path1",
            "SignalHound_fixed_frequency",
            "plot_msg",
        }

        assert isinstance(
            a_obj.quantities_of_interest["sequencer0_offset_awg_path0"], float
        )
        assert isinstance(
            a_obj.quantities_of_interest["sequencer0_offset_awg_path1"], float
        )
        assert isinstance(
            a_obj.quantities_of_interest["SignalHound_fixed_frequency"], float
        )
        # Tests that the optimal values are correct
        assert a_obj.quantities_of_interest["sequencer0_offset_awg_path0"] == approx(
            offset[0], rel=0.05
        )
        assert a_obj.quantities_of_interest["sequencer0_offset_awg_path1"] == approx(
            offset[1], rel=0.05
        )
        assert a_obj.quantities_of_interest["SignalHound_fixed_frequency"] == approx(
            POWER, rel=0.05
        )
