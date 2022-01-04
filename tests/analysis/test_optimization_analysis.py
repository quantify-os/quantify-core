# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
from typing import Iterable, List

import pytest
from pytest import approx

import quantify_core.data.handling as dh
from quantify_core.analysis import optimization_analysis as oa
from quantify_core.data.types import TUID

tuid_list = [TUID("20210419-170747-902-9c5a05")]
offset_list = [[0.0008868002631485698, 0.006586920009126688]]
POWER = -118.08462066650391


@pytest.fixture(scope="session", autouse=True)
def analyses(tmp_test_data_dir: str) -> List:
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    dh.set_datadir(tmp_test_data_dir)
    analyses = [oa.OptimizationAnalysis(tuid=tuid).run() for tuid in tuid_list]

    return analyses


def test_figures_generated(analyses: Iterable) -> None:
    """
    Test that the right figures get created.
    """
    for analysis in analyses:
        assert set(analysis.figs_mpl.keys()) == {
            "Line plot sequencer0_offset_awg_path0 vs iteration",
            "Line plot sequencer0_offset_awg_path1 vs iteration",
            "Line plot SignalHound_fixed_frequency vs iteration",
        }


def test_quantities_of_interest(analyses: Iterable) -> None:
    """
    Test that the optimization returns the correct values
    """
    for analysis, offset in zip(analyses, offset_list):
        assert set(analysis.quantities_of_interest.keys()) == {
            "sequencer0_offset_awg_path0",
            "sequencer0_offset_awg_path1",
            "SignalHound_fixed_frequency",
            "plot_msg",
        }

        assert isinstance(
            analysis.quantities_of_interest["sequencer0_offset_awg_path0"], float
        )
        assert isinstance(
            analysis.quantities_of_interest["sequencer0_offset_awg_path1"], float
        )
        assert isinstance(
            analysis.quantities_of_interest["SignalHound_fixed_frequency"], float
        )
        # Tests that the optimal values are correct
        assert analysis.quantities_of_interest["sequencer0_offset_awg_path0"] == approx(
            offset[0], rel=0.05
        )
        assert analysis.quantities_of_interest["sequencer0_offset_awg_path1"] == approx(
            offset[1], rel=0.05
        )
        assert analysis.quantities_of_interest["SignalHound_fixed_frequency"] == approx(
            POWER, rel=0.05
        )
