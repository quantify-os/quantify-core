# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file

from typing import Iterable, List

import pytest

import quantify_core.data.handling as dh
from quantify_core.analysis import interpolation_analysis as ia
from quantify_core.data.types import TUID

tuid_list = [TUID("20210419-170747-902-9c5a05")]
offset_list = [[0.0008868002631485698, 0.006586920009126688]]


@pytest.fixture(scope="session", autouse=True)
def analysis(tmp_test_data_dir: str) -> List:
    """
    Used to run the analysis a single time and run unit tests against the created
    analysis object.
    """
    dh.set_datadir(tmp_test_data_dir)
    analysis = [ia.InterpolationAnalysis2D(tuid=tuid).run() for tuid in tuid_list]

    return analysis


def test_figures_generated(analysis: Iterable) -> None:
    """
    Test that the right figures get created.
    """
    for a_obj in analysis:
        assert set(a_obj.figs_mpl.keys()) == {
            "SignalHound_fixed_frequency interpolating",
        }
