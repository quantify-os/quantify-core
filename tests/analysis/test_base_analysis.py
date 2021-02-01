import pytest
import quantify.data.handling as dh
from quantify.analysis import base_analysis as ba
from quantify.utilities._tests_helpers import get_test_data_dir


def test_load_dataset():
    dh.set_datadir(get_test_data_dir())

    tuid = "20200430-170837-001-315f36"
    a = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a.figs_mpl.keys()) == {"Line plot x0-y0"}

    tuid = "20210118-202044-211-58ddb0"
    a = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a.figs_mpl.keys()) == {"Line plot x0-y0", "Line plot x0-y1"}


def test_Basic2DAnalysis():
    dh.set_datadir(get_test_data_dir())

    tuid = "20210126-162726-170-de4f78"
    a = ba.Basic2DAnalysis(tuid=tuid)
    assert set(a.figs_mpl.keys()) == {"Heatmap x0x1-y0", "Heatmap x0x1-y1"}


class DummyAnalysisSubclassRaises(ba.BaseAnalysis):
    def run_fitting(self):
        raise ValueError("Dummy exception!")


def test_flow_exception_in_step():
    dh.set_datadir(get_test_data_dir())

    with pytest.raises(RuntimeError) as excinfo:
        DummyAnalysisSubclassRaises(tuid="20200430-170837-001-315f36")

    assert "run_fitting" in str(excinfo)


def test_flow_manual():
    dh.set_datadir(get_test_data_dir())
    DummyAnalysisSubclassRaises(
        tuid="20200430-170837-001-315f36",
        flow=(
            "extract_data",
            "process_data",
            "prepare_fitting",
            # "run_fitting",  # skip undesired step
            "save_fit_results",
            "analyze_fit_results",
            "save_quantities_of_interest",
            "create_figures",
            "adjust_figures",
            "save_figures",
        ),
    )


def test_flow_skip_step():
    dh.set_datadir(get_test_data_dir())
    DummyAnalysisSubclassRaises(
        tuid="20200430-170837-001-315f36",
        flow_skip_steps=("run_fitting",),  # skip undesired step
    )


def test_flow_xlim():
    pass


def test_flow_ylim():
    pass


def test_flow_clim():
    pass
