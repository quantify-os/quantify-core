import os
import pytest
from pathlib import Path
import quantify.data.handling as dh
from quantify.analysis import base_analysis as ba
from quantify.utilities._tests_helpers import get_test_data_dir

TUID_1D_1PLOT = "20200430-170837-001-315f36"
TUID_1D_2PLOTS = "20210118-202044-211-58ddb0"
TUID_2D_2PLOTS = "20210126-162726-170-de4f78"


class DummyAnalysisSubclassRaisesA(ba.Basic1DAnalysis):
    def create_figures(self):
        raise ValueError("Dummy exception!")


class DummyAnalysisSubclassRaisesB(ba.Basic1DAnalysis):
    def run_fitting(self):
        raise ValueError("Dummy exception!")


def test_flow_exception_in_step():
    dh.set_datadir(get_test_data_dir())

    with pytest.raises(ValueError):
        DummyAnalysisSubclassRaisesA(tuid=TUID_1D_1PLOT)


def test_flow_interrupt():
    dh.set_datadir(get_test_data_dir())
    a_obj = DummyAnalysisSubclassRaisesA(
        tuid=TUID_1D_1PLOT, interrupt_after="save_quantities_of_interest"
    )
    assert len(a_obj.figs_mpl) == 0


def test_flow_skip_step_continue_manually():
    dh.set_datadir(get_test_data_dir())
    a_obj = DummyAnalysisSubclassRaisesB(
        tuid=TUID_1D_1PLOT, interrupt_after="prepare_fitting"
    )

    a_obj.continue_analysis_from(method_name="save_quantities_of_interest")

    assert len(a_obj.figs_mpl)


def test_flow_xlim_all():
    dh.set_datadir(get_test_data_dir())
    xlim = (0.0, 4.0)
    method_name = "adjust_figures"
    a_obj = ba.Basic1DAnalysis(
        tuid=TUID_1D_2PLOTS,
        interrupt_after=method_name,
    )
    a_obj.adjust_xlim(*xlim)
    a_obj.continue_analysis_after(method_name)

    for ax in a_obj.axs_mpl.values():
        assert ax.get_xlim() == xlim


def test_flow_ylim_all():
    dh.set_datadir(get_test_data_dir())
    ylim = (0.0, 0.8)
    method_name = "adjust_figures"
    a_obj = ba.Basic1DAnalysis(
        tuid=TUID_1D_2PLOTS,
        interrupt_after=method_name,
    )
    a_obj.adjust_ylim(*ylim)
    a_obj.continue_analysis_after(method_name)

    for ax in a_obj.axs_mpl.values():
        assert ax.get_ylim() == ylim


def test_flow_clim_all():
    dh.set_datadir(get_test_data_dir())
    clim = (1.0, 2.0)
    method_name = "adjust_figures"
    a_obj = ba.Basic2DAnalysis(tuid=TUID_2D_2PLOTS, interrupt_after=method_name)
    a_obj.adjust_clim(*clim)
    a_obj.continue_analysis_after(method_name)

    for ax in a_obj.axs_mpl.values():
        assert ax.collections[0].get_clim() == clim


def test_flow_clim_specific():
    dh.set_datadir(get_test_data_dir())
    clim = (0.0, 180.0)
    method_name = "adjust_figures"
    a_obj = ba.Basic2DAnalysis(tuid=TUID_2D_2PLOTS, interrupt_after=method_name)
    a_obj.adjust_clim(*clim, ax_ids=["Heatmap x0x1-y1"])
    a_obj.continue_analysis_after(method_name)

    ax = a_obj.axs_mpl["Heatmap x0x1-y1"]
    assert ax.collections[0].get_clim() == clim


# Run defaults at the end to overwrite the previous figures


def test_Basic1DAnalysis():
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_1D_1PLOT
    a_obj = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a_obj.figs_mpl.keys()) == {"Line plot x0-y0"}

    tuid = TUID_1D_2PLOTS
    a_obj = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a_obj.figs_mpl.keys()) == {"Line plot x0-y0", "Line plot x0-y1"}

    exp_dir = dh.locate_experiment_container(a_obj.tuid, dh.get_datadir())
    assert "analysis_Basic1DAnalysis" in os.listdir(exp_dir)
    analysis_dir = os.listdir(Path(exp_dir) / "analysis_Basic1DAnalysis")
    assert "figs_mpl" in analysis_dir


def test_Basic2DAnalysis():
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_2D_2PLOTS
    a_obj = ba.Basic2DAnalysis(tuid=tuid)
    assert set(a_obj.figs_mpl.keys()) == {"Heatmap x0x1-y0", "Heatmap x0x1-y1"}

    exp_dir = dh.locate_experiment_container(a_obj.tuid, dh.get_datadir())
    assert "analysis_Basic2DAnalysis" in os.listdir(exp_dir)
    analysis_dir = os.listdir(Path(exp_dir) / "analysis_Basic2DAnalysis")
    assert "figs_mpl" in analysis_dir


def test_display_figs():
    dh.set_datadir(get_test_data_dir())
    a_obj = ba.Basic1DAnalysis(tuid=TUID_1D_2PLOTS)
    a_obj.display_figs_mpl()  # should display figures in the output
