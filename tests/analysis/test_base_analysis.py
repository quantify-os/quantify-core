import os
import logging
from pathlib import Path
from jsonschema import ValidationError
import pytest
import quantify.data.handling as dh
from quantify.analysis import base_analysis as ba
from quantify.utilities._tests_helpers import get_test_data_dir

TUID_1D_1PLOT = "20200430-170837-001-315f36"
TUID_1D_2PLOTS = "20210118-202044-211-58ddb0"
# TUID_2D_2PLOTS = "20210126-162726-170-de4f78"
TUID_2D_2PLOTS = "20210227-172939-723-53d82c"
TUID_2D_CYCLIC = "20210227-172939-723-53d82c"

# disable figure saving for all analyses for performance
ba.settings["mpl_fig_formats"] = []


class DummyAnalysisSubclassRaisesA(ba.Basic1DAnalysis):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.var = False

    def run_fitting(self):
        raise ValueError("Dummy exception!")

    def save_quantities_of_interest(self):
        super().save_quantities_of_interest()
        # Flag method was executed
        self.var = True


class DummyAnalysisSubclassArgs(ba.Basic1DAnalysis):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.var = 5

    def run_fitting(self, var=None):
        # pylint: disable=arguments-differ
        if var:
            self.var = var


def test_flow_exception_in_step():
    dh.set_datadir(get_test_data_dir())

    # try-except required to preserve __context__
    try:
        DummyAnalysisSubclassRaisesA(tuid=TUID_1D_1PLOT)
    except RuntimeError as e:
        assert isinstance(e.__context__, ValueError)


def test_flow_interrupt(caplog):
    dh.set_datadir(get_test_data_dir())
    _ = DummyAnalysisSubclassRaisesA(
        tuid=TUID_1D_1PLOT, interrupt_before=ba.AnalysisSteps.S03_RUN_FITTING
    )

    log_msgs = [
        "Executing run_analysis of",
        "execution step 0: <bound method BaseAnalysis.extract_data of",
        "execution step 1: <bound method BaseAnalysis.process_data of",
        "execution step 2: <bound method BaseAnalysis.prepare_fitting of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_flow_skip_step_continue_manually(caplog):

    dh.set_datadir(get_test_data_dir())
    with caplog.at_level(logging.INFO):
        a_obj = DummyAnalysisSubclassRaisesA(
            tuid=TUID_1D_1PLOT, interrupt_before=ba.AnalysisSteps.S03_RUN_FITTING
        )

    log_msgs = [
        "Executing run_analysis of",
        "execution step 0: <bound method BaseAnalysis.extract_data of",
        "execution step 1: <bound method BaseAnalysis.process_data of",
        "execution step 2: <bound method BaseAnalysis.prepare_fitting of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)

    a_obj.continue_analysis_from(step=ba.AnalysisSteps.S05_CREATE_FIGURES)

    log_msgs = [
        "Executing run_analysis of",
        "execution step 0: <bound method BaseAnalysis.extract_data of",
        "execution step 1: <bound method BaseAnalysis.process_data of",
        "execution step 2: <bound method BaseAnalysis.prepare_fitting of",
        # New steps:
        "execution step 5: <bound method BaseAnalysis.create_figures of",
        "execution step 6: <bound method BaseAnalysis.adjust_figures of",
        "execution step 7: <bound method BaseAnalysis.save_figures_mpl of",
        "execution step 8: <bound method BaseAnalysis.save_quantities_of_interest of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_pass_options():
    """How to change default arguments of the methods in the analysis flow."""
    dh.set_datadir(get_test_data_dir())

    step = ba.AnalysisSteps.S03_RUN_FITTING
    a_obj = DummyAnalysisSubclassArgs(tuid=TUID_1D_1PLOT, interrupt_before=step)
    a_obj.run_fitting(var=7)
    a_obj.continue_analysis_after(step=step)

    assert a_obj.var == 7


def test_flow_xlim_all():
    dh.set_datadir(get_test_data_dir())
    xlim = (0.0, 4.0)
    step = ba.AnalysisSteps.S07_SAVE_FIGURES
    a_obj = ba.Basic1DAnalysis(
        tuid=TUID_1D_2PLOTS,
        interrupt_before=step,
    )
    a_obj.adjust_xlim(*xlim)
    a_obj.continue_analysis_after(step)

    for ax in a_obj.axs_mpl.values():
        assert ax.get_xlim() == xlim


def test_flow_ylim_all(caplog):
    dh.set_datadir(get_test_data_dir())
    ylim = (0.0, 0.8)
    step = ba.AnalysisSteps.S07_SAVE_FIGURES
    a_obj = ba.Basic1DAnalysis(
        tuid=TUID_1D_2PLOTS,
        interrupt_before=step,
    )
    a_obj.adjust_ylim(*ylim)
    a_obj.continue_analysis_after(step)

    for ax in a_obj.axs_mpl.values():
        assert ax.get_ylim() == ylim

    log_msgs = [
        "Executing run_analysis of",
        "execution step 0: <bound method BaseAnalysis.extract_data of",
        "execution step 1: <bound method BaseAnalysis.process_data of",
        "execution step 2: <bound method BaseAnalysis.prepare_fitting of",
        "execution step 3: <bound method BaseAnalysis.run_fitting of",
        "execution step 4: <bound method BaseAnalysis.analyze_fit_results of",
        "execution step 5: <bound method BaseAnalysis.create_figures of",
        "execution step 6: <bound method BaseAnalysis.adjust_figures of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_flow_clim_all():
    dh.set_datadir(get_test_data_dir())
    clim = (1.0, 2.0)
    step = ba.AnalysisSteps.S07_SAVE_FIGURES
    a_obj = ba.Basic2DAnalysis(tuid=TUID_2D_2PLOTS, interrupt_before=step)
    a_obj.adjust_clim(*clim)
    a_obj.continue_analysis_after(step)

    ax = a_obj.axs_mpl["Heatmap x0x1-y1"]
    assert ax.collections[0].get_clim() == clim
    ax = a_obj.axs_mpl["Heatmap x0x1-y0"]
    assert ax.collections[0].get_clim() == clim


def test_flow_clim_specific():
    dh.set_datadir(get_test_data_dir())
    clim = (0.0, 180.0)
    step = ba.AnalysisSteps.S07_SAVE_FIGURES
    a_obj = ba.Basic2DAnalysis(tuid=TUID_2D_2PLOTS, interrupt_before=step)
    a_obj.adjust_clim(*clim, ax_ids=["Heatmap x0x1-y1"])
    a_obj.continue_analysis_after(step)

    ax = a_obj.axs_mpl["Heatmap x0x1-y1"]
    assert ax.collections[0].get_clim() == clim


def test_basic1danalysis_settings_validation():
    dh.set_datadir(get_test_data_dir())
    tuid = TUID_1D_1PLOT

    with pytest.raises(ValidationError) as excinfo:
        _ = ba.Basic1DAnalysis(tuid=tuid, settings_overwrite={"mpl_fig_formats": "png"})

    assert "'png' is not of type 'array'" in str(excinfo.value)


# Run defaults at the end to overwrite the previous figures


def test_basic1d_analysis(caplog):
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_1D_1PLOT
    a_obj = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a_obj.figs_mpl.keys()) == {"Line plot x0-y0"}

    tuid = TUID_1D_2PLOTS
    # here we see if figures are created
    ba.settings["mpl_fig_formats"] = [
        "png",
        "svg",
    ]
    a_obj = ba.Basic1DAnalysis(tuid=tuid)
    ba.settings["mpl_fig_formats"] = []  # disabled again after running analysis

    # test that the right figures get created.
    assert set(a_obj.figs_mpl.keys()) == {"Line plot x0-y0", "Line plot x0-y1"}

    exp_dir = dh.locate_experiment_container(a_obj.tuid, dh.get_datadir())
    assert "analysis_Basic1DAnalysis" in os.listdir(exp_dir)
    analysis_dir = os.listdir(Path(exp_dir) / "analysis_Basic1DAnalysis")
    assert "figs_mpl" in analysis_dir

    log_msgs = [
        "Executing run_analysis of",
        "execution step 0: <bound method BaseAnalysis.extract_data of",
        "execution step 1: <bound method BaseAnalysis.process_data of",
        "execution step 2: <bound method BaseAnalysis.prepare_fitting of",
        "execution step 3: <bound method BaseAnalysis.run_fitting of",
        "execution step 4: <bound method BaseAnalysis.analyze_fit_results of",
        "execution step 5: <bound method BaseAnalysis.create_figures of",
        "execution step 6: <bound method BaseAnalysis.adjust_figures of",
        "execution step 7: <bound method BaseAnalysis.save_figures_mpl of",
        "execution step 8: <bound method BaseAnalysis.save_quantities_of_interest of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_basic2d_analysis():
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_2D_2PLOTS
    # here we see if figures are created
    ba.settings["mpl_fig_formats"] = [
        "svg",
    ]  # no png as this is very slow
    a_obj = ba.Basic2DAnalysis(tuid=tuid)
    ba.settings["mpl_fig_formats"] = []  # disabled again after running analysis

    assert set(a_obj.figs_mpl.keys()) == {
        "Heatmap x0x1-y0",
        "Heatmap x0x1-y1",
        "Linecuts x0x1-y0",
        "Linecuts x0x1-y1",
    }

    exp_dir = dh.locate_experiment_container(a_obj.tuid, dh.get_datadir())
    assert "analysis_Basic2DAnalysis" in os.listdir(exp_dir)
    analysis_dir = os.listdir(Path(exp_dir) / "analysis_Basic2DAnalysis")
    assert "figs_mpl" in analysis_dir


def test_Basic2DAnalysis_cyclic_cmap_detection():
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_2D_CYCLIC
    # here we see if figures are created
    ba.settings["mpl_fig_formats"] = [
        "svg",
    ]  # no png as this is very slow
    a_obj = ba.Basic2DAnalysis(tuid=tuid)
    ba.settings["mpl_fig_formats"] = []  # disabled again after running analysis

    # no changes are made
    fig = a_obj.figs_mpl["Heatmap x0x1-y0"]
    qm = fig.axes[0].collections[0]
    # assert colormap is default
    assert qm.get_cmap().name == "viridis"

    # range is auto set appropriately
    fig = a_obj.figs_mpl["Heatmap x0x1-y1"]
    qm = fig.axes[0].collections[0]
    assert qm.get_clim() == (-180, 180)
    assert qm.get_cmap().name == "twilight_shifted"


def test_display_figs():
    dh.set_datadir(get_test_data_dir())
    a_obj = ba.Basic1DAnalysis(tuid=TUID_1D_2PLOTS)
    a_obj.display_figs_mpl()  # should display figures in the output
