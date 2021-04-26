# pylint: disable=invalid-name # disabled because of capital SI in module name
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import os
import logging
from pathlib import Path
import xarray as xr
from jsonschema import ValidationError
import pytest
import lmfit
import numpy as np
import quantify.data.handling as dh
from quantify.analysis import base_analysis as ba
from quantify.utilities._tests_helpers import get_test_data_dir

TUID_1D_1PLOT = "20200430-170837-001-315f36"
TUID_1D_2PLOTS = "20210118-202044-211-58ddb0"
TUID_1D_ALLXY = "20210331-133734-718-aa9c83"
# TUID_2D_2PLOTS = "20210126-162726-170-de4f78"
TUID_2D_2PLOTS = "20210227-172939-723-53d82c"
TUID_2D_CYCLIC = "20210227-172939-723-53d82c"

# disable figure saving for all analyses for performance
ba.settings["mpl_fig_formats"] = []

# pylint: disable=attribute-defined-outside-init
class DummyAnalysisSubclassRaisesA(ba.Basic1DAnalysis):
    def run(self):
        self.var = False
        self.execute_analysis_steps()
        return self

    def run_fitting(self):
        raise ValueError("Dummy exception!")

    def save_quantities_of_interest(self):
        super().save_quantities_of_interest()
        # Flag method was executed
        self.var = True


# pylint: disable=attribute-defined-outside-init
class DummyAnalysisSubclassArgs(ba.Basic1DAnalysis):
    def run(self, var=None):
        # pylint: disable=arguments-differ
        if var:
            self.var = var
        self.execute_analysis_steps()
        return self


def test_run_partial(caplog):
    dh.set_datadir(get_test_data_dir())
    _ = DummyAnalysisSubclassRaisesA(tuid=TUID_1D_1PLOT).run_until(
        interrupt_before=DummyAnalysisSubclassRaisesA.analysis_steps.STEP_2_RUN_FITTING
    )

    log_msgs = [
        "Executing",
        "<bound method BaseAnalysis.extract_data of",
        "<bound method BaseAnalysis.process_data of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_flow_skip_step_continue_manually(caplog):

    dh.set_datadir(get_test_data_dir())
    # test both string or Enum member specification of the analysis steps
    with caplog.at_level(logging.INFO):
        a_obj = DummyAnalysisSubclassRaisesA(tuid=TUID_1D_1PLOT).run_until(
            interrupt_before="run_fitting"
        )

    log_msgs = [
        "Executing",
        "<bound method BaseAnalysis.extract_data of",
        "<bound method BaseAnalysis.process_data of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)

    a_obj.run_from(step="create_figures")

    log_msgs = [
        "Executing",
        "<bound method BaseAnalysis.extract_data of",
        "<bound method BaseAnalysis.process_data of",
        # New steps:
        "<bound method BaseAnalysis.create_figures of",
        "<bound method BaseAnalysis.adjust_figures of",
        "<bound method BaseAnalysis.save_figures_mpl of",
        "<bound method BaseAnalysis.save_quantities_of_interest of",
        "<bound method BaseAnalysis.save_processed_dataset of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_pass_options():
    """How to change default arguments of the methods in the analysis flow."""
    dh.set_datadir(get_test_data_dir())
    a_obj = DummyAnalysisSubclassArgs(tuid=TUID_1D_1PLOT).run(var=7)
    assert a_obj.var == 7


def test_flow_xlim_all():
    dh.set_datadir(get_test_data_dir())
    xlim = (0.0, 4.0)
    step = ba.Basic1DAnalysis.analysis_steps.STEP_6_SAVE_FIGURES
    a_obj = ba.Basic1DAnalysis(tuid=TUID_1D_2PLOTS).run_until(interrupt_before=step)
    a_obj.adjust_xlim(*xlim)
    a_obj.run_from(step)

    for ax in a_obj.axs_mpl.values():
        assert ax.get_xlim() == xlim


def test_flow_ylim_all(caplog):
    dh.set_datadir(get_test_data_dir())
    ylim = (0.0, 0.8)
    step = ba.Basic1DAnalysis.analysis_steps.STEP_6_SAVE_FIGURES
    a_obj = ba.Basic1DAnalysis(tuid=TUID_1D_2PLOTS).run_until(interrupt_before=step)
    a_obj.adjust_ylim(*ylim)
    a_obj.run_from(step)

    for ax in a_obj.axs_mpl.values():
        assert ax.get_ylim() == ylim

    log_msgs = [
        "Executing",
        "<bound method BaseAnalysis.extract_data of",
        "<bound method BaseAnalysis.process_data of",
        "<bound method BaseAnalysis.run_fitting of",
        "<bound method BaseAnalysis.analyze_fit_results of",
        "<bound method BaseAnalysis.create_figures of",
        "<bound method BaseAnalysis.adjust_figures of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_flow_clim_all():
    dh.set_datadir(get_test_data_dir())
    clim = (1.0, 2.0)
    step = ba.Basic2DAnalysis.analysis_steps.STEP_6_SAVE_FIGURES
    a_obj = ba.Basic2DAnalysis(tuid=TUID_2D_2PLOTS).run_until(interrupt_before=step)
    a_obj.adjust_clim(*clim)
    a_obj.run_from(step)

    ax = a_obj.axs_mpl["Heatmap x0x1-y1"]
    assert ax.collections[0].get_clim() == clim
    ax = a_obj.axs_mpl["Heatmap x0x1-y0"]
    assert ax.collections[0].get_clim() == clim


def test_flow_clim_specific():
    dh.set_datadir(get_test_data_dir())
    clim = (0.0, 180.0)
    step = ba.Basic2DAnalysis.analysis_steps.STEP_6_SAVE_FIGURES
    a_obj = ba.Basic2DAnalysis(tuid=TUID_2D_2PLOTS).run_until(interrupt_before=step)
    a_obj.adjust_clim(*clim, ax_ids=["Heatmap x0x1-y1"])
    a_obj.run_from(step)

    ax = a_obj.axs_mpl["Heatmap x0x1-y1"]
    assert ax.collections[0].get_clim() == clim


def test_basic1danalysis_settings_validation():
    dh.set_datadir(get_test_data_dir())
    tuid = TUID_1D_1PLOT

    with pytest.raises(ValidationError) as excinfo:
        _ = ba.Basic1DAnalysis(
            tuid=tuid, settings_overwrite={"mpl_fig_formats": "png"}
        ).run()

    assert "'png' is not of type 'array'" in str(excinfo.value)


# Run defaults at the end to overwrite the previous figures


def test_basic1d_analysis(caplog):
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_1D_1PLOT
    a_obj = ba.Basic1DAnalysis(tuid=tuid).run()

    # test that the right figures get created.
    assert set(a_obj.figs_mpl.keys()) == {"Line plot x0-y0"}

    tuid = TUID_1D_2PLOTS
    # here we see if figures are created
    ba.settings["mpl_fig_formats"] = [
        "png",
        "svg",
    ]
    a_obj = ba.Basic1DAnalysis(tuid=tuid).run()
    ba.settings["mpl_fig_formats"] = []  # disabled again after running analysis

    # test that the right figures get created.
    assert set(a_obj.figs_mpl.keys()) == {"Line plot x0-y0", "Line plot x0-y1"}

    exp_dir = dh.locate_experiment_container(a_obj.tuid, dh.get_datadir())
    assert "analysis_Basic1DAnalysis" in os.listdir(exp_dir)
    analysis_dir = os.listdir(Path(exp_dir) / "analysis_Basic1DAnalysis")
    assert "figs_mpl" in analysis_dir

    log_msgs = [
        "Executing",
        "<bound method BaseAnalysis.extract_data of",
        "<bound method BaseAnalysis.process_data of",
        "<bound method BaseAnalysis.run_fitting of",
        "<bound method BaseAnalysis.analyze_fit_results of",
        "<bound method BaseAnalysis.create_figures of",
        "<bound method BaseAnalysis.adjust_figures of",
        "<bound method BaseAnalysis.save_figures_mpl of",
        "<bound method BaseAnalysis.save_quantities_of_interest of",
    ]

    for log_msg, rec in zip(log_msgs, caplog.records):
        assert log_msg in str(rec.msg)


def test_basic1d_analysis_plot_repeated_pnts():
    dh.set_datadir(get_test_data_dir())
    a_obj = ba.Basic1DAnalysis(tuid=TUID_1D_ALLXY).run()

    # test that the duplicated setpoints measured are plotted
    assert len(a_obj.axs_mpl["Line plot x0-y0"].lines[0].get_data()[0]) == len(
        a_obj.dataset_raw.x0
    )


def test_basic2d_analysis():
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_2D_2PLOTS
    # here we see if figures are created
    ba.settings["mpl_fig_formats"] = [
        "svg",
    ]  # no png as this is very slow
    a_obj = ba.Basic2DAnalysis(tuid=tuid).run()
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


def test_basic2d_cyclic_cmap_detection():
    dh.set_datadir(get_test_data_dir())

    tuid = TUID_2D_CYCLIC
    # here we see if figures are created
    ba.settings["mpl_fig_formats"] = [
        "svg",
    ]  # no png as this is very slow
    a_obj = ba.Basic2DAnalysis(tuid=tuid).run()
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
    a_obj = ba.Basic1DAnalysis(tuid=TUID_1D_2PLOTS).run()
    a_obj.display_figs_mpl()  # should display figures in the output


def test_dataset_input_invalid():
    # a dataset with missing tuid is an invalid dataset

    # Create a custom dataset
    x0 = np.linspace(0, 2 * np.pi, 31)
    y0 = np.cos(x0)
    x0r = xr.DataArray(
        x0, name="x0", attrs={"name": "t", "long_name": "Time", "units": "s"}
    )
    y0r = xr.DataArray(
        y0, name="y0", attrs={"name": "A", "long_name": "Amplitude", "units": "V"}
    )

    dset = xr.Dataset({"x0": x0r, "y0": y0r}, attrs={"name": "custom cosine"})
    dset = dset.set_coords(["x0"])
    # no TUID attribute present

    with pytest.raises(AttributeError):
        ba.Basic1DAnalysis(dataset_raw=dset).run()


def test_dataset_input():
    # Create a custom dataset
    x0 = np.linspace(0, 2 * np.pi, 31)
    y0 = np.cos(x0)
    x0r = xr.DataArray(
        x0, name="x0", attrs={"name": "t", "long_name": "Time", "units": "s"}
    )
    y0r = xr.DataArray(
        y0, name="y0", attrs={"name": "A", "long_name": "Amplitude", "units": "V"}
    )

    dset = xr.Dataset(
        {"x0": x0r, "y0": y0r},
        attrs={"name": "custom cosine", "tuid": "20210417-191934-749-41de74"},
    )
    dset = dset.set_coords(["x0"])

    # execute analysis with dataset as input argument
    a_obj = ba.Basic1DAnalysis(
        dataset_raw=dset, settings_overwrite={"mpl_fig_formats": ["png"]}
    ).run()

    assert a_obj.dataset_raw == dset

    exp_dir = dh.locate_experiment_container(a_obj.tuid, dh.get_datadir())
    # assert a copy of the dataset was stored to disk.
    assert "dataset.hdf5" in os.listdir(exp_dir)
    # assert figures where stored to disk.
    assert "analysis_Basic1DAnalysis" in os.listdir(exp_dir)
    analysis_dir = os.listdir(Path(exp_dir) / "analysis_Basic1DAnalysis")
    assert "figs_mpl" in analysis_dir
    assert "Line plot x0-y0.png" in os.listdir(
        Path(exp_dir) / "analysis_Basic1DAnalysis" / "figs_mpl"
    )

    # test that the right figures get created.
    assert set(a_obj.figs_mpl.keys()) == {"Line plot x0-y0"}


def test_lmfit_par_to_ufloat():
    par = lmfit.Parameter("freq", value=4)
    par.stderr = 1

    ufloat_obj = ba.lmfit_par_to_ufloat(par)

    assert ufloat_obj.nominal_value == 4
    assert ufloat_obj.std_dev == 1

    # Make sure None does not raise errors
    par.stderr = None

    ufloat_obj = ba.lmfit_par_to_ufloat(par)

    assert ufloat_obj.nominal_value == 4
    assert np.isnan(ufloat_obj.std_dev)
