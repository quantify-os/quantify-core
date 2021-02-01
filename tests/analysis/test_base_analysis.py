import pytest
import quantify.data.handling as dh
from quantify.analysis import base_analysis as ba
from quantify.utilities._tests_helpers import get_test_data_dir

tuid_1D_1plot = "20200430-170837-001-315f36"
tuid_1D_2plots = "20210118-202044-211-58ddb0"
tuid_2D_2plots = "20210126-162726-170-de4f78"


class DummyAnalysisSubclassRaises(ba.BaseAnalysis):
    def run_fitting(self):
        raise ValueError("Dummy exception!")


def test_flow_exception_in_step():
    dh.set_datadir(get_test_data_dir())

    with pytest.raises(RuntimeError) as excinfo:
        DummyAnalysisSubclassRaises(tuid=tuid_1D_1plot)

    assert "run_fitting" in str(excinfo)


def test_flow_manual():
    dh.set_datadir(get_test_data_dir())
    DummyAnalysisSubclassRaises(
        tuid=tuid_1D_1plot,
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


def test_flow_xlim_all():
    dh.set_datadir(get_test_data_dir())
    xlim = (0.0, 4.0)
    a = ba.Basic1DAnalysis(
        tuid=tuid_1D_2plots,
        flow_override_step=(
            "adjust_figures",
            dict(user_func=ba.mk_adjust_xylim(*xlim)),
        ),
    )
    for ax in a.axs_mpl.values():
        assert ax.get_xlim() == xlim


def test_flow_xlim_all_manual():
    dh.set_datadir(get_test_data_dir())
    xlim = (0.0, 4.0)
    a = ba.Basic1DAnalysis(
        tuid=tuid_1D_2plots,
        flow=(
            "extract_data",
            "process_data",
            "prepare_fitting",
            # "run_fitting",  # skip undesired step
            "save_fit_results",
            "analyze_fit_results",
            "save_quantities_of_interest",
            "create_figures",
            ("adjust_figures", dict(user_func=ba.mk_adjust_xylim(*xlim))),
            "save_figures",
        ),
    )
    for ax in a.axs_mpl.values():
        assert ax.get_xlim() == xlim


def test_flow_ylim_all():
    dh.set_datadir(get_test_data_dir())
    ylim = (0.0, 0.8)
    a = ba.Basic1DAnalysis(
        tuid=tuid_1D_2plots,
        flow_override_step=(
            "adjust_figures",
            dict(user_func=ba.mk_adjust_xylim(None, None, *ylim)),
        ),
    )
    for ax in a.axs_mpl.values():
        assert ax.get_ylim() == ylim


def test_flow_clim_all():
    dh.set_datadir(get_test_data_dir())
    clim = (1.0, 2.0)
    a = ba.Basic2DAnalysis(
        tuid=tuid_2D_2plots,
        flow_override_step=(
            "adjust_figures",
            dict(user_func=ba.mk_adjust_clim(*clim)),
        ),
    )
    for ax in a.axs_mpl.values():
        assert ax.collections[0].get_clim() == clim


def test_flow_clim_specific():
    dh.set_datadir(get_test_data_dir())
    clim = (0.0, 180.0)
    a = ba.Basic2DAnalysis(
        tuid=tuid_2D_2plots,
        flow_override_step=(
            "adjust_figures",
            dict(user_func=ba.mk_adjust_clim(*clim, "Phase")),
        ),
    )
    ax = a.axs_mpl["Heatmap x0x1-y1"]
    assert ax.collections[0].get_clim() == clim


# Run defaults at the end to overwrite the previous figures


def test_Basic1DAnalysis():
    dh.set_datadir(get_test_data_dir())

    tuid = tuid_1D_1plot
    a = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a.figs_mpl.keys()) == {"Line plot x0-y0"}

    tuid = tuid_1D_2plots
    a = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a.figs_mpl.keys()) == {"Line plot x0-y0", "Line plot x0-y1"}


def test_Basic2DAnalysis():
    dh.set_datadir(get_test_data_dir())

    tuid = tuid_2D_2plots
    a = ba.Basic2DAnalysis(tuid=tuid)
    assert set(a.figs_mpl.keys()) == {"Heatmap x0x1-y0", "Heatmap x0x1-y1"}
