# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
import tempfile
from distutils.dir_util import copy_tree
from pathlib import Path

import numpy as np
import pytest

import quantify_core.data.handling as dh
from quantify_core.data.types import TUID
from quantify_core.visualization import PlotMonitor_pyqt


@pytest.fixture(scope="function")
def plotmon_instance(request, tmp_test_data_dir):

    dh.set_datadir(tmp_test_data_dir)
    plotmon = PlotMonitor_pyqt(name="plotmon")

    def reset_datadir():
        dh._datadir = None

    request.addfinalizer(plotmon.close)
    request.addfinalizer(reset_datadir)

    return plotmon


def test_attributes_created_during_init(plotmon_instance):
    hasattr(plotmon_instance, "main_QtPlot")
    hasattr(plotmon_instance, "secondary_QtPlot")


def test_validator_accepts_tuid_objects(plotmon_instance):
    plotmon_instance.tuids_append(TUID("20200430-170837-001-315f36"))
    plotmon_instance.tuids([])  # reset for next tests


def test_basic_1d_plot(plotmon_instance):
    plotmon_instance.tuids_max_num(1)
    # Test 1D plotting using an example dataset
    tuid = "20200430-170837-001-315f36"
    plotmon_instance.tuids_append(tuid)
    plotmon_instance.update()

    curves_dict = plotmon_instance._get_curves_config()
    x = curves_dict[tuid]["x0y0"]["config"]["x"]
    y = curves_dict[tuid]["x0y0"]["config"]["y"]

    x_exp = np.linspace(0, 5, 50)
    y_exp = np.cos(np.pi * x_exp)
    np.testing.assert_allclose(x, x_exp)
    np.testing.assert_allclose(y, y_exp)
    plotmon_instance.tuids([])  # reset for next tests


def test_basic_2d_plot(plotmon_instance):
    plotmon_instance.tuids_max_num(1)
    # Test 1D plotting using an example dataset
    tuid = "20200504-191556-002-4209ee"
    plotmon_instance.tuids_append(tuid)
    plotmon_instance.update()

    curves_dict = plotmon_instance._get_curves_config()
    x = curves_dict[tuid]["x0y0"]["config"]["x"]
    y = curves_dict[tuid]["x0y0"]["config"]["y"]

    x_exp = np.linspace(0, 5, 50)
    y_exp = np.cos(np.pi * x_exp) * -1
    np.testing.assert_allclose(x[:50], x_exp)
    np.testing.assert_allclose(y[:50], y_exp)

    cfg = plotmon_instance._get_traces_config(which="secondary_QtPlot")[0]["config"]
    assert np.shape(cfg["z"]) == (11, 50)
    assert cfg["xlabel"] == "Time"
    assert cfg["xunit"] == "s"
    assert cfg["ylabel"] == "Amplitude"
    assert cfg["yunit"] == "V"
    assert cfg["zlabel"] == "Signal level"
    assert cfg["zunit"] == "V"
    plotmon_instance.tuids([])  # reset for next tests


def test_persistence(plotmon_instance):
    """
    NB this test reuses same too datasets, ideally the user will
    never do this
    """
    # Clear the state to keep this test independent
    plotmon_instance.tuids([])
    plotmon_instance.tuids_max_num(3)

    tuids = [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
        "20201124-184722-988-0463d4",
        "20201124-184729-618-85970f",
        "20201124-184736-341-3628d4",
    ]

    time_tags = [
        ":".join(tuid.split("-")[1][i : i + 2] for i in range(0, 6, 2))
        for tuid in tuids
    ]

    for tuid in tuids:
        plotmon_instance.tuids_append(tuid)

    # Update a few times to mimic MC integration
    _ = [plotmon_instance.update() for i in range(3)]

    # Confirm persistent datasets are being accumulated
    assert plotmon_instance.tuids()[::-1] == tuids[2:]

    traces = plotmon_instance._get_traces_config(which="main_QtPlot")
    # this is a bit lazy to not deal with the indices of all traces
    # of all plots
    names = set(trace["config"]["name"] for trace in traces)
    labels_exist = [any(tt in name for name in names) for tt in time_tags[2:]]
    assert all(labels_exist)

    plotmon_instance.tuids_append(tuids[0])
    assert plotmon_instance.tuids()[0] == tuids[0]
    # Confirm maximum accumulation works
    assert len(plotmon_instance.tuids()) == 3

    # the latest dataset is always blue circle
    traces = plotmon_instance._get_traces_config(which="main_QtPlot")
    assert traces[-1]["config"]["color"] == (31, 119, 180, 255)
    assert traces[-1]["config"]["symbol"] == "o"

    # test reset works
    plotmon_instance.tuids([])
    assert plotmon_instance.tuids() == []

    plotmon_instance.tuids_extra(tuids[1:3])
    assert len(plotmon_instance.tuids_extra()) == 2
    traces = plotmon_instance._get_traces_config(which="main_QtPlot")
    assert len(traces) > 0

    plotmon_instance.tuids_extra([])
    assert plotmon_instance.tuids_extra() == []

    plotmon_instance.tuids([])  # reset for next tests


def test_set_geometry(plotmon_instance):
    # N.B. x an y are absolute, OS docs or menu bars might prevent certain positions
    xywh = (400, 400, 300, 400)
    xywh_init_main = plotmon_instance._remote_plotmon._get_qt_plot_geometry(
        which="main_QtPlot"
    )
    xywh_init_sec = plotmon_instance._remote_plotmon._get_qt_plot_geometry(
        which="secondary_QtPlot"
    )

    plotmon_instance.setGeometry_main(*xywh)
    xywh_new = plotmon_instance._remote_plotmon._get_qt_plot_geometry(
        which="main_QtPlot"
    )
    assert xywh_new != xywh_init_main

    plotmon_instance.setGeometry_secondary(*xywh)
    xywh_new = plotmon_instance._remote_plotmon._get_qt_plot_geometry(
        which="secondary_QtPlot"
    )
    assert xywh_new != xywh_init_sec


def test_changed_datadir_main_process(plotmon_instance):
    # This test ensures that the remote process always uses the same datadir
    # even when it is changed in the main process
    plotmon_instance.tuids([])  # reset
    plotmon_instance.tuids_extra([])  # reset

    # load dataset in main process
    tuid = "20201124-184709-137-8a5112"

    # change datadir in the main process
    tmp_dir = tempfile.TemporaryDirectory()
    dir_to_copy = Path(
        dh._locate_experiment_file(tuid=tuid, datadir=dh.get_datadir(), name="")
    ).parent
    dh.set_datadir(tmp_dir.name)
    daydir = Path(tmp_dir.name) / dir_to_copy.name
    Path.mkdir(daydir)
    copy_tree(dir_to_copy, str(daydir))

    # .update()
    plotmon_instance.update(tuid)
    plotmon_instance._remote_plotmon._exec_queue()
    assert (
        tuid == tuple(plotmon_instance._remote_plotmon._dsets.values())[0].attrs["tuid"]
    )
    plotmon_instance.tuids([])  # reset

    # .tuids()
    plotmon_instance.tuids([tuid])
    plotmon_instance._remote_plotmon._exec_queue()
    assert (
        tuid == tuple(plotmon_instance._remote_plotmon._dsets.values())[0].attrs["tuid"]
    )
    plotmon_instance.tuids([])  # reset

    # .tuids_append()
    plotmon_instance.tuids_append(tuid)
    plotmon_instance._remote_plotmon._exec_queue()
    assert (
        tuid == tuple(plotmon_instance._remote_plotmon._dsets.values())[0].attrs["tuid"]
    )
    plotmon_instance.tuids([])  # reset

    # .tuids_extra()
    plotmon_instance.tuids_extra([tuid])
    plotmon_instance._remote_plotmon._exec_queue()
    assert (
        tuid == tuple(plotmon_instance._remote_plotmon._dsets.values())[0].attrs["tuid"]
    )

    tmp_dir.cleanup()
