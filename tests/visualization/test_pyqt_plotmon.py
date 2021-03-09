import tempfile
from pathlib import Path
from distutils.dir_util import copy_tree

import numpy as np
from quantify.visualization import PlotMonitor_pyqt
from quantify.utilities._tests_helpers import get_test_data_dir
from quantify.data.types import TUID
import quantify.data.handling as dh

test_datadir = get_test_data_dir()


class TestPlotMonitor_pyqt:
    @classmethod
    def setup_class(cls):
        # ensures the default datadir is used which is excluded from git
        dh.set_datadir(test_datadir)
        # directory needs to be set before creating the plotting monitor
        # this avoids having to pass around the datadir between processes
        cls.plotmon = PlotMonitor_pyqt(name="plotmon")

    @classmethod
    def teardown_class(cls):
        cls.plotmon.close()
        dh._datadir = None

    def test_attributes_created_during_init(self):
        hasattr(self.plotmon, "main_QtPlot")
        hasattr(self.plotmon, "secondary_QtPlot")

    def test_validator_accepts_TUID_objects(self):
        self.plotmon.tuids_append(TUID("20200430-170837-001-315f36"))
        self.plotmon.tuids([])  # reset for next tests

    def test_basic_1D_plot(self):
        self.plotmon.tuids_max_num(1)
        # Test 1D plotting using an example dataset
        tuid = "20200430-170837-001-315f36"
        self.plotmon.tuids_append(tuid)
        self.plotmon.update()

        curves_dict = self.plotmon._get_curves_config()
        x = curves_dict[tuid]["x0y0"]["config"]["x"]
        y = curves_dict[tuid]["x0y0"]["config"]["y"]

        x_exp = np.linspace(0, 5, 50)
        y_exp = np.cos(np.pi * x_exp)
        np.testing.assert_allclose(x, x_exp)
        np.testing.assert_allclose(y, y_exp)
        self.plotmon.tuids([])  # reset for next tests

    def test_basic_2D_plot(self):
        self.plotmon.tuids_max_num(1)
        # Test 1D plotting using an example dataset
        tuid = "20200504-191556-002-4209ee"
        self.plotmon.tuids_append(tuid)
        self.plotmon.update()

        curves_dict = self.plotmon._get_curves_config()
        x = curves_dict[tuid]["x0y0"]["config"]["x"]
        y = curves_dict[tuid]["x0y0"]["config"]["y"]

        x_exp = np.linspace(0, 5, 50)
        y_exp = np.cos(np.pi * x_exp) * -1
        np.testing.assert_allclose(x[:50], x_exp)
        np.testing.assert_allclose(y[:50], y_exp)

        cfg = self.plotmon._get_traces_config(which="secondary_QtPlot")[0]["config"]
        assert np.shape(cfg["z"]) == (11, 50)
        assert cfg["xlabel"] == "Time"
        assert cfg["xunit"] == "s"
        assert cfg["ylabel"] == "Amplitude"
        assert cfg["yunit"] == "V"
        assert cfg["zlabel"] == "Signal level"
        assert cfg["zunit"] == "V"
        self.plotmon.tuids([])  # reset for next tests

    def test_persistence(self):
        """
        NB this test reuses same too datasets, ideally the user will
        never do this
        """
        # Clear the state to keep this test independent
        self.plotmon.tuids([])
        self.plotmon.tuids_max_num(3)

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
            self.plotmon.tuids_append(tuid)

        # Update a few times to mimic MC integration
        [self.plotmon.update() for i in range(3)]

        # Confirm persistent datasets are being accumulated
        assert self.plotmon.tuids()[::-1] == tuids[2:]

        traces = self.plotmon._get_traces_config(which="main_QtPlot")
        # this is a bit lazy to not deal with the indices of all traces
        # of all plots
        names = set(trace["config"]["name"] for trace in traces)
        labels_exist = [any(tt in name for name in names) for tt in time_tags[2:]]
        assert all(labels_exist)

        self.plotmon.tuids_append(tuids[0])
        assert self.plotmon.tuids()[0] == tuids[0]
        # Confirm maximum accumulation works
        assert len(self.plotmon.tuids()) == 3

        # the latest dataset is always blue circle
        traces = self.plotmon._get_traces_config(which="main_QtPlot")
        assert traces[-1]["config"]["color"] == (31, 119, 180, 255)
        assert traces[-1]["config"]["symbol"] == "o"

        # test reset works
        self.plotmon.tuids([])
        assert self.plotmon.tuids() == []

        self.plotmon.tuids_extra(tuids[1:3])
        assert len(self.plotmon.tuids_extra()) == 2
        traces = self.plotmon._get_traces_config(which="main_QtPlot")
        assert len(traces) > 0

        self.plotmon.tuids_extra([])
        assert self.plotmon.tuids_extra() == []

        # This does not work for now because this logic is now in another
        # process, some more work is needed to propagate exceptions
        # This is not a severe one though, the code will do nothing
        # and keep working
        # tuid1 = "20200430-170837-001-315f36"  # 1D
        # tuid2 = "20200504-191556-002-4209ee"  # 2D
        # with pytest.raises(
        #     NotImplementedError,
        #     match=r"Datasets with different x and/or y variables not supported",
        # ):
        #     # Datasets with distinct xi and/or yi variables not supported
        #     self.plotmon.tuids_extra([tuid1, tuid2])

        # with pytest.raises(
        #     NotImplementedError,
        #     match=r"Datasets with different x and/or y variables not supported",
        # ):
        #     # Datasets with distinct xi and/or yi variables not supported
        #     self.plotmon.tuids([tuid1, tuid2])

        self.plotmon.tuids([])  # reset for next tests

    def test_setGeometry(self):
        # N.B. x an y are absolute, OS docs or menu bars might prevent certain positions
        xywh = (400, 400, 300, 400)
        xywh_init_main = self.plotmon.remote_plotmon._get_QtPlot_geometry(
            which="main_QtPlot"
        )
        xywh_init_sec = self.plotmon.remote_plotmon._get_QtPlot_geometry(
            which="secondary_QtPlot"
        )

        self.plotmon.setGeometry_main(*xywh)
        xywh_new = self.plotmon.remote_plotmon._get_QtPlot_geometry(which="main_QtPlot")
        assert xywh_new != xywh_init_main

        self.plotmon.setGeometry_secondary(*xywh)
        xywh_new = self.plotmon.remote_plotmon._get_QtPlot_geometry(
            which="secondary_QtPlot"
        )
        assert xywh_new != xywh_init_sec

    def test_changed_datadir_main_process(self):
        # This test ensures that the remote process always uses the same datadir
        # even when it is changed in the main process
        self.plotmon.tuids([])  # reset
        self.plotmon.tuids_extra([])  # reset

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
        self.plotmon.update(tuid)
        self.plotmon.remote_plotmon._exec_queue()
        assert (
            tuid == tuple(self.plotmon.remote_plotmon._dsets.values())[0].attrs["tuid"]
        )
        self.plotmon.tuids([])  # reset

        # .tuids()
        self.plotmon.tuids([tuid])
        self.plotmon.remote_plotmon._exec_queue()
        assert (
            tuid == tuple(self.plotmon.remote_plotmon._dsets.values())[0].attrs["tuid"]
        )
        self.plotmon.tuids([])  # reset

        # .tuids_append()
        self.plotmon.tuids_append(tuid)
        self.plotmon.remote_plotmon._exec_queue()
        assert (
            tuid == tuple(self.plotmon.remote_plotmon._dsets.values())[0].attrs["tuid"]
        )
        self.plotmon.tuids([])  # reset

        # .tuids_extra()
        self.plotmon.tuids_extra([tuid])
        self.plotmon.remote_plotmon._exec_queue()
        assert (
            tuid == tuple(self.plotmon.remote_plotmon._dsets.values())[0].attrs["tuid"]
        )

        tmp_dir.cleanup()
