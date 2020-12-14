import numpy as np
from quantify.visualization import PlotMonitor_pyqt

import pytest
from tests.helpers import get_test_data_dir
from quantify.data.types import TUID
from quantify.data.handling import set_datadir

test_datadir = get_test_data_dir()


class TestPlotMonitor_pyqt:
    @classmethod
    def setup_class(cls):
        # ensures the default datadir is used which is excluded from git
        set_datadir(test_datadir)
        # directory needs to be set before creating the plotting monitor
        # this avoids having to pass around the datadir between processes
        cls.plotmon = PlotMonitor_pyqt(name="plotmon")

    @classmethod
    def teardown_class(cls):
        cls.plotmon.close()
        set_datadir(None)

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

        time_tags = [":".join(tuid.split("-")[1][i : i + 2] for i in range(0, 6, 2)) for tuid in tuids]

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
