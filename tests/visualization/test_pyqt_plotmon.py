import numpy as np
from quantify.visualization import PlotMonitor_pyqt

import pytest
from tests.helpers import get_test_data_dir
from quantify.data.types import TUID
from quantify.data.handling import set_datadir
from quantify.visualization.color_utilities import darker_color_cycle

test_datadir = get_test_data_dir()


class TestPlotMonitor_pyqt:
    @classmethod
    def setup_class(cls):
        cls.plotmon = PlotMonitor_pyqt(name="plotmon")
        # ensures the default datadir is used which is excluded from git
        set_datadir(test_datadir)

    @classmethod
    def teardown_class(cls):
        cls.plotmon.close()
        set_datadir(None)

    def test_attributes_created_during_init(self):
        hasattr(self.plotmon, "main_QtPlot")
        hasattr(self.plotmon, "secondary_QtPlot")

    def test_validator_accepts_TUID_objects(self):
        self.plotmon.tuid(TUID("20200430-170837-001-315f36"))

    def test_basic_1D_plot(self):
        self.plotmon.max_num_previous_dsets(0)
        # Test 1D plotting using an example dataset
        self.plotmon.tuid("20200430-170837-001-315f36")
        self.plotmon.update()

        x = self.plotmon.curves[0]["config"]["x"]
        y = self.plotmon.curves[0]["config"]["y"]

        x_exp = np.linspace(0, 5, 50)
        y_exp = np.cos(np.pi * x_exp)
        np.testing.assert_allclose(x, x_exp)
        np.testing.assert_allclose(y, y_exp)

    def test_basic_2D_plot(self):
        self.plotmon.max_num_previous_dsets(0)
        # Test 1D plotting using an example dataset
        self.plotmon.tuid("20200504-191556-002-4209ee")
        self.plotmon.update()

        x = self.plotmon.curves[0]["config"]["x"]
        y = self.plotmon.curves[0]["config"]["y"]

        x_exp = np.linspace(0, 5, 50)
        y_exp = np.cos(np.pi * x_exp) * -1
        np.testing.assert_allclose(x[:50], x_exp)
        np.testing.assert_allclose(y[:50], y_exp)

        cfg = self.plotmon.secondary_QtPlot.traces[0]["config"]
        assert np.shape(cfg["z"]) == (11, 50)
        assert cfg["xlabel"] == "Time"
        assert cfg["xunit"] == "s"
        assert cfg["ylabel"] == "Amplitude"
        assert cfg["yunit"] == "V"
        assert cfg["zlabel"] == "Signal level"
        assert cfg["zunit"] == "V"

    def test_persistence(self):
        """
        NB this test reuses same too datasets, ideally the user will
        never do this
        """
        # Clear the state to keep this test independent
        self.plotmon.max_num_previous_dsets(3)

        tuid1 = "20200430-170837-001-315f36"  # 1D
        tuid2 = "20200504-191556-002-4209ee"  # 2D

        tuids = [
            "20201124-184709-137-8a5112",
            "20201124-184716-237-918bee",
            "20201124-184722-988-0463d4",
            "20201124-184729-618-85970f",
            "20201124-184736-341-3628d4",
        ]

        hashes = [tuid.split("-")[-1] for tuid in tuids]

        for tuid in tuids:
            self.plotmon.tuid(tuid)

        # Update a few times to mimic MC integration
        [self.plotmon.update() for i in range(3)]

        # Confirm persistent datasets are being accumulated
        assert list(self.plotmon.previous_tuids()) == tuids[1:-1]
        assert len(self.plotmon._previous_dsets) == 3

        traces = self.plotmon.main_QtPlot.traces
        # this is a bit lazy to not deal with the indices of all traces
        # of all plots
        names = set(trace["config"]["name"] for trace in traces)
        labels_exist = [
            any(_hash in name for name in names)
            for _hash in hashes[1:]
        ]
        assert all(labels_exist)

        self.plotmon.tuid(tuids[-1])
        assert self.plotmon.previous_tuids()[-1] == tuids[-1]
        # Confirm maximum accumulation works
        assert len(self.plotmon._previous_dsets) == 3

        # the latest dataset is always blue circle
        traces = self.plotmon.main_QtPlot.traces
        assert traces[-1]["config"]["color"] == darker_color_cycle[0]
        assert traces[-1]["config"]["symbol"] == "o"

        # test that reset works
        self.plotmon.max_num_previous_dsets(0)
        assert len(self.plotmon._previous_dsets) == 0
        traces = self.plotmon.main_QtPlot.traces
        len_tr = len(traces)
        assert len_tr == 2

        # test reset works
        self.plotmon.persistent_tuids(tuids[0:2])
        assert len(self.plotmon._persistent_dsets) == 2
        traces = self.plotmon.main_QtPlot.traces
        assert len(traces) > 0
        self.plotmon.persistent_tuids([])
        assert len(self.plotmon._persistent_dsets) == 0

        with pytest.raises(NotImplementedError, match=r"Datasets with different x and/or y variables not supported"):
            # Datasets with distinct xi and/or yi variables not supported
            self.plotmon.persistent_tuids([tuid1, tuid2])
