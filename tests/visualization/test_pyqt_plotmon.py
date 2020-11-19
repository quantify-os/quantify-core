import numpy as np
from quantify.visualization import PlotMonitor_pyqt

from tests.helpers import get_test_data_dir
from quantify.data.types import TUID
from quantify.data.handling import set_datadir


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
        # Clear the state
        self.plotmon._last_tuid = None
        self.plotmon.tuid("latest")
        self.plotmon.num_persistent_dsets(3)

        tuid1 = "20200430-170837-001-315f36"  # 1D
        tuid2 = "20200504-191556-002-4209ee"  # 2D

        tuids = [tuid1, tuid2, tuid1, tuid2]
        hashes = [tuid.split("-")[-1] for tuid in tuids]

        for tuid in tuids:
            self.plotmon.tuid(tuid)
            self.plotmon.update()
        assert len(self.plotmon._persistent_dsets) == 3

        traces = self.plotmon.main_QtPlot.traces
        assert traces[0]["config"]["color"] == "#ff7f0e"

        labels_correct = [
            _hash in trace["config"]["name"]
            for trace, _hash in zip(traces, hashes + [hashes[1]] * 2)
        ]
        assert all(labels_correct)

        self.plotmon.tuid(tuid2)
        self.plotmon.update()
        assert len(self.plotmon._persistent_dsets) == 3

        self.plotmon.num_persistent_dsets(1)
        self.plotmon.tuid(tuid1)
        self.plotmon.update()
        assert len(self.plotmon._persistent_dsets) == 1
