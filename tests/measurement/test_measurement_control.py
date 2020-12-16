import time
import random
import xarray as xr
import numpy as np
import pytest
import adaptive
from scipy import optimize
from qcodes import ManualParameter, Parameter
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from quantify.measurement.control import MeasurementControl, tile_setpoints_grid
from quantify.data.handling import set_datadir
from quantify.data.types import TUID
from quantify.visualization.pyqt_plotmon import PlotMonitor_pyqt
from quantify.visualization.instrument_monitor import InstrumentMonitor
from quantify.utilities.experiment_helpers import load_settings_onto_instrument
from tests.helpers import get_test_data_dir
try:
    from adaptive import SKOptLearner
    with_skoptlearner = True
except ImportError:
    with_skoptlearner = False

test_datadir = get_test_data_dir()


def CosFunc(t, amplitude, frequency, phase):
    """A simple cosine function"""
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


def SinFunc(t, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


# Parameters are created to emulate a system being measured
t = ManualParameter("t", initial_value=1, unit="s", label="Time")
amp = ManualParameter("amp", initial_value=1, unit="V", label="Amplitude")
freq = ManualParameter("freq", initial_value=1, unit="Hz", label="Frequency")


def sine_model():
    return SinFunc(t(), amplitude=amp(), frequency=freq(), phase=0)


def cosine_model():
    return CosFunc(t(), amplitude=amp(), frequency=freq(), phase=0)


# We wrap our function in a Parameter to be able to give
sig = Parameter(name="sig", label="Signal level", unit="V", get_cmd=cosine_model)


class DualWave:
    def __init__(self):
        self.name = ["sin", "cos"]
        self.unit = ["V", "V"]
        self.label = ["Sine", "Cosine"]

    def get(self):
        return np.array([sine_model(), cosine_model()])


class DummyParHolder(Instrument):
    def __init__(self, name):
        super().__init__(name)

        for parname in ["x", "y", "z", "x0", "y0", "z0"]:
            self.add_parameter(
                parname,
                unit="m",
                parameter_class=ManualParameter,
                vals=vals.Numbers(),
                initial_value=0.0,
            )

        self.add_parameter(
            "noise",
            unit="V",
            label="white noise amplitude",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        self.add_parameter(
            "delay",
            unit="s",
            label="Sampling delay",
            parameter_class=ManualParameter,
            vals=vals.Numbers(),
            initial_value=0,
        )

        self.add_parameter("parabola", unit="V", get_cmd=self._measure_parabola)

    def _measure_parabola(self):
        time.sleep(self.delay())
        return (
            (self.x() - self.x0()) ** 2
            + (self.y() - self.y0()) ** 2
            + (self.z() - self.z0()) ** 2
            + self.noise() * np.random.rand(1)
        )


def hardware_mock_values(setpoints):
    return np.sin(setpoints / np.pi)


class DummyHardwareSettable:
    def __init__(self):
        self.name = "DummyHardwareSettable"
        self.label = "Amp"
        self.unit = "V"
        self.batched = True
        self.setpoints = []

    # copy what qcodes does for easy testing interop
    def __call__(self):
        return self.setpoints

    def set(self, setpoints):
        self.setpoints = setpoints


class DummyHardwareGettable:
    def __init__(self, settables, noise=0.0):
        self.name = ["DummyHardwareGettable_0"]
        self.unit = ["W"]
        self.label = ["Watts"]
        self.batched = True
        self.settables = [settables] if not isinstance(settables, list) else settables
        self.noise = noise
        self.get_func = hardware_mock_values

    def set_return_2D(self):
        self.name.append("DummyHardwareGettable_1")
        self.unit.append("V")
        self.label.append("Amp")

    def prepare(self):
        assert len(self.settables) > 0
        for settable in self.settables:
            assert settable is not None

    def _get_data(self):
        return np.array([i() for i in self.settables])

    def get(self):
        data = self._get_data()
        data = self.get_func(data)
        noise = self.noise * (np.random.rand(1, data.shape[1]) - 0.5)
        data += noise
        return data[0 : len(self.name), :]


class TestMeasurementControl:
    @classmethod
    def setup_class(cls):
        cls.MC = MeasurementControl(name="MC")
        # ensures the default datadir is used which is excluded from git
        cls.dummy_parabola = DummyParHolder("parabola")
        set_datadir(None)

    @classmethod
    def teardown_class(cls):
        cls.MC.close()
        cls.dummy_parabola.close()
        set_datadir(None)

    def test_MeasurementControl_name(self):
        assert self.MC.name == "MC"

    def test_setpoints(self):
        x = np.linspace(0, 10, 11)
        self.MC.setpoints(x)
        assert np.array_equal(self.MC._setpoints[:, 0], x)

        x = np.random.rand(15, 2)
        self.MC.setpoints(x)
        assert np.array_equal(self.MC._setpoints, x)

        x = np.random.rand(15, 4)
        self.MC.setpoints(x)
        assert np.array_equal(self.MC._setpoints, x)

    def test_soft_sweep_1D(self):
        xvals = np.linspace(0, 2 * np.pi, 31)

        self.MC.settables(t)
        self.MC.setpoints(xvals)
        self.MC.gettables(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(t=xvals, amplitude=1, frequency=1, phase=0)

        assert np.array_equal(dset["x0"].values, xvals)
        assert np.array_equal(dset["y0"].values, expected_vals)

        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {"x0", "y0"}
        assert np.array_equal(dset["x0"], xvals)
        assert dset["x0"].attrs == {"name": "t", "long_name": "Time", "unit": "s"}
        assert dset["y0"].attrs == {
            "name": "sig",
            "long_name": "Signal level",
            "unit": "V",
        }

    def test_soft_sweep_1D_multi_return(self):
        xvals = np.linspace(0, 2 * np.pi, 31)

        self.MC.settables(t)
        self.MC.setpoints(xvals)
        self.MC.gettables(DualWave())
        dset = self.MC.run()

        exp_y0 = SinFunc(xvals, 1, 1, 0)
        exp_y1 = CosFunc(xvals, 1, 1, 0)

        assert dset.keys() == {"x0", "y0", "y1"}
        np.testing.assert_array_equal(dset["y0"], exp_y0)
        np.testing.assert_array_equal(dset["y1"], exp_y1)

    def test_soft_averages_soft_sweep_1D(self):
        def rand():
            return random.uniform(0.0, t())

        rand_get = Parameter(name="sig", label="Signal level", unit="V", get_cmd=rand)
        setpoints = np.arange(100.0)
        self.MC.settables(t)
        self.MC.setpoints(setpoints)
        self.MC.gettables(rand_get)
        r_dset = self.MC.run("random")

        self.MC.soft_avg(50)
        avg_dset = self.MC.run("averaged")

        expected_vals = 0.5 * np.arange(100.0)
        r_delta = abs(r_dset["y0"].values - expected_vals)
        avg_delta = abs(avg_dset["y0"].values - expected_vals)
        assert np.mean(avg_delta) < np.mean(r_delta)

    def test_hard_sweep_1D(self):
        x = np.linspace(0, 10, 5)
        device = DummyHardwareSettable()
        self.MC.settables(device)
        self.MC.setpoints(x)
        self.MC.gettables(DummyHardwareGettable(device))
        dset = self.MC.run()

        expected_vals = hardware_mock_values(x)
        assert np.array_equal(dset["x0"].values, x)
        assert np.array_equal(dset["y0"].values, expected_vals)

        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {"x0", "y0"}
        assert dset["x0"].attrs == {
            "name": "DummyHardwareSettable",
            "long_name": "Amp",
            "unit": "V",
        }
        assert dset["y0"].attrs == {
            "name": "DummyHardwareGettable_0",
            "long_name": "Watts",
            "unit": "W",
        }

    def test_soft_averages_hard_sweep_1D(self):
        setpoints = np.arange(50.0)
        settable = DummyHardwareSettable()
        gettable = DummyHardwareGettable(settable)
        gettable.noise = 0.4
        self.MC.settables(settable)
        self.MC.setpoints(setpoints)
        self.MC.gettables(gettable)
        noisy_dset = self.MC.run("noisy")
        xn_0 = noisy_dset["x0"].values
        expected_vals = hardware_mock_values(xn_0)
        yn_0 = abs(noisy_dset["y0"].values - expected_vals)

        self.MC.soft_avg(5000)
        avg_dset = self.MC.run("averaged")
        yavg_0 = abs(avg_dset["y0"].values - expected_vals)

        np.testing.assert_array_equal(xn_0, setpoints)
        assert np.mean(yn_0) > np.mean(yavg_0)
        np.testing.assert_array_almost_equal(yavg_0, np.zeros(len(xn_0)), decimal=2)

    def test_soft_set_hard_get_1D(self):
        gettable = DummyHardwareGettable(t)

        def mock_get():
            return np.sin(t())

        gettable.get = mock_get
        setpoints = np.linspace(0, 360, 8)
        self.MC.settables(t)
        self.MC.setpoints(setpoints)
        self.MC.gettables(gettable)
        dset = self.MC.run("soft_sweep_hard_det")

        x = dset["x0"].values
        y0 = dset["y0"].values
        np.testing.assert_array_equal(x, setpoints)
        np.testing.assert_array_equal(y0, np.sin(setpoints))

    def test_soft_sweep_2D_grid(self):

        times = np.linspace(0, 5, 20)
        amps = np.linspace(-1, 1, 5)

        self.MC.settables([t, amp])
        self.MC.setpoints_grid([times, amps])

        exp_sp = tile_setpoints_grid([times, amps])
        assert np.array_equal(self.MC._setpoints, exp_sp)

        self.MC.gettables(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(
            t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=1, phase=0
        )

        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])
        assert np.array_equal(dset["y0"].values, expected_vals)

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {"x0", "x1", "y0"}

        assert all(e in dset["x0"].values for e in times)
        assert all(e in dset["x1"].values for e in amps)

        assert dset["x0"].attrs == {"name": "t", "long_name": "Time", "unit": "s"}
        assert dset["x1"].attrs == {
            "name": "amp",
            "long_name": "Amplitude",
            "unit": "V",
        }
        assert dset["y0"].attrs == {
            "name": "sig",
            "long_name": "Signal level",
            "unit": "V",
        }

    def test_soft_sweep_2D_arbitrary(self):

        r = np.linspace(0, 1.5, 50)
        dt = np.linspace(0, 1, 50)

        f = 10

        # create a fancy polar coordinates loop
        theta = np.cos(2 * np.pi * f * dt)

        def polar_coords(r, theta):

            x = r * np.cos(2 * np.pi * theta)
            y = r * np.sin(2 * np.pi * theta)
            return x, y

        x, y = polar_coords(r, theta)
        setpoints = np.column_stack([x, y])

        self.MC.settables([t, amp])
        self.MC.setpoints(setpoints)
        self.MC.gettables(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(t=x, amplitude=y, frequency=1, phase=0)

        assert np.array_equal(dset["x0"].values, x)
        assert np.array_equal(dset["x1"].values, y)
        assert np.array_equal(dset["y0"].values, expected_vals)

    def test_hard_sweep_2D_grid(self):
        times = np.linspace(10, 20, 3)
        amps = np.linspace(0, 10, 5)

        settables = [DummyHardwareSettable(), DummyHardwareSettable()]
        gettable = DummyHardwareGettable(settables)
        self.MC.settables(settables)
        self.MC.setpoints_grid([times, amps])
        self.MC.gettables(gettable)
        dset = self.MC.run("2D Hard")

        exp_sp = tile_setpoints_grid([times, amps])
        assert np.array_equal(exp_sp, self.MC._setpoints)
        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])

        expected_vals = hardware_mock_values(dset["x0"].values)
        assert np.array_equal(dset["y0"].values, expected_vals)

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)

        assert dset["x0"].attrs == {
            "name": "DummyHardwareSettable",
            "long_name": "Amp",
            "unit": "V",
        }
        assert dset["x1"].attrs == {
            "name": "DummyHardwareSettable",
            "long_name": "Amp",
            "unit": "V",
        }
        assert dset["y0"].attrs == {
            "name": "DummyHardwareGettable_0",
            "long_name": "Watts",
            "unit": "W",
        }

    def test_hard_sweep_2D_grid_multi_return(self):
        times = np.linspace(10, 20, 3)
        amps = np.linspace(0, 10, 5)

        settables = [DummyHardwareSettable(), DummyHardwareSettable()]
        gettable = DummyHardwareGettable(settables)
        gettable.set_return_2D()
        self.MC.settables(settables)
        self.MC.setpoints_grid([times, amps])
        self.MC.gettables(gettable)
        dset = self.MC.run("2D Hard")

        exp_sp = tile_setpoints_grid([times, amps])
        assert np.array_equal(exp_sp, self.MC._setpoints)
        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])

        expected_vals = hardware_mock_values(
            np.stack((dset["x0"].values, dset["x1"].values))
        )
        assert np.array_equal(dset["y0"].values, expected_vals[0])
        assert np.array_equal(dset["y1"].values, expected_vals[1])

    def test_hard_sweep_2D_grid_multi_return_soft_avg(self):
        x0 = np.arange(5)
        x1 = np.linspace(5, 10, 5)
        settables = [DummyHardwareSettable(), DummyHardwareSettable()]
        gettable = DummyHardwareGettable(settables)
        gettable.noise = 0.4
        gettable.set_return_2D()
        self.MC.settables(settables)
        self.MC.setpoints_grid([x0, x1])
        self.MC.gettables(gettable)
        noisy_dset = self.MC.run("noisy_hard_grid")

        expected_vals = hardware_mock_values(
            np.stack((noisy_dset["x0"].values, noisy_dset["x1"]))
        )
        yn_0 = abs(noisy_dset["y0"].values - expected_vals[0])
        yn_1 = abs(noisy_dset["y1"].values - expected_vals[1])

        self.MC.soft_avg(1000)
        avg_dset = self.MC.run("avg_hard_grid")
        yavg_0 = abs(avg_dset["y0"].values - expected_vals[0])
        yavg_1 = abs(avg_dset["y1"].values - expected_vals[1])

        assert np.mean(yavg_0) < np.mean(yn_0)
        assert np.mean(yavg_1) < np.mean(yn_1)
        np.testing.assert_array_almost_equal(
            yavg_0, np.zeros(len(noisy_dset["x0"].values)), decimal=2
        )
        np.testing.assert_array_almost_equal(
            yavg_1, np.zeros(len(noisy_dset["x0"].values)), decimal=2
        )

    def test_hard_sweep_2D_arbitrary(self):
        r = np.linspace(0, 1.5, 5)
        dt = np.linspace(0, 1, 5)
        f = 10
        theta = np.cos(2 * np.pi * f * dt)

        def polar_coords(r, theta):
            x = r * np.cos(2 * np.pi * theta)
            y = r * np.sin(2 * np.pi * theta)
            return x, y

        x, y = polar_coords(r, theta)
        setpoints = np.column_stack([x, y])

        settables = [DummyHardwareSettable(), DummyHardwareSettable()]
        self.MC.settables(settables)
        self.MC.setpoints(setpoints)
        self.MC.gettables(DummyHardwareGettable(settables))
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = hardware_mock_values(x)

        assert np.array_equal(dset["x0"].values, x)
        assert np.array_equal(dset["x1"].values, y)
        assert np.array_equal(dset["y0"].values, expected_vals)

    def test_variable_return_hard_sweep(self):
        counter_param = ManualParameter("counter", initial_value=0)

        def v_size(setpoints):
            idx = counter_param() % 3
            counter_param(counter_param() + 1)
            if idx == 0:
                return 2 * setpoints[:7]
            elif idx == 1:
                return 2 * setpoints[:4]
            elif idx == 2:
                return 2 * setpoints[:]

        setpoints = np.arange(30.0)
        settable = DummyHardwareSettable()
        gettable = DummyHardwareGettable(settable)
        gettable.get_func = v_size
        self.MC.settables(settable)
        self.MC.setpoints(setpoints)
        self.MC.gettables(gettable)
        dset = self.MC.run("varying")

        assert np.array_equal(dset["x0"], setpoints)
        assert np.array_equal(dset["y0"], 2 * setpoints)

    def test_variable_return_hard_sweep_soft_avg(self):
        counter_param = ManualParameter("counter", initial_value=0)

        def v_size(setpoints):
            idx = counter_param() % 3
            counter_param(counter_param() + 1)
            if idx == 0:
                return 2 * setpoints[:7]
            elif idx == 1:
                return 2 * setpoints[:4]
            elif idx == 2:
                return 2 * setpoints[:]

        setpoints = np.arange(30.0)
        settable = DummyHardwareSettable()
        gettable = DummyHardwareGettable(settable)
        gettable.get_func = v_size
        gettable.noise = 0.25
        self.MC.settables(settable)
        self.MC.setpoints(setpoints)
        self.MC.gettables(gettable)
        self.MC.soft_avg(5000)
        dset = self.MC.run("varying")

        assert np.array_equal(dset["x0"], setpoints)
        np.testing.assert_array_almost_equal(dset.y0, 2 * setpoints, decimal=2)

    def test_soft_sweep_3D_grid(self):
        times = np.linspace(0, 5, 2)
        amps = np.linspace(-1, 1, 3)
        freqs = np.linspace(41000, 82000, 2)

        self.MC.settables([t, amp, freq])
        self.MC.setpoints_grid([times, amps, freqs])

        exp_sp = tile_setpoints_grid([times, amps, freqs])
        assert np.array_equal(self.MC._setpoints, exp_sp)

        self.MC.gettables(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(
            t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=exp_sp[:, 2], phase=0
        )

        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])
        assert np.array_equal(dset["x2"].values, exp_sp[:, 2])
        assert np.array_equal(dset["y0"].values, expected_vals)

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {"x0", "x1", "x2", "y0"}
        assert all(e in dset["x0"] for e in times)
        assert all(e in dset["x1"] for e in amps)
        assert all(e in dset["x2"] for e in freqs)

        assert dset["x0"].attrs == {"name": "t", "long_name": "Time", "unit": "s"}
        assert dset["x2"].attrs == {
            "name": "freq",
            "long_name": "Frequency",
            "unit": "Hz",
        }
        assert dset["y0"].attrs == {
            "name": "sig",
            "long_name": "Signal level",
            "unit": "V",
        }

    def test_soft_sweep_3D_multi_return(self):
        times = np.linspace(0, 5, 2)
        amps = np.linspace(-1, 1, 3)
        freqs = np.linspace(41000, 82000, 2)

        self.MC.settables([t, amp, freq])
        self.MC.setpoints_grid([times, amps, freqs])
        self.MC.gettables([DualWave(), sig, DualWave()])
        dset = self.MC.run()

        exp_sp = tile_setpoints_grid([times, amps, freqs])
        exp_y0 = exp_y3 = SinFunc(
            t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=exp_sp[:, 2], phase=0
        )
        exp_y1 = exp_y2 = exp_y4 = CosFunc(
            t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=exp_sp[:, 2], phase=0
        )

        np.testing.assert_array_equal(dset["y0"], exp_y0)
        np.testing.assert_array_equal(dset["y1"], exp_y1)
        np.testing.assert_array_equal(dset["y2"], exp_y2)
        np.testing.assert_array_equal(dset["y3"], exp_y3)
        np.testing.assert_array_equal(dset["y4"], exp_y4)

    def test_hard_sweep_3D_multi_return_soft_averaging(self):
        def v_get(setpoints):
            return setpoints

        times = np.linspace(0, 5, 2)
        amps = np.linspace(-1, 1, 3)
        freqs = np.linspace(41000, 82000, 2)
        hardware_settable_0 = DummyHardwareSettable()
        hardware_settable_1 = DummyHardwareSettable()
        nd_gettable = DummyHardwareGettable([hardware_settable_0, hardware_settable_1])
        nd_gettable.set_return_2D()
        noisy_gettable = DummyHardwareGettable([t])
        noisy_gettable.get_func = v_get
        noisy_gettable.noise = 0.25

        self.MC.settables([t, hardware_settable_0, hardware_settable_1])
        self.MC.setpoints_grid([times, amps, freqs])
        self.MC.gettables([noisy_gettable, nd_gettable])
        self.MC.soft_avg(5000)
        dset = self.MC.run()

        exp_sp = tile_setpoints_grid([times, amps, freqs])
        np.testing.assert_array_almost_equal(dset.y0, exp_sp[:, 0], decimal=2)
        np.testing.assert_array_almost_equal(
            dset.y1, hardware_mock_values(exp_sp[:, 1]), decimal=4
        )
        np.testing.assert_array_almost_equal(
            dset.y2, hardware_mock_values(exp_sp[:, 2]), decimal=4
        )

    def test_adaptive_no_averaging(self):
        self.MC.soft_avg(5)
        with pytest.raises(
            ValueError,
            match=r"software averaging not allowed in adaptive loops; currently set to 5",
        ):
            self.MC.run_adaptive("fail", {})
        self.MC.soft_avg(1)

    def test_adaptive_nelder_mead(self):
        self.MC.settables([self.dummy_parabola.x, self.dummy_parabola.y])
        af_pars = {
            "adaptive_function": optimize.minimize,
            "x0": [-50, -50],
            "method": "Nelder-Mead",
        }
        self.dummy_parabola.noise(0.5)
        self.MC.gettables(self.dummy_parabola.parabola)
        dset = self.MC.run_adaptive("nelder_mead", af_pars)

        assert dset["x0"][-1] < 0.7
        assert dset["x1"][-1] < 0.7
        assert dset["y0"][-1] < 0.7

    def test_adaptive_multi_return(self):
        freq = ManualParameter(name="frequency", unit="Hz", label="Frequency")
        amp = ManualParameter(name="amp", unit="V", label="Amplitude", initial_value=1)
        dummy = ManualParameter(name="dummy", unit="V", label="Dummy", initial_value=99)
        fwhm = 300
        resonance_freq = random.uniform(6e9, 7e9)

        def lorenz():
            return (
                amp()
                * ((fwhm / 2.0) ** 2)
                / ((freq() - resonance_freq) ** 2 + (fwhm / 2.0) ** 2)
            )

        # gimmick class for return
        class ResonancePlus:
            def __init__(self):
                self.name = ["resonance", "dummy"]
                self.label = ["Amplitude", "Dummy"]
                self.unit = ["V", "V"]

            def get(self):
                return [lorenz(), dummy()]

        self.MC.settables(freq)
        af_pars = {
            "adaptive_function": adaptive.learner.Learner1D,
            "goal": lambda l: l.npoints > 5,
            "bounds": (6e9, 7e9),
        }
        self.MC.gettables([ResonancePlus(), dummy])
        dset = self.MC.run_adaptive("multi return", af_pars)
        assert (dset["y1"].values == 99).all()
        assert (dset["y2"].values == 99).all()

    def test_adaptive_bounds_1D(self):
        freq = ManualParameter(name="frequency", unit="Hz", label="Frequency")
        amp = ManualParameter(name="amp", unit="V", label="Amplitude", initial_value=1)
        fwhm = 300
        resonance_freq = random.uniform(6e9, 7e9)

        def lorenz():
            return (
                amp()
                * ((fwhm / 2.0) ** 2)
                / ((freq() - resonance_freq) ** 2 + (fwhm / 2.0) ** 2)
            )

        resonance = Parameter("resonance", unit="V", label="Amplitude", get_cmd=lorenz)

        self.MC.settables(freq)
        af_pars = {
            "adaptive_function": adaptive.learner.Learner1D,
            "goal": lambda l: l.npoints > 20 * 20,
            "bounds": (6e9, 7e9),
        }
        self.MC.gettables(resonance)
        dset = self.MC.run_adaptive("adaptive sample", af_pars)
        print("jej")

    def test_adaptive_sampling(self):
        self.dummy_parabola.noise(0)
        self.MC.settables([self.dummy_parabola.x, self.dummy_parabola.y])
        af_pars = {
            "adaptive_function": adaptive.learner.Learner2D,
            "goal": lambda l: l.npoints > 20 * 20,
            "bounds": ((-50, 50), (-20, 30)),
        }
        self.MC.gettables(self.dummy_parabola.parabola)
        dset = self.MC.run_adaptive("adaptive sample", af_pars)
        # todo pycqed has no verification step here, what should we do?

    @pytest.mark.skipif(not with_skoptlearner, reason="scikit-optimize is not installed")
    def test_adaptive_skoptlearner(self):
        self.dummy_parabola.noise(0)
        self.MC.settables([self.dummy_parabola.x, self.dummy_parabola.y])
        af_pars = {
            "adaptive_function": SKOptLearner,
            "goal": lambda l: l.npoints > 15,
            "dimensions": [(-50.0, +50.0), (-20.0, +30.0)],
            "base_estimator": "gp",
            "acq_func": "EI",
            "acq_optimizer": "lbfgs",
        }
        self.MC.gettables(self.dummy_parabola.parabola)
        dset = self.MC.run_adaptive("skopt", af_pars)
        # todo pycqed has no verification step here, what should we do?

    def test_progress_callback(self):
        progress_param = ManualParameter("progress", initial_value=0)

        def set_progress_param_callable(progress):
            progress_param(progress)

        self.MC.on_progress_callback(set_progress_param_callable)
        assert progress_param() == 0

        xvals = np.linspace(0, 2 * np.pi, 31)
        self.MC.settables(t)
        self.MC.setpoints(xvals)
        self.MC.gettables(sig)
        self.MC.run()

        assert progress_param() == 100

    def test_MC_plotmon_integration(self):
        plotmon = PlotMonitor_pyqt("plotmon_MC")

        self.MC.instr_plotmon(plotmon.name)

        assert len(plotmon.tuids()) == 0

        times = np.linspace(0, 5, 18)
        amps = np.linspace(-1, 1, 5)

        self.MC.settables([t, amp])
        self.MC.setpoints_grid([times, amps])
        self.MC.gettables(sig)
        self.MC.run("2D Cosine test")

        assert len(plotmon.tuids()) > 0

        plotmon.close()
        self.MC.instr_plotmon("")

    def test_MC_insmon_integration(self):
        inst_mon = InstrumentMonitor("insmon_MC")
        self.MC.instrument_monitor(inst_mon.name)
        assert self.MC.instrument_monitor.get_instr().tree.getNodes()
        inst_mon.close()
        self.MC.instrument_monitor("")

    def test_instrument_settings_from_disk(self):
        load_settings_onto_instrument(
            self.dummy_parabola, TUID("20200814-134652-492-fbf254"), test_datadir
        )
        assert self.dummy_parabola.x() == 40.0
        assert self.dummy_parabola.y() == 90.0
        assert self.dummy_parabola.z() == -20.0
        assert self.dummy_parabola.delay() == 10.0

        non_existing = DummyParHolder("the mac")
        with pytest.raises(
            ValueError, match='Instrument "the mac" not found in snapshot'
        ):
            load_settings_onto_instrument(
                non_existing, TUID("20200814-134652-492-fbf254"), test_datadir
            )

        non_existing.close()

    def test_exception_not_silenced_set_get(self):
        class badSetter:
            def __init__(self, param):
                self.name = "spec_freq"
                self.label = "Frequency"
                self.unit = "Hz"
                self.soft = True
                # simulate user mistake
                # self.non_existing_param = param
                pass

            def prepare(self):
                # Exception expected
                print(self.non_existing_param)

            def set(self, val):
                pass

        self.MC.settables(badSetter("badSetter"))
        self.MC.setpoints(np.linspace(0, 1, 10))
        self.MC.gettables(freq)
        with pytest.raises(
            AttributeError,
            match="'badSetter' object has no attribute 'non_existing_param'",
        ):
            self.MC.run("This rises exception as expected")

        class badGetter:
            def __init__(self, param2):
                self.name = "mag"
                self.label = "Magnitude"
                self.unit = "a.u."
                self.soft = True
                # self.non_existing_param = param2

            def prepare(self):
                pass

            def get(self):
                print(self.non_existing_param)

            def finish(self):
                pass

        self.MC.settables(t)
        self.MC.setpoints(np.linspace(0, 1, 10) * 1e-9)
        self.MC.gettables(badGetter("badGetter"))
        with pytest.raises(
            AttributeError,
            match="'badGetter' object has no attribute 'non_existing_param'",
        ):
            self.MC.run("This rises exception as expected")


def test_tile_setpoints_grid():
    x = np.arange(5)
    y = np.linspace(-1, 1, 3)

    sp = tile_setpoints_grid([x, y])
    assert sp[:, 0].all() == np.tile(np.arange(5), 3).all()
    assert sp[:, 1].all() == np.repeat(y, 5).all()

    z = np.linspace(100, 200, 2)
    sp = tile_setpoints_grid([x, y, z])
    assert all(e in sp[:, 0] for e in x)
    assert all(e in sp[:, 1] for e in y)
    assert all(e in sp[:, 2] for e in z)
