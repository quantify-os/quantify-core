# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import os
import random
import signal
import sys
import tempfile
import time
from collections.abc import Iterable
from unittest.mock import Mock

import adaptive
import numpy as np
import pytest
import xarray as xr
from qcodes import ManualParameter, Parameter
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from scipy import optimize

import quantify_core.data.handling as dh
from quantify_core.data.types import TUID
from quantify_core.measurement.control import MeasurementControl, grid_setpoints
from quantify_core.utilities.experiment_helpers import load_settings_onto_instrument
from quantify_core.visualization.instrument_monitor import InstrumentMonitor
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt

try:
    from adaptive import SKOptLearner

    WITH_SKOPTLEARNER = True
except ImportError:
    WITH_SKOPTLEARNER = False


# seed the randomization with fixed seed
random.seed(202104)
np.random.seed(202104)


def CosFunc(t, amplitude, frequency, phase):
    """A simple cosine function"""
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


def SinFunc(t, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


# Issue #155: each unit test should be stateless and independent from previous tests
# Parameters are created to emulate a system being measured
t = ManualParameter("t", initial_value=1, unit="s", label="Time")
amp = ManualParameter("amp", initial_value=1, unit="V", label="Amplitude")
freq = ManualParameter("freq", initial_value=1, unit="Hz", label="Frequency")
other_freq = ManualParameter(
    "other_freq", initial_value=1, unit="Hz", label="Other frequency"
)


def sine_model():
    return SinFunc(t(), amplitude=amp(), frequency=freq(), phase=0)


def cosine_model():
    return CosFunc(t(), amplitude=amp(), frequency=freq(), phase=0)


def cosine_model_2():
    return CosFunc(t(), amplitude=amp(), frequency=other_freq(), phase=0)


sig = Parameter(name="sig", label="Signal level", unit="V", get_cmd=cosine_model)

# A signal that uses 4 parameters
sig2 = Parameter(
    name="sig2",
    label="Signal level",
    unit="V",
    get_cmd=lambda: cosine_model() + cosine_model_2(),
)


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


def batched_mock_values(setpoints):
    assert isinstance(setpoints, Iterable)
    return np.sin(setpoints / np.pi)


class DummyBatchedSettable:
    def __init__(self):
        self.name = "DummyBatchedSettable"
        self.label = "Amp"
        self.unit = "V"
        self.batched = True
        # If present must be an integer to comply with JSON schema
        self.batch_size = 0xFFFF
        self.setpoints = []

    def set(self, setpoints):
        self.setpoints = setpoints

    def get(self):
        return self.setpoints


class DummyBatchedGettable:
    # settables are passed for tests purposes only
    def __init__(self, settables, noise=0.0):
        self.name = ["DummyBatchedGettable_0"]
        self.unit = ["W"]
        self.label = ["Watts"]
        self.batched = True
        # If present must be an integer to comply with JSON schema
        self.batch_size = 0xFFFF
        self.settables = [settables] if not isinstance(settables, list) else settables
        self.noise = noise
        self.get_func = batched_mock_values

    def set_return_2D(self):
        self.name.append("DummyBatchedGettable_1")
        self.unit.append("V")
        self.label.append("Amp")

    def prepare(self):
        assert len(self.settables) > 0
        for settable in self.settables:
            assert settable is not None

    def _get_data(self):
        return np.array([spar.get() for spar in self.settables])

    def get(self):
        data = self._get_data()
        data = self.get_func(data)
        noise = self.noise * (np.random.rand(1, data.shape[1]) - 0.5)
        data += noise
        return data[0 : len(self.name), :]


class TestMeasurementControl:
    @classmethod
    def setup_class(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        dh.set_datadir(cls.tmp_dir.name)
        cls.meas_ctrl = MeasurementControl(name="meas_ctrl")
        # ensures the default datadir is used which is excluded from git
        cls.dummy_parabola = DummyParHolder("parabola")

    @classmethod
    def teardown_class(cls):
        cls.meas_ctrl.close()
        cls.dummy_parabola.close()
        dh._datadir = None
        cls.tmp_dir.cleanup()

    def test_MeasurementControl_name(self):
        assert self.meas_ctrl.name == "meas_ctrl"

    def test_repr(self):
        number_points = 5
        xvals = np.linspace(0, 2, number_points)
        yvals = np.linspace(0, 1, number_points * 2)

        self.meas_ctrl.settables(t)
        self.meas_ctrl.setpoints(xvals)
        self.meas_ctrl.gettables(sig)
        repr1 = self.meas_ctrl.__repr__()
        _ = self.meas_ctrl.run()
        repr2 = self.meas_ctrl.__repr__()
        expected = (
            "<MeasurementControl: meas_ctrl>\n"
            "    settables: ['t']\n"
            "    gettables: ['sig']\n"
            f"    setpoints shape: ({len(xvals)}, 1)\n"
        )
        assert repr1 == repr2 == expected

        self.meas_ctrl.setpoints_grid([xvals, yvals])
        repr1 = self.meas_ctrl.__repr__()
        expected = (
            "<MeasurementControl: meas_ctrl>\n"
            "    settables: ['t']\n"
            "    gettables: ['sig']\n"
            f"    setpoints_grid input shapes: [({len(xvals)},), ({len(yvals)},)]\n"
        )
        assert repr1 == expected

        self.meas_ctrl.settables([t, amp])
        _ = self.meas_ctrl.run()
        repr2 = self.meas_ctrl.__repr__()
        expected = (
            "<MeasurementControl: meas_ctrl>\n"
            "    settables: ['t', 'amp']\n"
            "    gettables: ['sig']\n"
            f"    setpoints_grid input shapes: [({len(xvals)},), ({len(yvals)},)]\n"
            f"    setpoints shape: ({len(xvals) * len(yvals)}, 2)\n"
        )
        assert repr2 == expected

    def test_setpoints(self):
        x = np.linspace(0, 10, 11)
        self.meas_ctrl.setpoints(x)
        assert np.array_equal(self.meas_ctrl._setpoints[:, 0], x)

        x = np.random.rand(15, 2)
        self.meas_ctrl.setpoints(x)
        assert np.array_equal(self.meas_ctrl._setpoints, x)

        x = np.random.rand(15, 4)
        self.meas_ctrl.setpoints(x)
        assert np.array_equal(self.meas_ctrl._setpoints, x)

    def test_iterative_1D(self):
        xvals = np.linspace(0, 2 * np.pi, 31)

        self.meas_ctrl.settables(t)
        self.meas_ctrl.setpoints(xvals)
        self.meas_ctrl.gettables(sig)
        dset = self.meas_ctrl.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(t=xvals, amplitude=1, frequency=1, phase=0)

        assert np.array_equal(dset["x0"].values, xvals)
        assert np.array_equal(dset["y0"].values, expected_vals)

        assert isinstance(dset, xr.Dataset)
        assert set(dset.variables.keys()) == {"x0", "y0"}
        assert np.array_equal(dset["x0"], xvals)
        assert dset["x0"].attrs == {
            "name": "t",
            "long_name": "Time",
            "units": "s",
            "batched": False,
        }
        assert dset["y0"].attrs == {
            "name": "sig",
            "long_name": "Signal level",
            "units": "V",
            "batched": False,
        }

    def test_iterative_1D_multi_return(self):
        xvals = np.linspace(0, 2 * np.pi, 31)

        self.meas_ctrl.settables(t)
        self.meas_ctrl.setpoints(xvals)
        self.meas_ctrl.gettables(DualWave())
        dset = self.meas_ctrl.run()

        exp_y0 = SinFunc(xvals, 1, 1, 0)
        exp_y1 = CosFunc(xvals, 1, 1, 0)

        assert set(dset.variables.keys()) == {"x0", "y0", "y1"}
        np.testing.assert_array_equal(dset["y0"], exp_y0)
        np.testing.assert_array_equal(dset["y1"], exp_y1)

    def test_soft_averages_iterative_1D(self):
        def rand():
            return random.uniform(0.0, t())

        rand_get = Parameter(name="sig", label="Signal level", unit="V", get_cmd=rand)
        setpoints = np.arange(100.0)
        self.meas_ctrl.settables(t)
        self.meas_ctrl.setpoints(setpoints)
        self.meas_ctrl.gettables(rand_get)
        r_dset = self.meas_ctrl.run("random")

        avg_dset = self.meas_ctrl.run("averaged", soft_avg=50)

        expected_vals = 0.5 * np.arange(100.0)
        r_delta = abs(r_dset["y0"].values - expected_vals)
        avg_delta = abs(avg_dset["y0"].values - expected_vals)
        assert np.mean(avg_delta) < np.mean(r_delta)

    def test_batched_1D(self):
        x = np.linspace(0, 10, 5)
        device = DummyBatchedSettable()
        self.meas_ctrl.settables(device)
        self.meas_ctrl.setpoints(x)
        # settables are passed for test purposes only, this is not a design pattern!
        self.meas_ctrl.gettables(DummyBatchedGettable(device))
        dset = self.meas_ctrl.run()

        expected_vals = batched_mock_values(x)
        assert np.array_equal(dset["x0"].values, x)
        assert np.array_equal(dset["y0"].values, expected_vals)

        assert isinstance(dset, xr.Dataset)
        assert set(dset.variables.keys()) == {"x0", "y0"}
        assert dset["x0"].attrs == {
            "name": "DummyBatchedSettable",
            "long_name": "Amp",
            "units": "V",
            "batched": True,
            "batch_size": 0xFFFF,
        }
        assert dset["y0"].attrs == {
            "name": "DummyBatchedGettable_0",
            "long_name": "Watts",
            "units": "W",
            "batched": True,
            "batch_size": 0xFFFF,
        }

    def test_batched_batch_size_1D(self):
        x = np.linspace(0, 10, 50)
        device = DummyBatchedSettable()
        self.meas_ctrl.settables(device)
        self.meas_ctrl.setpoints(x)
        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(device)
        # Must be specified otherwise all setpoints will be passed to the settable
        gettable.batch_size = 10
        self.meas_ctrl.gettables(gettable)
        dset = self.meas_ctrl.run()

        expected_vals = batched_mock_values(x)
        assert np.array_equal(dset["x0"].values, x)
        assert np.array_equal(dset["y0"].values, expected_vals)

    def test_soft_averages_batched_1D(self):
        setpoints = np.arange(50.0)
        settable = DummyBatchedSettable()
        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(settable)
        gettable.noise = 0.4
        self.meas_ctrl.settables(settable)
        self.meas_ctrl.setpoints(setpoints)
        self.meas_ctrl.gettables(gettable)
        noisy_dset = self.meas_ctrl.run("noisy")
        xn_0 = noisy_dset["x0"].values
        expected_vals = batched_mock_values(xn_0)
        yn_0 = abs(noisy_dset["y0"].values - expected_vals)

        avg_dset = self.meas_ctrl.run("averaged", soft_avg=1000)
        yavg_0 = abs(avg_dset["y0"].values - expected_vals)

        np.testing.assert_array_equal(xn_0, setpoints)
        assert np.mean(yn_0) > np.mean(yavg_0)
        np.testing.assert_array_almost_equal(yavg_0, np.zeros(len(xn_0)), decimal=2)

    def test_iterative_set_batched_get_1D_raises(self):
        # Mixing iterative and batched settables is allowed as long
        # as at least one settable is batched.

        setpoints = np.linspace(0, 360, 8)
        self.meas_ctrl.settables(t)  # iterative settable
        self.meas_ctrl.setpoints(setpoints)

        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(t)
        self.meas_ctrl.gettables(gettable)  # batched gettable
        with pytest.raises(
            RuntimeError,
            match=r"At least one settable must have `settable.batched=True`",
        ):
            self.meas_ctrl.run("raises")

    def test_iterative_2D_grid(self):

        times = np.linspace(0, 5, 20)
        amps = np.linspace(-1, 1, 5)

        self.meas_ctrl.settables([t, amp])
        self.meas_ctrl.setpoints_grid([times, amps])

        exp_sp = grid_setpoints([times, amps])

        self.meas_ctrl.gettables(sig)
        dset = self.meas_ctrl.run()

        assert np.array_equal(self.meas_ctrl._setpoints, exp_sp)

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(
            t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=1, phase=0
        )

        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])
        assert np.array_equal(dset["y0"].values, expected_vals)

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)
        assert set(dset.variables.keys()) == {"x0", "x1", "y0"}

        assert all(e in dset["x0"].values for e in times)
        assert all(e in dset["x1"].values for e in amps)

        assert dset["x0"].attrs == {
            "name": "t",
            "long_name": "Time",
            "units": "s",
            "batched": False,
        }
        assert dset["x1"].attrs == {
            "name": "amp",
            "long_name": "Amplitude",
            "units": "V",
            "batched": False,
        }
        assert dset["y0"].attrs == {
            "name": "sig",
            "long_name": "Signal level",
            "units": "V",
            "batched": False,
        }

    @pytest.mark.parametrize(
        "lazy_set_parameter,lazy_set_argument,lazy_set_turned_on",
        [
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, False),
            (True, None, True),
            (False, None, False),
        ],
    )
    def test_iterative_2D_grid_with_lazy_set(
        self, lazy_set_parameter, lazy_set_argument, lazy_set_turned_on
    ):
        t.set = Mock(wraps=t.set)
        amp.set = Mock(wraps=amp.set)

        times = np.linspace(0, 5, 20)
        amps = np.linspace(-1, 1, 5)

        self.meas_ctrl.settables([t, amp])
        self.meas_ctrl.setpoints_grid([times, amps])

        exp_sp = grid_setpoints([times, amps])

        self.meas_ctrl.gettables(sig)
        # the lazy_set_argument should override the lazy_set_parameter
        self.meas_ctrl.lazy_set(lazy_set_parameter)
        dset = self.meas_ctrl.run(lazy_set=lazy_set_argument)

        # if lazy_set is turned on, verify that the slow axis (amp) was only called each
        # row (=5 times), but the fast
        # axis (times) each time (=100 times). Otherwise, both should be set each time.
        assert t.set.call_count == 100
        if lazy_set_turned_on:
            assert amp.set.call_count == 5
        else:
            assert amp.set.call_count == 20 * 5

        assert np.array_equal(self.meas_ctrl._setpoints, exp_sp)

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(
            t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=1, phase=0
        )

        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])
        assert np.array_equal(dset["y0"].values, expected_vals)

        # Test properties of the dataset
        assert set(dset.variables.keys()) == {"x0", "x1", "y0"}

        assert all(e in dset["x0"].values for e in times)
        assert all(e in dset["x1"].values for e in amps)

        assert dset["x0"].attrs == {
            "name": "t",
            "long_name": "Time",
            "units": "s",
            "batched": False,
        }
        assert dset["x1"].attrs == {
            "name": "amp",
            "long_name": "Amplitude",
            "units": "V",
            "batched": False,
        }
        assert dset["y0"].attrs == {
            "name": "sig",
            "long_name": "Signal level",
            "units": "V",
            "batched": False,
        }

    def test_iterative_2D_arbitrary(self):

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

        self.meas_ctrl.settables([t, amp])
        self.meas_ctrl.setpoints(setpoints)
        self.meas_ctrl.gettables(sig)
        dset = self.meas_ctrl.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = CosFunc(t=x, amplitude=y, frequency=1, phase=0)

        assert np.array_equal(dset["x0"].values, x)
        assert np.array_equal(dset["x1"].values, y)
        assert np.array_equal(dset["y0"].values, expected_vals)

    def test_batched_2D_grid(self):
        times = np.linspace(10, 20, 3)
        amps = np.linspace(0, 10, 5)

        settables = [DummyBatchedSettable(), DummyBatchedSettable()]
        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(settables)
        self.meas_ctrl.settables(settables)
        self.meas_ctrl.setpoints_grid([times, amps])
        self.meas_ctrl.gettables(gettable)
        dset = self.meas_ctrl.run("2D batched")

        exp_sp = grid_setpoints([times, amps])
        assert np.array_equal(exp_sp, self.meas_ctrl._setpoints)
        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])

        expected_vals = batched_mock_values(dset["x0"].values)
        assert np.array_equal(dset["y0"].values, expected_vals)

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)

        assert dset["x0"].attrs == {
            "name": "DummyBatchedSettable",
            "long_name": "Amp",
            "units": "V",
            "batched": True,
            "batch_size": 0xFFFF,
        }
        assert dset["x1"].attrs == {
            "name": "DummyBatchedSettable",
            "long_name": "Amp",
            "units": "V",
            "batched": True,
            "batch_size": 0xFFFF,
        }
        assert dset["y0"].attrs == {
            "name": "DummyBatchedGettable_0",
            "long_name": "Watts",
            "units": "W",
            "batched": True,
            "batch_size": 0xFFFF,
        }

    def test_iterative_outer_loop_with_inner_batched_2D(self):
        _, _, _ = t(1), amp(1), freq(1)  # Reset globals
        self.meas_ctrl.settables([t, freq])
        times = np.linspace(0, 15, 20)
        freqs = np.linspace(0.1, 1, 10)
        setpoints = [times, freqs]
        self.meas_ctrl.setpoints_grid(setpoints)

        # Using the same gettable for test purposes
        self.meas_ctrl.gettables([sig, sig])

        t.batched = True
        freq.batched = False

        # Must be specified otherwise all setpoints will be passed to the settable
        sig.batch_size = len(times)
        sig.batched = True

        dset = self.meas_ctrl.run("iterative-outer-loop-with-inner-batched-2D")

        expected_vals = CosFunc(
            t=self.meas_ctrl._setpoints[:, 0],
            frequency=self.meas_ctrl._setpoints[:, 1],
            amplitude=1,
            phase=0,
        )
        assert np.array_equal(dset["y0"].values, expected_vals)

        delattr(sig, "batch_size")
        delattr(sig, "batched")
        delattr(t, "batched")
        delattr(freq, "batched")
        _, _, _ = t(1), amp(1), freq(1)  # Reset globals

    def test_batched_2D_grid_multi_return(self):
        times = np.linspace(10, 20, 3)
        amps = np.linspace(0, 10, 5)

        settables = [DummyBatchedSettable(), DummyBatchedSettable()]
        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(settables)
        gettable.set_return_2D()
        self.meas_ctrl.settables(settables)
        self.meas_ctrl.setpoints_grid([times, amps])
        self.meas_ctrl.gettables(gettable)
        dset = self.meas_ctrl.run("2D batched multi return")

        exp_sp = grid_setpoints([times, amps])
        assert np.array_equal(exp_sp, self.meas_ctrl._setpoints)
        assert np.array_equal(dset["x0"].values, exp_sp[:, 0])
        assert np.array_equal(dset["x1"].values, exp_sp[:, 1])

        expected_vals = batched_mock_values(
            np.stack((dset["x0"].values, dset["x1"].values))
        )
        assert np.array_equal(dset["y0"].values, expected_vals[0])
        assert np.array_equal(dset["y1"].values, expected_vals[1])

    def test_batched_2D_grid_multi_return_soft_avg(self):
        x0 = np.arange(5)
        x1 = np.linspace(5, 10, 5)
        settables = [DummyBatchedSettable(), DummyBatchedSettable()]
        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(settables)
        gettable.noise = 0.2
        gettable.set_return_2D()
        self.meas_ctrl.settables(settables)
        self.meas_ctrl.setpoints_grid([x0, x1])
        self.meas_ctrl.gettables(gettable)
        noisy_dset = self.meas_ctrl.run("noisy_batched_grid")

        expected_vals = batched_mock_values(
            np.stack((noisy_dset["x0"].values, noisy_dset["x1"]))
        )
        yn_0 = abs(noisy_dset["y0"].values - expected_vals[0])
        yn_1 = abs(noisy_dset["y1"].values - expected_vals[1])

        avg_dset = self.meas_ctrl.run("avg_batched_grid", soft_avg=1000)
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

    def test_batched_2D_arbitrary(self):
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

        settables = [DummyBatchedSettable(), DummyBatchedSettable()]
        self.meas_ctrl.settables(settables)
        self.meas_ctrl.setpoints(setpoints)
        # settables are passed for test purposes only, this is not a design pattern!
        self.meas_ctrl.gettables(DummyBatchedGettable(settables))
        dset = self.meas_ctrl.run()

        assert TUID.is_valid(dset.attrs["tuid"])

        expected_vals = batched_mock_values(x)

        assert np.array_equal(dset["x0"].values, x)
        assert np.array_equal(dset["x1"].values, y)
        assert np.array_equal(dset["y0"].values, expected_vals)

    def test_variable_return_batched(self):
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
        settable = DummyBatchedSettable()
        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(settable)
        gettable.get_func = v_size
        self.meas_ctrl.settables(settable)
        self.meas_ctrl.setpoints(setpoints)
        self.meas_ctrl.gettables(gettable)
        dset = self.meas_ctrl.run("varying")

        assert np.array_equal(dset["x0"], setpoints)
        assert np.array_equal(dset["y0"], 2 * setpoints)

    def test_variable_return_batched_soft_avg(self):
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
        settable = DummyBatchedSettable()
        # settables are passed for test purposes only, this is not a design pattern!
        gettable = DummyBatchedGettable(settable)
        gettable.get_func = v_size
        gettable.noise = 0.25
        self.meas_ctrl.settables(settable)
        self.meas_ctrl.setpoints(setpoints)
        self.meas_ctrl.gettables(gettable)
        dset = self.meas_ctrl.run("varying", soft_avg=1000)

        assert np.array_equal(dset["x0"], setpoints)
        np.testing.assert_array_almost_equal(dset.y0, 2 * setpoints, decimal=2)

    def test_iterative_3D_grid(self):
        times = np.linspace(0, 5, 2)
        amps = np.linspace(-1, 1, 3)
        freqs = np.linspace(41000, 82000, 2)

        self.meas_ctrl.settables([t, amp, freq])
        self.meas_ctrl.setpoints_grid([times, amps, freqs])

        exp_sp = grid_setpoints([times, amps, freqs])

        self.meas_ctrl.gettables(sig)
        dset = self.meas_ctrl.run()

        assert np.array_equal(self.meas_ctrl._setpoints, exp_sp)

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
        assert set(dset.variables.keys()) == {"x0", "x1", "x2", "y0"}
        assert all(e in dset["x0"] for e in times)
        assert all(e in dset["x1"] for e in amps)
        assert all(e in dset["x2"] for e in freqs)

        assert dset["x0"].attrs == {
            "name": "t",
            "long_name": "Time",
            "units": "s",
            "batched": False,
        }
        assert dset["x2"].attrs == {
            "name": "freq",
            "long_name": "Frequency",
            "units": "Hz",
            "batched": False,
        }
        assert dset["y0"].attrs == {
            "name": "sig",
            "long_name": "Signal level",
            "units": "V",
            "batched": False,
        }

    def test_iterative_3D_multi_return(self):
        times = np.linspace(0, 5, 2)
        amps = np.linspace(-1, 1, 3)
        freqs = np.linspace(41000, 82000, 2)

        self.meas_ctrl.settables([t, amp, freq])
        self.meas_ctrl.setpoints_grid([times, amps, freqs])
        self.meas_ctrl.gettables([DualWave(), sig, DualWave()])
        dset = self.meas_ctrl.run()

        exp_sp = grid_setpoints([times, amps, freqs])
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

    def test_batched_3D_multi_return_soft_averaging(self):
        _, _, _ = t(1), amp(1), freq(1)  # Reset globals

        def v_get(setpoints):
            return setpoints

        times = np.linspace(0, 5, 2)
        amps = np.linspace(-1, 1, 3)
        freqs = np.linspace(41000, 82000, 2)
        batched_settable_t = DummyBatchedSettable()
        batched_settable_0 = DummyBatchedSettable()
        batched_settable_1 = DummyBatchedSettable()
        # settables are passed for test purposes only, this is not a design pattern!
        nd_gettable = DummyBatchedGettable([batched_settable_0, batched_settable_1])
        nd_gettable.set_return_2D()
        # settables are passed for test purposes only, this is not a design pattern!
        noisy_gettable = DummyBatchedGettable([batched_settable_t])
        noisy_gettable.get_func = v_get
        noisy_gettable.noise = 0.25

        self.meas_ctrl.settables(
            [batched_settable_t, batched_settable_0, batched_settable_1]
        )
        self.meas_ctrl.setpoints_grid([times, amps, freqs])
        self.meas_ctrl.gettables([noisy_gettable, nd_gettable])
        dset = self.meas_ctrl.run(soft_avg=1000)

        exp_sp = grid_setpoints([times, amps, freqs])
        np.testing.assert_array_almost_equal(dset.y0, exp_sp[:, 0], decimal=2)
        np.testing.assert_array_almost_equal(
            dset.y1, batched_mock_values(exp_sp[:, 1]), decimal=4
        )
        np.testing.assert_array_almost_equal(
            dset.y2, batched_mock_values(exp_sp[:, 2]), decimal=4
        )
        _, _, _ = t(1), amp(1), freq(1)  # Reset globals

    def test_batched_attr_raises(self):
        times = np.linspace(0, 5, 3)
        freqs = np.linspace(41000, 82000, 8)

        freq.batched = True
        freq.batch_size = 8
        t.batched = False

        self.meas_ctrl.settables([freq, t])
        self.meas_ctrl.setpoints_grid([freqs, times])
        # Ensure forgetting this raises exception
        # sig.batched = True
        self.meas_ctrl.gettables(sig)

        with pytest.raises(RuntimeError):
            self.meas_ctrl.run()

        # reset for other tests
        delattr(freq, "batched")
        delattr(freq, "batch_size")
        delattr(t, "batched")

    def test_batched_grid_mixed(self):
        _, _, _, _ = t(1), amp(1), freq(1), other_freq(1)  # Reset globals
        times = np.linspace(0, 5, 3)
        amps = np.linspace(-1, 1, 4)
        freqs = np.linspace(41000, 82000, 8)
        other_freqs = np.linspace(46000, 88000, 8)

        freq.batched = True
        freq.batch_size = 5  # odd size for extra test
        t.batched = False
        other_freq.batched = True
        other_freq.batch_size = 3  # odd size and different value for extra test
        amp.batched = False

        sig2.batched = True

        settables = [freq, t, other_freq, amp]
        self.meas_ctrl.settables(settables)
        setpoints = [freqs, times, other_freqs, amps]
        self.meas_ctrl.setpoints_grid(setpoints)
        self.meas_ctrl.gettables(sig2)
        dset = self.meas_ctrl.run("bla", soft_avg=2)

        assert isinstance(freq(), Iterable)
        assert not isinstance(t(), Iterable)
        assert isinstance(other_freq(), Iterable)
        assert not isinstance(amp(), Iterable)

        exp_sp = grid_setpoints(setpoints, settables=settables)
        assert np.array_equal(self.meas_ctrl._setpoints, exp_sp)

        exp_sp = exp_sp.T
        _, _, _, _ = (
            freq(exp_sp[0]),
            t(exp_sp[1]),
            other_freq(exp_sp[2]),
            amp(exp_sp[3]),
        )
        assert np.array_equal(dset["y0"].values, sig2())

        # Reset for other tests
        delattr(freq, "batched")
        delattr(freq, "batch_size")
        delattr(t, "batched")
        delattr(other_freq, "batched")
        delattr(other_freq, "batch_size")
        delattr(amp, "batched")
        delattr(sig2, "batched")
        _, _, _, _ = t(1), amp(1), freq(1), other_freq(1)  # Reset globals

    def test_adaptive_nelder_mead(self):
        self.meas_ctrl.settables([self.dummy_parabola.x, self.dummy_parabola.y])
        af_pars = {
            "adaptive_function": optimize.minimize,
            "x0": [-50, -50],
            "method": "Nelder-Mead",
        }
        self.dummy_parabola.noise(0.5)
        self.meas_ctrl.gettables(self.dummy_parabola.parabola)
        dset = self.meas_ctrl.run_adaptive("nelder_mead", af_pars)

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

        self.meas_ctrl.settables(freq)
        af_pars = {
            "adaptive_function": adaptive.learner.Learner1D,
            "goal": lambda l: l.npoints > 5,
            "bounds": (6e9, 7e9),
        }
        self.meas_ctrl.gettables([ResonancePlus(), dummy])
        dset = self.meas_ctrl.run_adaptive("multi return", af_pars)
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

        self.meas_ctrl.settables(freq)
        af_pars = {
            "adaptive_function": adaptive.learner.Learner1D,
            "goal": lambda l: l.npoints > 20 * 20,
            "bounds": (6e9, 7e9),
        }
        self.meas_ctrl.gettables(resonance)
        _ = self.meas_ctrl.run_adaptive("adaptive sample", af_pars)

    def test_adaptive_sampling(self):
        self.dummy_parabola.noise(0)
        self.meas_ctrl.settables([self.dummy_parabola.x, self.dummy_parabola.y])
        af_pars = {
            "adaptive_function": adaptive.learner.Learner2D,
            "goal": lambda l: l.npoints > 20 * 20,
            "bounds": ((-50, 50), (-20, 30)),
        }
        self.meas_ctrl.gettables(self.dummy_parabola.parabola)
        _ = self.meas_ctrl.run_adaptive("adaptive sample", af_pars)
        # todo pycqed has no verification step here, what should we do?

    @pytest.mark.parametrize(
        "lazy_set_parameter,lazy_set_argument,lazy_set_turned_on",
        [
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, False),
            (True, None, True),
            (False, None, False),
        ],
    )
    def test_adaptive_sampling_with_lazy_set(
        self, lazy_set_parameter, lazy_set_argument, lazy_set_turned_on
    ):
        self.dummy_parabola.x.set = Mock(wraps=self.dummy_parabola.x.set)
        self.dummy_parabola.y.set = Mock(wraps=self.dummy_parabola.y.set)

        self.dummy_parabola.noise(0)
        self.meas_ctrl.settables([self.dummy_parabola.x, self.dummy_parabola.y])

        num_x = 5
        num_y = 20

        def _adaptive_function(meas_func, **kwargs):
            """A dimple adaptive function that executes a 2D sweep and conforms with
            the accepted interface for an adaptive function."""
            points = grid_setpoints(
                [np.linspace(-50, 50, num_x), np.linspace(-20, 30, num_y)]
            )

            for setpoint in points:
                _ = meas_func(setpoint)

            # The meas_ctrl takes care of intercepting the measured values so no return
            # is specified here

        af_pars = {"adaptive_function": _adaptive_function}
        self.meas_ctrl.gettables(self.dummy_parabola.parabola)
        self.meas_ctrl.lazy_set(lazy_set_parameter)
        _ = self.meas_ctrl.run_adaptive(
            "adaptive sample", af_pars, lazy_set=lazy_set_argument
        )

        # if lazy_set is turned on, verify that the adaptive algorithm did not set the
        # parameters each time
        if lazy_set_turned_on:
            assert self.dummy_parabola.x.set.call_count == num_x * num_y
            assert self.dummy_parabola.y.set.call_count == num_y
        else:
            assert self.dummy_parabola.x.set.call_count == num_x * num_y
            assert self.dummy_parabola.y.set.call_count == num_x * num_y

    @pytest.mark.skipif(
        not WITH_SKOPTLEARNER, reason="scikit-optimize is not installed"
    )
    def test_adaptive_skoptlearner(self):
        self.dummy_parabola.noise(0)
        self.meas_ctrl.settables([self.dummy_parabola.x, self.dummy_parabola.y])
        af_pars = {
            "adaptive_function": SKOptLearner,
            "goal": lambda l: l.npoints > 15,
            "dimensions": [(-50.0, +50.0), (-20.0, +30.0)],
            "base_estimator": "gp",
            "acq_func": "EI",
            "acq_optimizer": "lbfgs",
        }
        self.meas_ctrl.gettables(self.dummy_parabola.parabola)
        _ = self.meas_ctrl.run_adaptive("skopt", af_pars)
        # todo pycqed has no verification step here, what should we do?

    def test_progress_callback(self):
        progress_param = ManualParameter("progress", initial_value=0)

        def set_progress_param_callable(progress):
            progress_param(progress)

        self.meas_ctrl.on_progress_callback(set_progress_param_callable)
        assert progress_param() == 0

        xvals = np.linspace(0, 2 * np.pi, 31)
        self.meas_ctrl.settables(t)
        self.meas_ctrl.setpoints(xvals)
        self.meas_ctrl.gettables(sig)
        self.meas_ctrl.run()

        assert progress_param() == 100

    def test_meas_ctrl_plotmon_integration(self):
        plotmon = PlotMonitor_pyqt("plotmon_meas_ctrl")

        self.meas_ctrl.instr_plotmon(plotmon.name)

        assert len(plotmon.tuids()) == 0

        times = np.linspace(0, 5, 18)
        amps = np.linspace(-1, 1, 5)

        self.meas_ctrl.settables([t, amp])
        self.meas_ctrl.setpoints_grid([times, amps])
        self.meas_ctrl.gettables(sig)
        self.meas_ctrl.run("2D Cosine test")

        assert len(plotmon.tuids()) > 0

        plotmon.close()
        self.meas_ctrl.instr_plotmon("")

    def test_meas_ctrl_insmon_integration(self):
        inst_mon = InstrumentMonitor("insmon_meas_ctrl")
        self.meas_ctrl.instrument_monitor(inst_mon.name)
        assert self.meas_ctrl.instrument_monitor.get_instr().widget.getNodes()
        inst_mon.close()
        self.meas_ctrl.instrument_monitor("")

    def test_instrument_settings_from_disk(self, tmp_test_data_dir):
        load_settings_onto_instrument(
            self.dummy_parabola, TUID("20200814-134652-492-fbf254"), tmp_test_data_dir
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
                non_existing, TUID("20200814-134652-492-fbf254"), tmp_test_data_dir
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

        self.meas_ctrl.settables(badSetter("badSetter"))
        self.meas_ctrl.setpoints(np.linspace(0, 1, 10))
        self.meas_ctrl.gettables(freq)
        with pytest.raises(
            AttributeError,
            match="'badSetter' object has no attribute 'non_existing_param'",
        ):
            self.meas_ctrl.run("This raises exception as expected")

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

        self.meas_ctrl.settables(t)
        self.meas_ctrl.setpoints(np.linspace(0, 1, 10) * 1e-9)
        self.meas_ctrl.gettables(badGetter("badGetter"))
        with pytest.raises(
            AttributeError,
            match="'badGetter' object has no attribute 'non_existing_param'",
        ):
            self.meas_ctrl.run("This raises exception as expected")

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Emulating this test on windows is too complicated",
    )
    def test_keyboard_interrupt(self):
        class GettableUserInterrupt:
            def __init__(self):
                self.name = "amp"
                self.label = "Amplitude"
                self.unit = "V"
                self._num_interrupts = 1

            def get(self):
                if time_par() == 3:
                    for _ in range(self._num_interrupts):
                        # This same signal is sent when pressing `ctrl` + `c`
                        # or the "Stop kernel" button is pressed in a Jupyter(Lab) notebook
                        # send a stop signal directly through the os interface
                        os.kill(os.getpid(), signal.SIGINT)
                return time_par()

        time_par = ManualParameter(name="time", unit="s", label="Measurement Time")
        time_par.batched = False

        gettable = GettableUserInterrupt()

        self.meas_ctrl.settables(time_par)
        self.meas_ctrl.setpoints(np.arange(10))
        self.meas_ctrl.gettables(gettable)
        with pytest.raises(KeyboardInterrupt):
            self.meas_ctrl.run("interrupt_test")

        dset = self.meas_ctrl._dataset
        # we stop after 4th iteration
        assert sum(np.isnan(dset.y0) ^ 1) == 4

        # Ensure force stop still possible
        gettable._num_interrupts = 5
        self.meas_ctrl.settables(time_par)
        self.meas_ctrl.setpoints(np.arange(10))
        self.meas_ctrl.gettables(gettable)

        with pytest.raises(KeyboardInterrupt):
            self.meas_ctrl.run("interrupt_test_force")

        dset = self.meas_ctrl._dataset
        # we stop right away
        assert sum(np.isnan(dset.y0) ^ 1) == 3


def test_repr_new_and_closed():

    meas_ctrl = MeasurementControl("meas_ctrl")
    assert meas_ctrl.__repr__()
    meas_ctrl.close()
    assert meas_ctrl.__repr__()
    meas_ctrl.close()


class TestGridSetpoints:
    @classmethod
    def setup_class(self):
        self.sp_i0 = np.array([0, 1, 2, 3])
        self.sp_i1 = np.array([4, 5])
        self.sp_i2 = np.array([-1, -2])

        base_batched = [1, 2, 6]
        self.sp_b0 = np.array(base_batched)
        self.sp_b1 = np.array(base_batched) * 2
        self.sp_b2 = np.array(base_batched) * 3

        self.iterative = [self.sp_i0, self.sp_i1, self.sp_i2]
        self.batched = [self.sp_b0, self.sp_b1, self.sp_b2]

    @classmethod
    def mk_settables(self, batched_mask):
        settables = []
        for i, m in enumerate(batched_mask):
            par = ManualParameter(name=f"x{i}", label=f"X{i}", unit="s")
            par.batched = m
            settables.append(par)
        return settables

    def test_grid_setpoints_mixed_simple(self):

        # Most simple case
        batched_mask = [True, False]
        settables = self.mk_settables(batched_mask)

        it_i = iter(self.iterative)
        it_b = iter(self.batched)
        setpoints = [(next(it_b) if m else next(it_i)) for m in batched_mask]
        gridded = grid_setpoints(setpoints, settables=settables)
        expected = np.column_stack(
            [
                np.tile(self.sp_b0, len(self.sp_i0)),
                np.repeat(self.sp_i0, len(self.sp_b0)),
            ]
        )
        np.testing.assert_array_equal(gridded, expected)

    def test_grid_setpoints_mixed_simple_reversed(self):

        # Most simple case reversed order
        batched_mask = [True, False]
        settables = self.mk_settables(batched_mask)
        it_i = iter(self.iterative)
        it_b = iter(self.batched)
        setpoints = [(next(it_b) if m else next(it_i)) for m in batched_mask]
        gridded = grid_setpoints(setpoints, settables=settables)
        expected = np.column_stack(
            [
                np.tile(self.sp_b0, len(self.sp_i0)),
                np.repeat(self.sp_i0, len(self.sp_b0)),
            ]
        )
        np.testing.assert_array_equal(gridded, expected)

    def test_grid_setpoints_mixed_two_batched(self):

        # Several batched settables
        batched_mask = [False, True, True]
        settables = self.mk_settables(batched_mask)
        it_i = iter(self.iterative)
        it_b = iter(self.batched)
        setpoints = [(next(it_b) if m else next(it_i)) for m in batched_mask]
        gridded = grid_setpoints(setpoints, settables=settables)
        expected = np.column_stack(
            [
                np.repeat(self.sp_i0, len(self.sp_b0) * len(self.sp_b0)),
                np.tile(self.sp_b0, len(self.sp_i0) * len(self.sp_b1)),
                np.tile(np.repeat(self.sp_b1, len(self.sp_b0)), len(self.sp_i0)),
            ]
        )
        np.testing.assert_array_equal(gridded, expected)

    def test_grid_setpoints_mixed_two_batched_two_iterative(self):

        # Several batched settables and several iterative settables
        batched_mask = [False, True, False, True]
        settables = self.mk_settables(batched_mask)
        settables[-1].batch_size = 2
        it_i = iter(self.iterative)
        it_b = iter(self.batched)
        setpoints = [(next(it_b) if m else next(it_i)) for m in batched_mask]
        gridded = grid_setpoints(setpoints, settables=settables)
        expected = np.column_stack(
            [
                np.tile(
                    np.repeat(self.sp_i0, len(self.sp_b1) * len(self.sp_b0)),
                    len(self.sp_i1),
                ),
                np.tile(
                    np.repeat(self.sp_b0, len(self.sp_b1)),
                    len(self.sp_i0) * len(self.sp_i1),
                ),
                np.repeat(
                    self.sp_i1, len(self.sp_b0) * len(self.sp_b1) * len(self.sp_i0)
                ),
                np.tile(
                    self.sp_b1, len(self.sp_i0) * len(self.sp_i1) * len(self.sp_b0)
                ),
            ]
        )
        np.testing.assert_array_equal(gridded, expected)


def test_grid_setpoints():
    x = np.arange(5)
    y = np.linspace(-1, 1, 3)

    sp = grid_setpoints([x, y])
    assert sp[:, 0].all() == np.tile(np.arange(5), 3).all()
    assert sp[:, 1].all() == np.repeat(y, 5).all()

    z = np.linspace(100, 200, 2)
    sp = grid_setpoints([x, y, z])
    assert all(e in sp[:, 0] for e in x)
    assert all(e in sp[:, 1] for e in y)
    assert all(e in sp[:, 2] for e in z)
