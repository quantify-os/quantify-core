# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
# pylint: disable=invalid-name
import random
import signal
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import Mock

import adaptive
import numpy as np
import pytest
import xarray as xr
from qcodes.instrument import ChannelTuple, Instrument
from qcodes.parameters import ManualParameter, Parameter

try:
    from qcodes.instrument_drivers.mock_instruments import DummyChannel, DummyInstrument
except ImportError:
    from qcodes.tests.instrument_mocks import (
        DummyChannel,
        DummyInstrument,
    )
from qcodes.utils import validators as vals
from scipy import optimize
from xarray import Dataset

import quantify_core.data.handling as dh
from quantify_core import __version__ as _quantify_version
from quantify_core.data.types import TUID
from quantify_core.measurement.control import MeasurementControl, grid_setpoints
from quantify_core.utilities.experiment_helpers import load_settings_onto_instrument
from quantify_core.utilities.general import without
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


def cosine_function(t, amplitude, frequency, phase):
    """A simple cosine function"""
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


def sine_function(t, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


@pytest.fixture(scope="function")
def parameters():
    t = ManualParameter("t", initial_value=1, unit="s", label="Time")
    amp = ManualParameter("amp", initial_value=1, unit="V", label="Amplitude")
    freq = ManualParameter("freq", initial_value=1, unit="Hz", label="Frequency")
    other_freq = ManualParameter(
        "other_freq", initial_value=1, unit="Hz", label="Other frequency"
    )

    def sine_model():
        return sine_function(t(), amplitude=amp(), frequency=freq(), phase=0)

    def cosine_model():
        return cosine_function(t(), amplitude=amp(), frequency=freq(), phase=0)

    def cosine_model_2():
        return cosine_function(t(), amplitude=amp(), frequency=other_freq(), phase=0)

    class DualWave:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self.name = ["sin", "cos"]
            self.unit = ["V", "V"]
            self.label = ["Sine", "Cosine"]

        def get(self):
            return np.array([sine_model(), cosine_model()])

    sig = Parameter(name="sig", label="Signal level", unit="V", get_cmd=cosine_model)

    # A signal that uses 4 parameters
    sig2 = Parameter(
        name="sig2",
        label="Signal level",
        unit="V",
        get_cmd=lambda: cosine_model() + cosine_model_2(),
    )

    return SimpleNamespace(
        t=t,
        amp=amp,
        freq=freq,
        other_freq=other_freq,
        sig=sig,
        sig2=sig2,
        DualWave=DualWave,
    )


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


@pytest.fixture(scope="module")
def dummy_parabola():
    yield (instrument := DummyParHolder("parabola"))
    instrument.close()


def batched_mock_values(setpoints):
    assert isinstance(setpoints, np.ndarray)
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


class DummyBatchedGettable:  # pylint: disable=too-many-instance-attributes
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

    def set_return_2d(self):
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


@pytest.fixture(scope="module")
def meas_ctrl(tmp_path_factory):
    prev_datadir = dh._datadir  # type: ignore
    tmpdir = tmp_path_factory.mktemp("mc_datadir")
    dh.set_datadir(str(tmpdir))
    meas_ctrl = MeasurementControl(name="meas_ctrl")
    yield meas_ctrl
    meas_ctrl.close()
    dh._datadir = prev_datadir  # type: ignore


@pytest.fixture(scope="function")
def meas_ctrl_empty(tmp_path_factory):
    prev_datadir = dh._datadir  # type: ignore
    tmpdir = tmp_path_factory.mktemp("mc_datadir_empty")
    dh.set_datadir(str(tmpdir))
    meas_ctrl = MeasurementControl(name="meas_ctrl_empty")
    yield meas_ctrl
    meas_ctrl.close()
    dh._datadir = prev_datadir  # type: ignore


def test_name(meas_ctrl):
    assert meas_ctrl.name == "meas_ctrl"


def test_save_data(meas_ctrl, parameters):
    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(np.array([0, 1, 2, 3]))
    meas_ctrl.gettables(parameters.sig)

    dataset = meas_ctrl.run(save_data=False)
    with pytest.raises(FileNotFoundError):
        _ = dh.locate_experiment_container(dataset.tuid)

    dataset = meas_ctrl.run(save_data=True)
    _ = dh.locate_experiment_container(dataset.tuid)
    _ = dh.load_dataset(dataset.tuid)
    _ = dh.load_snapshot(dataset.tuid)


def test_repr(meas_ctrl, parameters):
    number_points = 5
    xvals = np.linspace(0, 2, number_points)
    yvals = np.linspace(0, 1, number_points * 2)

    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(xvals)
    meas_ctrl.gettables(parameters.sig)
    repr1 = repr(meas_ctrl)
    _ = meas_ctrl.run()
    repr2 = repr(meas_ctrl)
    expected = (
        "<MeasurementControl: meas_ctrl>\n"
        "    settables: ['t']\n"
        "    gettables: ['sig']\n"
        f"    setpoints shape: ({len(xvals)}, 1)\n"
    )
    assert repr1 == repr2 == expected

    meas_ctrl.setpoints_grid([xvals, yvals])
    repr1 = repr(meas_ctrl)
    expected = (
        "<MeasurementControl: meas_ctrl>\n"
        "    settables: ['t']\n"
        "    gettables: ['sig']\n"
        f"    setpoints_grid input shapes: [({len(xvals)},), ({len(yvals)},)]\n"
    )
    assert repr1 == expected

    meas_ctrl.settables([parameters.t, parameters.amp])
    _ = meas_ctrl.run()
    repr2 = repr(meas_ctrl)
    expected = (
        "<MeasurementControl: meas_ctrl>\n"
        "    settables: ['t', 'amp']\n"
        "    gettables: ['sig']\n"
        f"    setpoints_grid input shapes: [({len(xvals)},), ({len(yvals)},)]\n"
        f"    setpoints shape: ({len(xvals) * len(yvals)}, 2)\n"
    )
    assert repr2 == expected


def test_setpoints(meas_ctrl):
    x = np.linspace(0, 10, 11)
    meas_ctrl.setpoints(x)
    assert np.array_equal(meas_ctrl._setpoints[0], x)

    x = np.random.rand(15, 2)
    meas_ctrl.setpoints(x)
    assert np.array_equal(meas_ctrl._setpoints, x.T)

    x = np.random.rand(15, 4)
    meas_ctrl.setpoints(x)
    assert np.array_equal(meas_ctrl._setpoints, x.T)


def test_iterative_1d(meas_ctrl, parameters):
    xvals = np.linspace(0, 2 * np.pi, 31)

    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(xvals)
    meas_ctrl.gettables(parameters.sig)
    dset = meas_ctrl.run()

    assert TUID.is_valid(dset.attrs["tuid"])

    expected_vals = cosine_function(t=xvals, amplitude=1, frequency=1, phase=0)

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


def test_iterative_1d_multi_return(meas_ctrl, parameters):
    xvals = np.linspace(0, 2 * np.pi, 31)

    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(xvals)
    meas_ctrl.gettables(parameters.DualWave())
    dset = meas_ctrl.run()

    exp_y0 = sine_function(xvals, 1, 1, 0)
    exp_y1 = cosine_function(xvals, 1, 1, 0)

    assert set(dset.variables.keys()) == {"x0", "y0", "y1"}
    np.testing.assert_array_equal(dset["y0"], exp_y0)
    np.testing.assert_array_equal(dset["y1"], exp_y1)


def test_soft_averages_iterative_1d(meas_ctrl, parameters):
    def rand():
        return random.uniform(0.0, parameters.t())

    rand_get = Parameter(name="sig", label="Signal level", unit="V", get_cmd=rand)
    setpoints = np.arange(100.0)
    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(setpoints)
    meas_ctrl.gettables(rand_get)
    r_dset = meas_ctrl.run("random")

    avg_dset = meas_ctrl.run("averaged", soft_avg=50)

    expected_vals = 0.5 * np.arange(100.0)
    r_delta = abs(r_dset["y0"].values - expected_vals)
    avg_delta = abs(avg_dset["y0"].values - expected_vals)
    assert np.mean(avg_delta) < np.mean(r_delta)


def test_batched_1d(meas_ctrl):
    x = np.linspace(0, 10, 5)
    device = DummyBatchedSettable()
    meas_ctrl.settables(device)
    meas_ctrl.setpoints(x)
    # settables are passed for test purposes only, this is not a design pattern!
    meas_ctrl.gettables(DummyBatchedGettable(device))
    dset = meas_ctrl.run()

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


def test_batched_batch_size_1d(meas_ctrl):
    x = np.linspace(0, 10, 50)
    device = DummyBatchedSettable()
    meas_ctrl.settables(device)
    meas_ctrl.setpoints(x)
    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(device)
    # Must be specified otherwise all setpoints will be passed to the settable
    gettable.batch_size = 10
    meas_ctrl.gettables(gettable)
    dset = meas_ctrl.run(save_data=False)

    expected_vals = batched_mock_values(x)
    assert np.array_equal(dset["x0"].values, x)
    assert np.array_equal(dset["y0"].values, expected_vals)


def test_measurement_description(meas_ctrl, parameters):
    number_points = 5
    xvals = np.linspace(0, 2, number_points)
    yvals = np.linspace(0, 1, number_points * 2)

    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(xvals)
    meas_ctrl.gettables(parameters.sig)
    _ = meas_ctrl.run()
    descr = meas_ctrl.measurement_description()
    expected = {
        "name": "",
        "settables": ["t"],
        "gettables": ["sig"],
        "setpoints_shape": (number_points, 1),
        "soft_avg": 1,
    }
    assert without(descr, ["acquired_dataset"]) == expected

    meas_ctrl.setpoints_grid([xvals, yvals])

    meas_ctrl.settables([parameters.t, parameters.amp])
    _ = meas_ctrl.run()
    descr = meas_ctrl.measurement_description()
    expected = {
        "name": "",
        "settables": ["t", "amp"],
        "gettables": ["sig"],
        "setpoints_shape": (len(xvals) * len(yvals), 2),
        "soft_avg": 1,
    }
    assert without(descr, ["acquired_dataset"]) == expected


def test_soft_averages_batched_1d(meas_ctrl):
    setpoints = np.arange(50.0)
    settable = DummyBatchedSettable()
    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(settable)
    gettable.noise = 0.4
    meas_ctrl.settables(settable)
    meas_ctrl.setpoints(setpoints)
    meas_ctrl.gettables(gettable)
    noisy_dset = meas_ctrl.run("noisy", save_data=False)
    xn_0 = noisy_dset["x0"].values
    expected_vals = batched_mock_values(xn_0)
    yn_0 = abs(noisy_dset["y0"].values - expected_vals)

    avg_dset = meas_ctrl.run("averaged", soft_avg=1000)
    yavg_0 = abs(avg_dset["y0"].values - expected_vals)

    np.testing.assert_array_equal(xn_0, setpoints)
    assert np.mean(yn_0) > np.mean(yavg_0)
    np.testing.assert_array_almost_equal(yavg_0, np.zeros(len(xn_0)), decimal=2)


def test_iterative_set_batched_get_1d_raises(meas_ctrl, parameters):
    # Mixing iterative and batched settables is allowed as long
    # as at least one settable is batched.

    setpoints = np.linspace(0, 360, 8)
    meas_ctrl.settables(parameters.t)  # iterative settable
    meas_ctrl.setpoints(setpoints)

    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(parameters.t)
    meas_ctrl.gettables(gettable)  # batched gettable
    with pytest.raises(
        RuntimeError,
        match=r"At least one settable must have `settable.batched=True`",
    ):
        meas_ctrl.run("raises")


def test_iterative_2d_grid(meas_ctrl, parameters):
    times = np.linspace(0, 5, 20)
    amps = np.linspace(-1, 1, 5)

    meas_ctrl.settables([parameters.t, parameters.amp])
    meas_ctrl.setpoints_grid([times, amps])

    exp_sp = grid_setpoints([times, amps])

    meas_ctrl.gettables(parameters.sig)
    dset = meas_ctrl.run()

    assert np.array_equal(meas_ctrl._setpoints, exp_sp)

    assert TUID.is_valid(dset.attrs["tuid"])

    expected_vals = cosine_function(
        t=exp_sp[0], amplitude=exp_sp[1], frequency=1, phase=0
    )

    assert np.array_equal(dset["x0"].values, exp_sp[0])
    assert np.array_equal(dset["x1"].values, exp_sp[1])
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
def test_iterative_2d_grid_with_lazy_set(
    meas_ctrl, lazy_set_parameter, lazy_set_argument, lazy_set_turned_on, parameters
):
    parameters.t.set = Mock(wraps=parameters.t.set)
    parameters.amp.set = Mock(wraps=parameters.amp.set)

    times = np.linspace(0, 5, 20)
    amps = np.linspace(-1, 1, 5)

    meas_ctrl.settables([parameters.t, parameters.amp])
    meas_ctrl.setpoints_grid([times, amps])

    exp_sp = grid_setpoints([times, amps])

    meas_ctrl.gettables(parameters.sig)
    # the lazy_set_argument should override the lazy_set_parameter
    meas_ctrl.lazy_set(lazy_set_parameter)
    dset = meas_ctrl.run(lazy_set=lazy_set_argument)

    # if lazy_set is turned on, verify that the slow axis (amp) was only called each
    # row (=5 times), but the fast
    # axis (times) each time (=100 times). Otherwise, both should be set each time.
    assert parameters.t.set.call_count == 100
    if lazy_set_turned_on:
        assert parameters.amp.set.call_count == 5
    else:
        assert parameters.amp.set.call_count == 20 * 5

    assert np.array_equal(meas_ctrl._setpoints, exp_sp)

    assert TUID.is_valid(dset.attrs["tuid"])

    expected_vals = cosine_function(
        t=exp_sp[0], amplitude=exp_sp[1], frequency=1, phase=0
    )

    assert np.array_equal(dset["x0"].values, exp_sp[0])
    assert np.array_equal(dset["x1"].values, exp_sp[1])
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


def test_iterative_2d_arbitrary(meas_ctrl, parameters):
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

    meas_ctrl.settables([parameters.t, parameters.amp])
    meas_ctrl.setpoints(setpoints)
    meas_ctrl.gettables(parameters.sig)
    dset = meas_ctrl.run()

    assert TUID.is_valid(dset.attrs["tuid"])

    expected_vals = cosine_function(t=x, amplitude=y, frequency=1, phase=0)

    assert np.array_equal(dset["x0"].values, x)
    assert np.array_equal(dset["x1"].values, y)
    assert np.array_equal(dset["y0"].values, expected_vals)


@pytest.mark.parametrize(
    "setpoints_x,setpoints_y,is_uniformly_spaced",
    [
        (np.linspace(0, 10, 101), np.linspace(10, 20, 101), True),
        (np.linspace(0, 10, 101) ** 2, np.linspace(10, 20, 101), False),
        (np.linspace(0, 10, 101) ** 2, np.linspace(10, 20, 101) ** 2, False),
    ],
)
def test_iterative_1d_with_two_settables_signals_plotmon_of_spacing(
    meas_ctrl, setpoints_x, setpoints_y, is_uniformly_spaced, parameters
):
    setpoints = np.column_stack((setpoints_x, setpoints_y))

    meas_ctrl.settables([parameters.freq, parameters.other_freq])
    meas_ctrl.setpoints(setpoints)
    meas_ctrl.gettables(parameters.sig)
    dset = meas_ctrl.run()

    assert dset.attrs["1d_2_settables_uniformly_spaced"] == is_uniformly_spaced


def test_batched_2d_grid(meas_ctrl):
    times = np.linspace(10, 20, 3)
    amps = np.linspace(0, 10, 5)

    settables = [DummyBatchedSettable(), DummyBatchedSettable()]
    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(settables)
    meas_ctrl.settables(settables)
    meas_ctrl.setpoints_grid([times, amps])
    meas_ctrl.gettables(gettable)
    dset = meas_ctrl.run("2D batched", save_data=False)

    exp_sp = grid_setpoints([times, amps])
    assert np.array_equal(exp_sp, meas_ctrl._setpoints)
    assert np.array_equal(dset["x0"].values, exp_sp[0])
    assert np.array_equal(dset["x1"].values, exp_sp[1])

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


def test_iterative_outer_loop_with_inner_batched_2d(meas_ctrl, parameters):
    meas_ctrl.settables([parameters.t, parameters.freq])
    times = np.linspace(0, 15, 20)
    freqs = np.linspace(0.1, 1, 10)
    setpoints = [times, freqs]
    meas_ctrl.setpoints_grid(setpoints)

    # Using the same gettable for test purposes
    meas_ctrl.gettables([parameters.sig, parameters.sig])

    parameters.t.batched = True
    parameters.freq.batched = False

    # Must be specified otherwise all setpoints will be passed to the settable
    parameters.sig.batch_size = len(times)
    parameters.sig.batched = True

    dset = meas_ctrl.run("iterative-outer-loop-with-inner-batched-2D")

    np_array_setpoints = meas_ctrl._setpoints
    expected_vals = cosine_function(
        t=np_array_setpoints[0],
        frequency=np_array_setpoints[1],
        amplitude=1,
        phase=0,
    )
    assert np.array_equal(dset["y0"].values, expected_vals)


def test_batched_2d_grid_multi_return(meas_ctrl):
    times = np.linspace(10, 20, 3)
    amps = np.linspace(0, 10, 5)

    settables = [DummyBatchedSettable(), DummyBatchedSettable()]
    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(settables)
    gettable.set_return_2d()
    meas_ctrl.settables(settables)
    meas_ctrl.setpoints_grid([times, amps])
    meas_ctrl.gettables(gettable)
    dset = meas_ctrl.run("2D batched multi return")

    exp_sp = grid_setpoints([times, amps])
    assert np.array_equal(exp_sp, meas_ctrl._setpoints)
    assert np.array_equal(dset["x0"].values, exp_sp[0])
    assert np.array_equal(dset["x1"].values, exp_sp[1])

    expected_vals = batched_mock_values(
        np.stack((dset["x0"].values, dset["x1"].values))
    )
    assert np.array_equal(dset["y0"].values, expected_vals[0])
    assert np.array_equal(dset["y1"].values, expected_vals[1])


def test_batched_2d_grid_multi_return_soft_avg(meas_ctrl):
    x0 = np.arange(5)
    x1 = np.linspace(5, 10, 5)
    settables = [DummyBatchedSettable(), DummyBatchedSettable()]
    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(settables)
    gettable.noise = 0.2
    gettable.set_return_2d()
    meas_ctrl.settables(settables)
    meas_ctrl.setpoints_grid([x0, x1])
    meas_ctrl.gettables(gettable)
    noisy_dset = meas_ctrl.run("noisy_batched_grid")

    expected_vals = batched_mock_values(
        np.stack((noisy_dset["x0"].values, noisy_dset["x1"]))
    )
    yn_0 = abs(noisy_dset["y0"].values - expected_vals[0])
    yn_1 = abs(noisy_dset["y1"].values - expected_vals[1])

    avg_dset = meas_ctrl.run("avg_batched_grid", soft_avg=1000)
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


def test_batched_2d_arbitrary(meas_ctrl):
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
    meas_ctrl.settables(settables)
    meas_ctrl.setpoints(setpoints)
    # settables are passed for test purposes only, this is not a design pattern!
    meas_ctrl.gettables(DummyBatchedGettable(settables))
    dset = meas_ctrl.run()

    assert TUID.is_valid(dset.attrs["tuid"])

    expected_vals = batched_mock_values(x)

    assert np.array_equal(dset["x0"].values, x)
    assert np.array_equal(dset["x1"].values, y)
    assert np.array_equal(dset["y0"].values, expected_vals)


def test_variable_return_batched(meas_ctrl):
    counter_param = ManualParameter("counter", initial_value=0)

    def v_size(setpoints):
        idx = counter_param() % 3
        counter_param(counter_param() + 1)
        if idx == 0:
            return 2 * setpoints[:7]
        if idx == 1:
            return 2 * setpoints[:4]
        return 2 * setpoints[:]

    setpoints = np.arange(30.0)
    settable = DummyBatchedSettable()
    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(settable)
    gettable.get_func = v_size
    meas_ctrl.settables(settable)
    meas_ctrl.setpoints(setpoints)
    meas_ctrl.gettables(gettable)
    dset = meas_ctrl.run("varying")

    assert np.array_equal(dset["x0"], setpoints)
    assert np.array_equal(dset["y0"], 2 * setpoints)


def test_variable_return_batched_soft_avg(meas_ctrl):
    counter_param = ManualParameter("counter", initial_value=0)

    def v_size(setpoints):
        idx = counter_param() % 3
        counter_param(counter_param() + 1)
        if idx == 0:
            return 2 * setpoints[:7]
        if idx == 1:
            return 2 * setpoints[:4]
        return 2 * setpoints[:]

    setpoints = np.arange(30.0)
    settable = DummyBatchedSettable()
    # settables are passed for test purposes only, this is not a design pattern!
    gettable = DummyBatchedGettable(settable)
    gettable.get_func = v_size
    gettable.noise = 0.25
    meas_ctrl.settables(settable)
    meas_ctrl.setpoints(setpoints)
    meas_ctrl.gettables(gettable)
    dset = meas_ctrl.run("varying", soft_avg=1000)

    assert np.array_equal(dset["x0"], setpoints)
    np.testing.assert_array_almost_equal(dset.y0, 2 * setpoints, decimal=2)


def test_iterative_3d_grid(meas_ctrl, parameters):
    times = np.linspace(0, 5, 2)
    amps = np.linspace(-1, 1, 3)
    freqs = np.linspace(41000, 82000, 2)

    meas_ctrl.settables([parameters.t, parameters.amp, parameters.freq])
    meas_ctrl.setpoints_grid([times, amps, freqs])

    exp_sp = grid_setpoints([times, amps, freqs])

    meas_ctrl.gettables(parameters.sig)
    dset = meas_ctrl.run()

    assert np.array_equal(meas_ctrl._setpoints, exp_sp)

    assert TUID.is_valid(dset.attrs["tuid"])

    expected_vals = cosine_function(
        t=exp_sp[0], amplitude=exp_sp[1], frequency=exp_sp[2], phase=0
    )

    assert np.array_equal(dset["x0"].values, exp_sp[0])
    assert np.array_equal(dset["x1"].values, exp_sp[1])
    assert np.array_equal(dset["x2"].values, exp_sp[2])
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


def test_iterative_3d_multi_return(meas_ctrl, parameters):
    times = np.linspace(0, 5, 2)
    amps = np.linspace(-1, 1, 3)
    freqs = np.linspace(41000, 82000, 2)

    meas_ctrl.settables([parameters.t, parameters.amp, parameters.freq])
    meas_ctrl.setpoints_grid([times, amps, freqs])
    meas_ctrl.gettables([parameters.DualWave(), parameters.sig, parameters.DualWave()])
    dset = meas_ctrl.run()

    exp_sp = grid_setpoints([times, amps, freqs])
    exp_y0 = exp_y3 = sine_function(
        t=exp_sp[0], amplitude=exp_sp[1], frequency=exp_sp[2], phase=0
    )
    exp_y1 = exp_y2 = exp_y4 = cosine_function(
        t=exp_sp[0], amplitude=exp_sp[1], frequency=exp_sp[2], phase=0
    )

    np.testing.assert_array_equal(dset["y0"], exp_y0)
    np.testing.assert_array_equal(dset["y1"], exp_y1)
    np.testing.assert_array_equal(dset["y2"], exp_y2)
    np.testing.assert_array_equal(dset["y3"], exp_y3)
    np.testing.assert_array_equal(dset["y4"], exp_y4)


def test_batched_3d_multi_return_soft_averaging(meas_ctrl):
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
    nd_gettable.set_return_2d()
    # settables are passed for test purposes only, this is not a design pattern!
    noisy_gettable = DummyBatchedGettable([batched_settable_t])
    noisy_gettable.get_func = v_get
    noisy_gettable.noise = 0.25

    meas_ctrl.settables([batched_settable_t, batched_settable_0, batched_settable_1])
    meas_ctrl.setpoints_grid([times, amps, freqs])
    meas_ctrl.gettables([noisy_gettable, nd_gettable])
    dset = meas_ctrl.run(soft_avg=1000)

    exp_sp = grid_setpoints([times, amps, freqs])
    np.testing.assert_array_almost_equal(dset.y0, exp_sp[0], decimal=2)
    np.testing.assert_array_almost_equal(
        dset.y1, batched_mock_values(exp_sp[1]), decimal=4
    )
    np.testing.assert_array_almost_equal(
        dset.y2, batched_mock_values(exp_sp[2]), decimal=4
    )


def test_batched_attr_raises(meas_ctrl, parameters):
    times = np.linspace(0, 5, 3)
    freqs = np.linspace(41000, 82000, 8)

    parameters.freq.batched = True
    parameters.freq.batch_size = 8
    parameters.t.batched = False

    meas_ctrl.settables([parameters.freq, parameters.t])
    meas_ctrl.setpoints_grid([freqs, times])
    # Ensure forgetting this raises exception
    # sig.batched = True
    meas_ctrl.gettables(parameters.sig)

    with pytest.raises(RuntimeError):
        meas_ctrl.run()


def test_batched_grid_mixed(meas_ctrl, parameters):
    times = np.linspace(0, 5, 3)
    amps = np.linspace(-1, 1, 4)
    freqs = np.linspace(41000, 82000, 8)
    other_freqs = np.linspace(46000, 88000, 8)

    parameters.freq.batched = True
    parameters.freq.batch_size = 5  # odd size for extra test
    parameters.t.batched = False
    parameters.other_freq.batched = True
    parameters.other_freq.batch_size = 3  # odd size and different value for extra test
    parameters.amp.batched = False

    parameters.sig2.batched = True

    settables = [
        parameters.freq,
        parameters.t,
        parameters.other_freq,
        parameters.amp,
    ]
    meas_ctrl.settables(settables)
    setpoints = [freqs, times, other_freqs, amps]
    meas_ctrl.setpoints_grid(setpoints)
    meas_ctrl.gettables(parameters.sig2)
    dset = meas_ctrl.run("bla", soft_avg=2)

    assert isinstance(parameters.freq(), Iterable)
    assert not isinstance(parameters.t(), Iterable)
    assert isinstance(parameters.other_freq(), Iterable)
    assert not isinstance(parameters.amp(), Iterable)

    exp_sp = grid_setpoints(setpoints, settables=settables)
    assert np.array_equal(meas_ctrl._setpoints, exp_sp)

    _, _, _, _ = (
        parameters.freq(exp_sp[0]),
        parameters.t(exp_sp[1]),
        parameters.other_freq(exp_sp[2]),
        parameters.amp(exp_sp[3]),
    )
    assert np.array_equal(dset["y0"].values, parameters.sig2())


def test_adaptive_nelder_mead(meas_ctrl, dummy_parabola):
    meas_ctrl.settables([dummy_parabola.x, dummy_parabola.y])
    af_pars = {
        "adaptive_function": optimize.minimize,
        "x0": [-50, -50],
        "method": "Nelder-Mead",
    }
    dummy_parabola.noise(0.5)
    meas_ctrl.gettables(dummy_parabola.parabola)
    dset = meas_ctrl.run_adaptive("nelder_mead", af_pars)

    assert dset["x0"][-1] < 0.7
    assert dset["x1"][-1] < 0.7
    assert dset["y0"][-1] < 0.7


def test_adaptive_multi_return(meas_ctrl):
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
    class ResonancePlus:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self.name = ["resonance", "dummy"]
            self.label = ["Amplitude", "Dummy"]
            self.unit = ["V", "V"]

        def get(self):
            return [lorenz(), dummy()]

    meas_ctrl.settables(freq)
    af_pars = {
        "adaptive_function": adaptive.learner.Learner1D,
        "goal": lambda l: l.npoints > 5,
        "bounds": (6e9, 7e9),
    }
    meas_ctrl.gettables([ResonancePlus(), dummy])
    dset = meas_ctrl.run_adaptive("multi return", af_pars)
    assert (dset["y1"].values == 99).all()
    assert (dset["y2"].values == 99).all()


def test_adaptive_bounds_1d(meas_ctrl):
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

    meas_ctrl.settables(freq)
    af_pars = {
        "adaptive_function": adaptive.learner.Learner1D,
        "goal": lambda l: l.npoints > 20 * 20,
        "bounds": (6e9, 7e9),
    }
    meas_ctrl.gettables(resonance)
    _ = meas_ctrl.run_adaptive("adaptive sample", af_pars)


def test_adaptive_sampling(meas_ctrl, dummy_parabola):
    dummy_parabola.noise(0)
    meas_ctrl.settables([dummy_parabola.x, dummy_parabola.y])
    af_pars = {
        "adaptive_function": adaptive.learner.Learner2D,
        "goal": lambda l: l.npoints > 20 * 20,
        "bounds": ((-50, 50), (-20, 30)),
    }
    meas_ctrl.gettables(dummy_parabola.parabola)
    _ = meas_ctrl.run_adaptive("adaptive sample", af_pars)


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
    meas_ctrl, dummy_parabola, lazy_set_parameter, lazy_set_argument, lazy_set_turned_on
):
    dummy_parabola.x.set = Mock(wraps=dummy_parabola.x.set)
    dummy_parabola.y.set = Mock(wraps=dummy_parabola.y.set)

    dummy_parabola.noise(0)
    meas_ctrl.settables([dummy_parabola.x, dummy_parabola.y])

    num_x = 5
    num_y = 20

    def _adaptive_function(meas_func, **_):
        """A simple adaptive function that executes a 2D sweep and conforms with
        the accepted interface for an adaptive function."""
        points = np.column_stack(
            [
                np.tile(np.linspace(-50, 50, num_x), num_y),
                np.repeat(np.linspace(-20, 30, num_y), num_x),
            ],
        )

        for setpoint in points:
            _ = meas_func(setpoint)

        # The meas_ctrl takes care of intercepting the measured values so no return
        # is specified here

    af_pars = {"adaptive_function": _adaptive_function}
    meas_ctrl.gettables(dummy_parabola.parabola)
    meas_ctrl.lazy_set(lazy_set_parameter)
    _ = meas_ctrl.run_adaptive("adaptive sample", af_pars, lazy_set=lazy_set_argument)

    # if lazy_set is turned on, verify that the adaptive algorithm did not set the
    # parameters each time
    if lazy_set_turned_on:
        assert dummy_parabola.x.set.call_count == num_x * num_y
        assert dummy_parabola.y.set.call_count == num_y
    else:
        assert dummy_parabola.x.set.call_count == num_x * num_y
        assert dummy_parabola.y.set.call_count == num_x * num_y


@pytest.mark.skipif(not WITH_SKOPTLEARNER, reason="scikit-optimize is not installed")
def test_adaptive_skoptlearner(meas_ctrl, dummy_parabola):
    dummy_parabola.noise(0)
    meas_ctrl.settables([dummy_parabola.x, dummy_parabola.y])
    af_pars = {
        "adaptive_function": SKOptLearner,
        "goal": lambda l: l.npoints > 15,
        "dimensions": [(-50.0, +50.0), (-20.0, +30.0)],
        "base_estimator": "gp",
        "acq_func": "EI",
        "acq_optimizer": "lbfgs",
    }
    meas_ctrl.gettables(dummy_parabola.parabola)
    _ = meas_ctrl.run_adaptive("skopt", af_pars)


def test_progress_callback(meas_ctrl, parameters):
    progress_param = ManualParameter("progress", initial_value=0)

    def set_progress_param_callable(progress):
        progress_param(progress)

    meas_ctrl.on_progress_callback(set_progress_param_callable)
    assert progress_param() == 0

    xvals = np.linspace(0, 2 * np.pi, 31)
    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(xvals)
    meas_ctrl.gettables(parameters.sig)
    meas_ctrl.run()

    assert progress_param() == 100


def test_meas_ctrl_plotmon_integration(meas_ctrl, parameters):
    plotmon = PlotMonitor_pyqt("plotmon_meas_ctrl")

    meas_ctrl.instr_plotmon(plotmon.name)

    assert len(plotmon.tuids()) == 0

    times = np.linspace(0, 5, 18)
    amps = np.linspace(-1, 1, 5)

    meas_ctrl.settables([parameters.t, parameters.amp])
    meas_ctrl.setpoints_grid([times, amps])
    meas_ctrl.gettables(parameters.sig)
    meas_ctrl.run("2D Cosine test")

    assert len(plotmon.tuids()) > 0

    plotmon.close()
    meas_ctrl.instr_plotmon("")


def test_meas_ctrl_insmon_integration():
    inst_mon = InstrumentMonitor("insmon_meas_ctrl")
    assert inst_mon.widget.getNodes()
    inst_mon.close()


def test_instrument_settings_from_disk(dummy_parabola, tmp_test_data_dir):
    load_settings_onto_instrument(
        dummy_parabola, TUID("20200814-134652-492-fbf254"), tmp_test_data_dir
    )
    assert dummy_parabola.x() == 40.0
    assert dummy_parabola.y() == 90.0
    assert dummy_parabola.z() == -20.0
    assert dummy_parabola.delay() == 10.0

    non_existing = DummyParHolder("the_mac")
    with pytest.raises(ValueError, match='Instrument "the_mac" not found in snapshot'):
        load_settings_onto_instrument(
            non_existing, TUID("20200814-134652-492-fbf254"), tmp_test_data_dir
        )

    non_existing.close()


def test_exception_not_silenced_set_get(meas_ctrl, parameters):
    class BadSetter:
        def __init__(self):
            self.name = "spec_freq"
            self.label = "Frequency"
            self.unit = "Hz"
            self.soft = True
            # simulate user mistake
            # self.non_existing_param = param

        def prepare(self):
            # Exception expected
            print(self.non_existing_param)  # type: ignore # pylint: disable=no-member

        def set(self, _):
            pass

    meas_ctrl.settables(BadSetter())
    meas_ctrl.setpoints(np.linspace(0, 1, 10))
    meas_ctrl.gettables(parameters.freq)
    with pytest.raises(
        AttributeError,
        match="'BadSetter' object has no attribute 'non_existing_param'",
    ):
        meas_ctrl.run("This raises exception as expected")

    class BadGetter:
        def __init__(self):
            self.name = "mag"
            self.label = "Magnitude"
            self.unit = "a.u."
            self.soft = True
            # self.non_existing_param = param2

        def prepare(self):
            pass

        def get(self):
            print(self.non_existing_param)  # type: ignore # pylint: disable=no-member

        def finish(self):
            pass

    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(np.linspace(0, 1, 10) * 1e-9)
    meas_ctrl.gettables(BadGetter())
    with pytest.raises(
        AttributeError,
        match="'BadGetter' object has no attribute 'non_existing_param'",
    ):
        meas_ctrl.run("This raises exception as expected")


def test_keyboard_interrupt(meas_ctrl):
    class GettableUserInterrupt:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self.name = "amp"
            self.label = "Amplitude"
            self.unit = "V"
            self._num_interrupts = 1

        def get(self):
            if time_par() == 3:
                for _ in range(self._num_interrupts):
                    # This same signal is sent when pressing `ctrl` + `c` or the
                    # "Stop kernel" button is pressed in a Jupyter(Lab) notebook
                    signal.raise_signal(signal.SIGINT)
            return time_par()

    time_par = ManualParameter(name="time", unit="s", label="Measurement Time")
    time_par.batched = False

    gettable = GettableUserInterrupt()

    meas_ctrl.settables(time_par)
    meas_ctrl.setpoints(np.arange(10))
    meas_ctrl.gettables(gettable)
    with pytest.raises(KeyboardInterrupt):
        meas_ctrl.run("interrupt_test")

    dset = meas_ctrl._dataset
    # we stop after 4th iteration
    assert sum(np.isnan(dset.y0) ^ 1) == 4

    # Ensure force stop still possible
    gettable._num_interrupts = 5
    meas_ctrl.settables(time_par)
    meas_ctrl.setpoints(np.arange(10))
    meas_ctrl.gettables(gettable)

    with pytest.raises(KeyboardInterrupt):
        meas_ctrl.run("interrupt_test_force")

    dset = meas_ctrl._dataset
    # we stop right away
    assert sum(np.isnan(dset.y0) ^ 1) == 3


def test_experiment_data(meas_ctrl):
    assert meas_ctrl.experiment_data.parameters == {}

    # Create some experiment_data
    experiment_data_1 = {
        "param_a": {
            "value": 0.25,
            "label": "parameter a",
            "unit": "Hz",
        },
        "param_b": {
            "value": -10,
        },
    }

    meas_ctrl.set_experiment_data(experiment_data_1)

    assert meas_ctrl.experiment_data.parameters.keys() == experiment_data_1.keys()
    assert meas_ctrl.experiment_data.param_a() == experiment_data_1["param_a"]["value"]
    assert (
        meas_ctrl.experiment_data.param_a.unit == experiment_data_1["param_a"]["unit"]
    )
    assert (
        meas_ctrl.experiment_data.param_a.label == experiment_data_1["param_a"]["label"]
    )
    assert meas_ctrl.experiment_data.param_b() == experiment_data_1["param_b"]["value"]

    # Set some new experiment_data, overwriting the previously saved values
    experiment_data_2 = {
        "param_b": {
            "value": 60,
        },
        "param_c": {
            "value": 12.4,
        },
    }

    meas_ctrl.set_experiment_data(experiment_data_2)

    assert meas_ctrl.experiment_data.parameters.keys() == experiment_data_2.keys()
    assert meas_ctrl.experiment_data.param_b() == experiment_data_2["param_b"]["value"]
    assert meas_ctrl.experiment_data.param_c() == experiment_data_2["param_c"]["value"]

    # Set some new experiment_data, keeping the old parameters and changing
    # them if necessary
    experiment_data_3 = {
        "param_b": {
            "value": 70,
        },
        "param_d": {
            "value": 1.1,
        },
    }

    meas_ctrl.set_experiment_data(experiment_data_3, overwrite=False)

    assert set(meas_ctrl.experiment_data.parameters.keys()) == {
        "param_b",
        "param_c",
        "param_d",
    }
    assert meas_ctrl.experiment_data.param_b() == experiment_data_3["param_b"]["value"]
    assert meas_ctrl.experiment_data.param_c() == experiment_data_2["param_c"]["value"]
    assert meas_ctrl.experiment_data.param_d() == experiment_data_3["param_d"]["value"]

    # Clear all experiment_data
    meas_ctrl.clear_experiment_data()

    assert meas_ctrl.experiment_data.parameters == {}


def test_repr_new_and_closed():
    meas_ctrl = MeasurementControl("a_meas_ctrl")
    assert repr(meas_ctrl)
    meas_ctrl.close()
    assert repr(meas_ctrl)
    meas_ctrl.close()


def make_settable_set(batched_mask: Sequence[bool]):
    settables = []
    for i, m in enumerate(batched_mask):
        par = ManualParameter(name=f"x{i}", label=f"X{i}", unit="s")
        par.batched = m
        settables.append(par)
    return settables


@pytest.fixture(scope="module")
def setpoints_iterative():
    return (np.array([0, 1, 2, 3]), np.array([4, 5]), np.array([-1, -2]))


@pytest.fixture(scope="module")
def setpoints_batched():
    base_batched = [1, 2, 6]
    return (
        np.array(base_batched),
        np.array(base_batched) * 2,
        np.array(base_batched) * 3,
    )


def test_grid_setpoints_mixed_simple_reversed(setpoints_iterative, setpoints_batched):
    # Most simple case reversed order
    batched_mask = [True, False]
    settables = make_settable_set(batched_mask)
    it_i = iter(setpoints_iterative)
    it_b = iter(setpoints_batched)
    setpoints = [(next(it_b) if m else next(it_i)) for m in batched_mask]
    gridded = grid_setpoints(setpoints, settables=settables)
    expected = np.column_stack(
        [
            np.tile(setpoints_batched[0], len(setpoints_iterative[0])),
            np.repeat(setpoints_iterative[0], len(setpoints_batched[0])),
        ]
    ).T
    np.testing.assert_array_equal(gridded, expected)


def test_grid_setpoints_mixed_two_batched(setpoints_iterative, setpoints_batched):
    # Several batched settables
    batched_mask = [False, True, True]
    settables = make_settable_set(batched_mask)
    it_i = iter(setpoints_iterative)
    it_b = iter(setpoints_batched)
    setpoints = [(next(it_b) if m else next(it_i)) for m in batched_mask]
    gridded = grid_setpoints(setpoints, settables=settables)
    expected = np.column_stack(
        [
            np.repeat(
                setpoints_iterative[0],
                len(setpoints_batched[0]) * len(setpoints_batched[0]),
            ),
            np.tile(
                setpoints_batched[0],
                len(setpoints_iterative[0]) * len(setpoints_batched[1]),
            ),
            np.tile(
                np.repeat(setpoints_batched[1], len(setpoints_batched[0])),
                len(setpoints_iterative[0]),
            ),
        ]
    ).T
    np.testing.assert_array_equal(gridded, expected)


def test_grid_setpoints_mixed_two_batched_two_iterative(
    setpoints_iterative, setpoints_batched
):
    # Several batched settables and several iterative settables
    batched_mask = [False, True, False, True]
    settables = make_settable_set(batched_mask)
    settables[-1].batch_size = 2
    it_i = iter(setpoints_iterative)
    it_b = iter(setpoints_batched)
    setpoints = [(next(it_b) if m else next(it_i)) for m in batched_mask]
    gridded = grid_setpoints(setpoints, settables=settables)
    expected = np.column_stack(
        [
            np.tile(
                np.repeat(
                    setpoints_iterative[0],
                    len(setpoints_batched[1]) * len(setpoints_batched[0]),
                ),
                len(setpoints_iterative[1]),
            ),
            np.tile(
                np.repeat(setpoints_batched[0], len(setpoints_batched[1])),
                len(setpoints_iterative[0]) * len(setpoints_iterative[1]),
            ),
            np.repeat(
                setpoints_iterative[1],
                len(setpoints_batched[0])
                * len(setpoints_batched[1])
                * len(setpoints_iterative[0]),
            ),
            np.tile(
                setpoints_batched[1],
                len(setpoints_iterative[0])
                * len(setpoints_iterative[1])
                * len(setpoints_batched[0]),
            ),
        ]
    ).T
    np.testing.assert_array_equal(gridded, expected)


def test_grid_setpoints():
    x = np.arange(5)
    y = np.linspace(-1, 1, 3)

    sp = grid_setpoints([x, y])
    assert sp[0].all() == np.tile(np.arange(5), 3).all()
    assert sp[1].all() == np.repeat(y, 5).all()

    z = np.linspace(100, 200, 2)
    sp = grid_setpoints([x, y, z])
    assert all(e in sp[0] for e in x)
    assert all(e in sp[1] for e in y)
    assert all(e in sp[2] for e in z)


def test_typed_settables(meas_ctrl, parameters):
    times = np.linspace(0, 5, 20)
    atts = list(range(0, 60, 10))
    att_settable = ManualParameter(
        "out0_att",
        label="Output 0 attenuation",
        unit="dB",
        vals=vals.Multiples(2, min_value=0, max_value=60),
        set_parser=int,
        get_parser=int,
    )

    meas_ctrl.settables([parameters.t, att_settable])
    meas_ctrl.setpoints_grid([times, atts])

    meas_ctrl.gettables(parameters.sig)
    # Should not raise TypeError
    _ = meas_ctrl.run()


def test_typed_settables_batched(meas_ctrl):
    times = list(range(20))
    freqs = list(range(1, 11))
    setpoints = [times, freqs]
    meas_ctrl.setpoints_grid(setpoints)

    strict_t = ManualParameter(
        "time",
        unit="s",
        vals=vals.Arrays(valid_types=[int]),
    )
    strict_freq = ManualParameter(
        "frequency",
        unit="Hz",
        vals=vals.Ints(),
    )
    meas_ctrl.settables([strict_t, strict_freq])

    # Using the same gettable for test purposes
    def cosine_model():
        return cosine_function(
            strict_t(), amplitude=1, frequency=strict_freq(), phase=0
        )

    sig = Parameter(name="sig", label="Signal level", unit="V", get_cmd=cosine_model)
    meas_ctrl.gettables([sig, sig])

    strict_t.batched = True
    strict_freq.batched = False

    sig.batch_size = len(times)
    sig.batched = True

    # Should not raise TypeError
    _ = meas_ctrl.run("iterative-outer-loop-with-inner-batched-2D")


def test_print_progress_no_setpoints(meas_ctrl_empty, mocker):
    mocker.patch(
        "quantify_core.measurement.control.MeasurementControl._get_fracdone",
        side_effect=[0],
    )
    with pytest.raises(
        ValueError, match="No setpoints available, progress cannot be defined"
    ):
        meas_ctrl_empty.print_progress()


def test_get_idn(meas_ctrl):
    assert meas_ctrl.get("IDN") == {
        "vendor": "Quantify",
        "model": f"{meas_ctrl.__module__}.{meas_ctrl.__class__.__name__}",
        "serial": meas_ctrl.name,
        "firmware": _quantify_version,
    }


@pytest.fixture
def setup_instrument_and_channels():
    dummy_instrument = DummyInstrument()
    channel_list = [
        DummyChannel(dummy_instrument, "channel0", "channel0"),
        DummyChannel(dummy_instrument, "channel1", "channel1"),
    ]
    ct = ChannelTuple(
        dummy_instrument,
        "dummy_channels",
        chan_type=DummyChannel,
        chan_list=channel_list,
    )
    dummy_instrument.add_submodule("gizmos", ct)
    measurement_control = MeasurementControl("measurement_control")
    return channel_list, dummy_instrument, measurement_control


def test_channel_tuple_integration_with_measurement_control(
    setup_instrument_and_channels,
):
    channel_list, dummy_instrument, measurement_control = setup_instrument_and_channels
    measurement_control.settables(channel_list[0].temperature)
    measurement_control.gettables(dummy_instrument.dac1)
    measurement_control.setpoints(np.linspace(20.0, 24.0, 5))
    measurement_control.run(save_data=False)


def test_run_in_thread(meas_ctrl, parameters):
    # Run measurement control in separate thread and check
    # if no exception raised.
    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(np.array([0, 1, 2, 3]))
    meas_ctrl.gettables(parameters.sig)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(meas_ctrl.run)
        future.result()  # This re-raises the exception


@pytest.fixture()
def myhandler(mocker):
    h = mocker.Mock()
    prev = signal.signal(signal.SIGINT, h)
    yield h
    signal.signal(signal.SIGINT, prev)


def test_clean_deinstall_of_interrupt_handler(meas_ctrl, parameters, myhandler):
    meas_ctrl.settables(parameters.t)
    meas_ctrl.setpoints(np.array([0, 1, 2, 3]))
    meas_ctrl.gettables(parameters.sig)
    meas_ctrl.run()
    assert signal.getsignal(signal.SIGINT) == myhandler


@pytest.mark.parametrize("num_channels, real_imag", [(1, True), (2, False), (10, True)])
def test_process_acquired_data(meas_ctrl, num_channels: int, real_imag: bool):
    # arrange
    mock_number = 4815 + 162342j
    mock_results = np.array([mock_number], dtype=np.complex64)
    mock_dataset = Dataset(
        {
            i: (
                [f"acq_index_{i}"],
                mock_results * i,
                {"acq_protocol": "SSBIntegrationComplex"},
            )
            for i in range(num_channels)
        }
    )

    # act
    processed_data = meas_ctrl._process_acquired_data(mock_dataset, False, real_imag)

    def transform_complex(c: complex) -> tuple:
        if real_imag:
            return (c.real, c.imag)
        else:
            return (abs(c), np.angle(c, deg=True))

    expected_data: tuple = tuple(
        np.array([transform_complex(mock_number * i)[elem]], dtype=np.float32)
        for i in range(num_channels)
        for elem in [0, 1]
    )

    # assert
    np.testing.assert_array_almost_equal(processed_data, expected_data, decimal=5)


def test_process_acquired_data_thresholded_acq(meas_ctrl):
    mock_number = 0.66
    mock_results = np.array([mock_number], dtype=np.float64)
    mock_dataset = Dataset(
        {
            1: (
                ["acq_index_1"],
                mock_results,
                {"acq_protocol": "ThresholdedAcquisition"},
            )
        }
    )

    processed_data = meas_ctrl._process_acquired_data(mock_dataset, False, False)

    expected_data = [np.array([mock_number], dtype=np.float64)]

    np.testing.assert_array_almost_equal(processed_data, expected_data, decimal=5)
