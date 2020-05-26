import time
import xarray as xr
import numpy as np
from qcodes import ManualParameter, Parameter
from quantify.measurement.control import MeasurementControl, \
    tile_setpoints_grid
from quantify import set_datadir
from quantify.data.types import TUID
from quantify.visualization.pyqt_plotmon import PlotMonitor_pyqt


# Define some helpers that are used in the tests


def CosFunc(t, amplitude, frequency, phase):
    """A simple cosine function"""
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


# Parameters are created to emulate a system being measured
t = ManualParameter('t', initial_value=1, unit='s', label='Time')
amp = ManualParameter('amp', initial_value=1, unit='V', label='Amplitude')
freq = ManualParameter('freq', initial_value=1, unit='Hz', label='Frequency')


def cosine_model():
    return CosFunc(t(), amplitude=amp(), frequency=freq(), phase=0)


# We wrap our function in a Parameter to be able to give
sig = Parameter(name='sig', label='Signal level', unit='V', get_cmd=cosine_model)


class DummyDetector:
    def __init__(self):
        self.name = 'Dummy'
        self.unit = 'W'
        self.label = 'Watts Per Dumb'
        self.setpoints = []
        self.noise = 0
        self.delay = 0
        self.internal = False

    def prepare(self, setpoints):
        self.setpoints = setpoints

    def get(self):
        x = self.setpoints
        noise = self.noise * (np.random.rand(2, len(x)) - 0.5)
        #data = np.array([np.sin(x / np.pi), np.cos(x / np.pi)])
        data = x
        data += noise
        time.sleep(self.delay)
        return data


class TestMeasurementControl:

    @classmethod
    def setup_class(cls):
        cls.MC = MeasurementControl(name='MC')
        # ensures the default datadir is used which is excluded from git
        set_datadir(None)

    @classmethod
    def teardown_class(cls):
        cls.MC.close()
        set_datadir(None)

    def test_MeasurementControl_name(self):
        assert self.MC.name == 'MC'

    def test_set_setpoints(self):
        x = np.linspace(0, 10, 11)
        self.MC.set_setpoints(x)
        assert np.array_equal(self.MC._setpoints[:, 0], x)

        x = np.random.rand(15, 2)
        self.MC.set_setpoints(x)
        assert np.array_equal(self.MC._setpoints, x)

        x = np.random.rand(15, 4)
        self.MC.set_setpoints(x)
        assert np.array_equal(self.MC._setpoints, x)

    def test_soft_sweep_1D(self):
        xvals = np.linspace(0, 2*np.pi, 31)

        self.MC.set_setpars(t)
        self.MC.set_setpoints(xvals)
        self.MC.set_getpars(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs['tuid'])

        expected_vals = CosFunc(t=xvals, amplitude=1, frequency=1, phase=0)

        assert (np.array_equal(dset['x0'].values, xvals))
        assert (np.array_equal(dset['y0'].values, expected_vals))

        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {'x0', 'y0'}
        assert (np.array_equal(dset['x0'], xvals))
        assert dset['x0'].attrs == {'name': 't', 'long_name': 'Time', 'unit': 's'}
        assert dset['y0'].attrs == {'name': 'sig', 'long_name': 'Signal level', 'unit': 'V'}

    def test_hard_sweep_1D(self):
        x = np.linspace(0, 10, 5)
        self.MC.set_setpars(t)
        self.MC.set_setpoints(x)
        self.MC.set_getpars(DummyDetector())
        dset = self.MC.run()

    def test_soft_sweep_2D_grid(self):

        times = np.linspace(0, 5, 20)
        amps = np.linspace(-1, 1, 5)

        self.MC.set_setpars([t, amp])
        self.MC.set_setpoints_grid([times, amps])

        exp_sp = tile_setpoints_grid([times, amps])
        assert (np.array_equal(self.MC._setpoints, exp_sp))

        self.MC.set_getpars(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs['tuid'])

        expected_vals = CosFunc(t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=1, phase=0)

        assert (np.array_equal(dset['x0'].values, exp_sp[:, 0]))
        assert (np.array_equal(dset['x1'].values, exp_sp[:, 1]))
        assert (np.array_equal(dset['y0'].values, expected_vals))

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {'x0', 'x1', 'y0'}

        assert (all(e in dset['x0'].values for e in times))
        assert (all(e in dset['x1'].values for e in amps))

        assert dset['x0'].attrs == {'name': 't', 'long_name': 'Time', 'unit': 's'}
        assert dset['x1'].attrs == {'name': 'amp', 'long_name': 'Amplitude', 'unit': 'V'}
        assert dset['y0'].attrs == {'name': 'sig', 'long_name': 'Signal level', 'unit': 'V'}

    def test_soft_sweep_2D_arbitrary(self):

        r = np.linspace(0, 1.5, 50)
        dt = np.linspace(0, 1, 50)

        f = 10

        # create a fancy polar coordinates loop
        theta = np.cos(2*np.pi*f*dt)

        def polar_coords(r, theta):

            x = r*np.cos(2*np.pi*theta)
            y = r*np.sin(2*np.pi*theta)
            return x, y

        x, y = polar_coords(r, theta)
        setpoints = np.column_stack([x, y])

        self.MC.set_setpars([t, amp])
        self.MC.set_setpoints(setpoints)
        self.MC.set_getpars(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs['tuid'])

        expected_vals = CosFunc(t=x, amplitude=y, frequency=1, phase=0)

        assert (np.array_equal(dset['x0'].values, x))
        assert (np.array_equal(dset['x1'].values, y))
        assert (np.array_equal(dset['y0'].values, expected_vals))

    def test_soft_sweep_3D_grid(self):

        times = np.linspace(0, 5, 2)
        amps = np.linspace(-1, 1, 3)
        freqs = np.linspace(41000, 82000, 2)

        self.MC.set_setpars([t, amp, freq])
        self.MC.set_setpoints_grid([times, amps, freqs])

        exp_sp = tile_setpoints_grid([times, amps, freqs])
        assert (np.array_equal(self.MC._setpoints, exp_sp))

        self.MC.set_getpars(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs['tuid'])

        expected_vals = CosFunc(t=exp_sp[:, 0], amplitude=exp_sp[:, 1], frequency=exp_sp[:, 2], phase=0)

        assert (np.array_equal(dset['x0'].values, exp_sp[:, 0]))
        assert (np.array_equal(dset['x1'].values, exp_sp[:, 1]))
        assert (np.array_equal(dset['x2'].values, exp_sp[:, 2]))
        assert (np.array_equal(dset['y0'].values, expected_vals))

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {'x0', 'x1', 'x2', 'y0'}
        assert (all(e in dset['x0'] for e in times))
        assert (all(e in dset['x1'] for e in amps))
        assert (all(e in dset['x2'] for e in freqs))

        assert dset['x0'].attrs == {'name': 't', 'long_name': 'Time', 'unit': 's'}
        assert dset['x2'].attrs == {'name': 'freq', 'long_name': 'Frequency', 'unit': 'Hz'}
        assert dset['y0'].attrs == {'name': 'sig', 'long_name': 'Signal level', 'unit': 'V'}

    def test_progress_callback(self):

        progress_param = ManualParameter("progress", initial_value=0)

        def set_progress_param_callable(progress):
            progress_param(progress)

        self.MC.on_progress_callback(set_progress_param_callable)

        assert progress_param() == 0

        xvals = np.linspace(0, 2*np.pi, 31)
        self.MC.set_setpars(t)
        self.MC.set_setpoints(xvals)
        self.MC.set_getpars(sig)
        self.MC.run()

        assert progress_param() == 100

    def test_MC_plotmon_integration(self):
        plotmon = PlotMonitor_pyqt('plotmon_MC')

        self.MC.instr_plotmon(plotmon.name)

        assert plotmon.tuid() == 'latest'

        times = np.linspace(0, 5, 18)
        amps = np.linspace(-1, 1, 5)

        self.MC.set_setpars([t, amp])
        self.MC.set_setpoints_grid([times, amps])
        self.MC.set_getpars(sig)
        self.MC.run('2D Cosine test')

        assert plotmon.tuid() != 'latest'

        plotmon.close()
        self.MC.instr_plotmon('')


def test_tile_setpoints_grid():
    x = np.arange(5)
    y = np.linspace(-1, 1, 3)

    sp = tile_setpoints_grid([x, y])
    assert sp[:, 0].all() == np.tile(np.arange(5), 3).all()
    assert sp[:, 1].all() == np.repeat(y, 5).all()

    z = np.linspace(100, 200, 2)
    sp = tile_setpoints_grid([x, y, z])
    assert (all(e in sp[:, 0] for e in x))
    assert (all(e in sp[:, 1] for e in y))
    assert (all(e in sp[:, 2] for e in z))
