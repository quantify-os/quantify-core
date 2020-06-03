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


class NoneSweep:
    def __init__(self, internal):
        self.name = 'none'
        self.unit = 'N'
        self.label = 'None'
        self.internal = internal

    def set(self, val):
        pass


def hardware_mock_values_1D(setpoints):
    return np.array([np.tan(setpoints / np.pi)])


def hardware_mock_values_2D(setpoints):
    return np.array([np.sin(setpoints / np.pi), np.cos(setpoints / np.pi)])


class DummyDetector:
    """
    Args:
        mode (str): Number of data rows to return, supports '1D' and '2D'
    """
    def __init__(self, return_dimensions: str):
        if return_dimensions == '1D':
            self.name = 'dum'
            self.unit = 'W'
            self.label = 'Watts'
            self.mock_fn = hardware_mock_values_1D
        elif return_dimensions == '2D':
            self.name = ['dum', 'mud']
            self.unit = ['W', 'M']
            self.label = ['Watts', 'Matts']
            self.mock_fn = hardware_mock_values_2D
        else:
            raise Exception('Unsupported mode: {}'.format(return_dimensions))
        self.delay = 0
        self.noise = 0
        self.internal = False

    def prepare(self, setpoints):
        self.setpoints = setpoints

    def get(self):
        x = self.setpoints
        noise = self.noise * (np.random.rand(2, len(x)) - .5)
        data = self.mock_fn(x)
        # todo, fix this hack, noise doesn't currently naturally sum with data in 1D
        if self.noise:
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
        self.MC.set_setpars(NoneSweep(internal=False))
        self.MC.set_setpoints(x)
        self.MC.set_getpars(DummyDetector("2D"))
        dset = self.MC.run()

        expected_vals = hardware_mock_values_2D(x)
        assert (np.array_equal(dset['x0'].values, x))
        assert (np.array_equal(dset['y0'].values, expected_vals[0]))
        assert (np.array_equal(dset['y1'].values, expected_vals[1]))

        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {'x0', 'y0', 'y1'}
        assert dset['x0'].attrs == {'name': 'none', 'long_name': 'None', 'unit': 'N'}
        assert dset['y0'].attrs == {'name': 'dum', 'long_name': 'Watts', 'unit': 'W'}
        assert dset['y1'].attrs == {'name': 'mud', 'long_name': 'Matts', 'unit': 'M'}

    def test_soft_averages_hard_sweep_1D(self):
        setpoints = np.arange(50.0)
        self.MC.set_setpars(NoneSweep(internal=False))
        self.MC.set_setpoints(setpoints)
        d = DummyDetector('2D')
        d.noise = 0.4
        self.MC.set_getpars(d)
        noisy_dset = self.MC.run('noisy')
        xn_0 = noisy_dset['x0'].values
        expected_vals = hardware_mock_values_2D(xn_0)
        yn_0 = abs(noisy_dset['y0'].values - expected_vals[0])
        yn_1 = abs(noisy_dset['y1'].values - expected_vals[1])

        self.MC.soft_avg(5000)
        self.MC.set_setpars(NoneSweep(internal=False))
        self.MC.set_setpoints(setpoints)
        self.MC.set_getpars(d)
        avg_dset = self.MC.run('averaged')
        yavg_0 = abs(avg_dset['y0'].values - expected_vals[0])
        yavg_1 = abs(avg_dset['y1'].values - expected_vals[1])

        np.testing.assert_array_equal(xn_0, setpoints)
        assert np.mean(yn_0) > np.mean(yavg_0)
        assert np.mean(yn_1) > np.mean(yavg_1)

        np.testing.assert_array_almost_equal(yavg_0, np.zeros(len(xn_0)), decimal=2)
        np.testing.assert_array_almost_equal(yavg_1, np.zeros(len(xn_0)), decimal=2)

    def test_soft_set_hard_get_1D(self):
        mock = ManualParameter('m', initial_value=1, unit='M', label='Mock')

        def mock_func(none):
            # to also test if the values are set correctly in the sweep
            arr = np.zeros([2, 2])
            arr[0, :] = np.array([mock()] * 2)
            arr[1, :] = np.array([mock() + 2] * 2)
            return arr

        d = DummyDetector(return_dimensions='2D')
        d.mock_fn = mock_func
        setpoints = np.repeat(np.arange(5.0), 2)

        self.MC.set_setpars(mock)
        self.MC.set_setpoints(setpoints)
        self.MC.set_getpars(d)
        dset = self.MC.run("soft_sweep_hard_det")

        x = dset['x0'].values
        y0 = dset['y0'].values
        y1 = dset['y1'].values
        np.testing.assert_array_equal(x, setpoints)
        np.testing.assert_array_equal(y0, setpoints)
        np.testing.assert_array_equal(y1, setpoints + 2)

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

    def test_hard_sweep_2D_grid(self):
        times = np.linspace(10, 20, 3)
        amps = np.linspace(0, 10, 5)

        self.MC.set_setpars([NoneSweep(internal=False), NoneSweep(internal=True)])
        self.MC.set_setpoints_grid([times, amps])
        self.MC.set_getpars(DummyDetector("2D"))
        dset = self.MC.run('2D Hard')

        exp_sp = tile_setpoints_grid([times, amps])
        assert np.array_equal(exp_sp, self.MC._setpoints)
        assert np.array_equal(dset['x0'].values, exp_sp[:, 0])
        assert np.array_equal(dset['x1'].values, exp_sp[:, 1])

        expected_vals = hardware_mock_values_2D(dset['x0'].values)
        assert np.array_equal(dset['y0'].values, expected_vals[0])
        assert np.array_equal(dset['y1'].values, expected_vals[1])

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)

        assert dset['x0'].attrs == {'name': 'none', 'long_name': 'None', 'unit': 'N'}
        assert dset['x1'].attrs == {'name': 'none', 'long_name': 'None', 'unit': 'N'}
        assert dset['y0'].attrs == {'name': 'dum', 'long_name': 'Watts', 'unit': 'W'}
        assert dset['y1'].attrs == {'name': 'mud', 'long_name': 'Matts', 'unit': 'M'}

    def test_hard_sweep_2D_arbitrary(self):
        r = np.linspace(0, 1.5, 5)
        dt = np.linspace(0, 1, 5)
        f = 10
        theta = np.cos(2*np.pi*f*dt)

        def polar_coords(r, theta):
            x = r*np.cos(2*np.pi*theta)
            y = r*np.sin(2*np.pi*theta)
            return x, y

        x, y = polar_coords(r, theta)
        setpoints = np.column_stack([x, y])

        self.MC.set_setpars([NoneSweep(internal=False), NoneSweep(internal=True)])
        self.MC.set_setpoints(setpoints)
        self.MC.set_getpars(DummyDetector("1D"))
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs['tuid'])

        expected_vals = hardware_mock_values_1D(x)

        assert np.array_equal(dset['x0'].values, x)
        assert np.array_equal(dset['x1'].values, y)
        assert np.array_equal(dset['y0'].values, expected_vals[0, :])

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
        d = DummyDetector('1D')
        d.mock_fn = v_size
        self.MC.set_setpars(NoneSweep(internal=False))
        self.MC.set_setpoints(setpoints)
        self.MC.set_getpars(d)
        dset = self.MC.run('varying')

        assert np.array_equal(dset['x0'], setpoints)
        assert np.array_equal(dset['y0'], 2 * setpoints)

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
