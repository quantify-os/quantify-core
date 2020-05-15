import pytest
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
amp = ManualParameter('amp', initial_value=1, unit='V', label='Time')

def cosine_model():
    return CosFunc(t(), amplitude=amp(), frequency=1, phase=0)


# We wrap our function in a Parameter to be able to give
sig = Parameter(name='sig', label='Signal level',
                unit='V', get_cmd=cosine_model)


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
        self.MC._setpoints[:, 0] == x

        x = np.linspace(0, 10, 11)

        x = np.random.rand(15, 2)
        self.MC.set_setpoints(x)
        self.MC._setpoints == x

        x = np.random.rand(15, 4)
        self.MC.set_setpoints(x)
        self.MC._setpoints == x

    def test_soft_sweep_1D(self):

        xvals = np.linspace(0, 2*np.pi, 31)

        self.MC.set_setpars(t)
        self.MC.set_setpoints(xvals)
        self.MC.set_getpars(sig)
        dset = self.MC.run()

        assert TUID.is_valid(dset.attrs['tuid'])

        expected_vals = CosFunc(
            t=xvals, amplitude=1, frequency=1, phase=0)

        assert (dset['x0'].values == xvals).all()
        assert (dset['y0'].values == expected_vals).all()

        # Test properties of the dataset
        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {'x0', 'y0'}
        assert (dset['x0'] == xvals).all()
        assert dset['x0'].attrs == {
            'name': 't', 'long_name': 'Time', 'unit': 's'}

        assert dset['y0'].attrs == {
            'name': 'sig', 'long_name': 'Signal level', 'unit': 'V'}

        # Test values

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
        dset = self.MC.run()

        assert progress_param() == 100


    def test_MC_plotmon_integration(self):
        plotmon = PlotMonitor_pyqt('plotmon_MC')

        self.MC.instr_plotmon(plotmon.name)

        assert plotmon.tuid() == 'latest'

        times = np.linspace(0, 5, 18)
        amps = np.linspace(-1, 1, 5)

        self.MC.set_setpars([t, amp])
        self.MC.set_setpoints(times)
        self.MC.set_setpoints_2D(amps)
        self.MC.set_getpars(sig)
        dset = self.MC.run('2D Cosine test')

        # Test that the
        assert plotmon.tuid() != 'latest'

        plotmon.close()
        self.MC.instr_plotmon('')


def test_tile_setpoints_grid():
    x = np.arange(5)
    x = x.reshape((len(x), 1))
    y = np.linspace(-1, 1, 3)

    sp = tile_setpoints_grid(x, y)
    sp[:,0] = np.tile(np.arange(5), 3)
    sp[:,1] = np.repeat(y, 5)
