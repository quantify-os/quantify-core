import xarray as xr
import numpy as np
from qcodes import ManualParameter, Parameter
from quantify.measurement.measurement_control import MeasurementControl


# Define some helpers that are used in the tests
def CosFunc(t, amplitude, frequency, phase):
    """A simple cosine function"""
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


# Parameters are created to emulate a system being measured
t = ManualParameter('t', initial_value=1, unit='s', label='Time')


def cosine_model():
    return CosFunc(t, amp=1, frequency=1, phase=0, offset=0)


# We wrap our function in a Parameter to be able to give
sig = Parameter(name='sig', label='Signal level',
                unit='V', get_cmd=cosine_model)


class TestMeasurementControl:

    @classmethod
    def setup_class(cls):
        cls.MC = MeasurementControl(name='MC')

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

    def test_initialize_dataset(self):

        xvals = np.linspace(0, 2*np.pi, 31)

        self.MC.set_setpars([t])
        self.MC.set_setpoints(xvals)
        self.MC.set_getpars(sig)

        self.MC._initialize_dataset()
        dset = self.MC._dataset
        assert isinstance(dset, xr.Dataset)
        assert dset.keys() == {'x0', 'y0'}
        assert (dset['x0'] == xvals).all()
        assert dset['x0'].attrs == {
            'name': 't', 'long_name': 'Time', 'unit': 's'}

        assert dset['y0'].attrs == {
            'name': 'sig', 'long_name': 'Signal level', 'unit': 'V'}
