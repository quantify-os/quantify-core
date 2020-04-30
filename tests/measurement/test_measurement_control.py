import pytest
import xarray as xr
import numpy as np
from qcodes import ManualParameter, Parameter
from quantify.measurement.measurement_control import MeasurementControl, \
    is_setable, is_getable
from quantify import set_datadir

# Define some helpers that are used in the tests


def CosFunc(t, amplitude, frequency, phase):
    """A simple cosine function"""
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


# Parameters are created to emulate a system being measured
t = ManualParameter('t', initial_value=1, unit='s', label='Time')


def cosine_model():
    return CosFunc(t(), amplitude=1, frequency=1, phase=0)


# We wrap our function in a Parameter to be able to give
sig = Parameter(name='sig', label='Signal level',
                unit='V', get_cmd=cosine_model)


class TestMeasurementControl:

    @classmethod
    def setup_class(cls):
        cls.MC = MeasurementControl(name='MC')
        # ensures the default datadir is used which is excluded from git
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


def test_is_setable():
    x = 5
    with pytest.raises(AttributeError):
        is_setable(x)

    def test_func(x):
        return 5
    with pytest.raises(AttributeError):
        is_setable(test_func)

    manpar = ManualParameter('x')
    assert is_setable(manpar)

    del manpar.unit
    with pytest.raises(AttributeError, match="does not have 'unit'"):
        is_setable(manpar)

    std_par = Parameter('x', set_cmd=test_func)
    assert is_setable(std_par)

    # Add test with another object that has no name attribute.
    # because removing it from a parameter breaks the object.
    # del std_par.name
    # with pytest.raises(AttributeError, match="does not have 'name'"):
    #     is_setable(std_par)


def test_is_getable():
    x = 5
    with pytest.raises(AttributeError):
        is_getable(x)

    def test_func():
        return 5
    with pytest.raises(AttributeError):
        is_getable(test_func)

    manpar = ManualParameter('x')
    assert is_getable(manpar)

    del manpar.unit
    with pytest.raises(AttributeError, match="does not have 'unit'"):
        is_getable(manpar)

    std_par = Parameter('x', get_cmd=test_func)
    assert is_getable(std_par)
