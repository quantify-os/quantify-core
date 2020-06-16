import pytest
from jsonschema import ValidationError
from qcodes import ManualParameter, Parameter
from quantify.measurement.types import Settable, Gettable


def test_settable():
    x = 5
    with pytest.raises(TypeError):
        Settable(x)

    def test_func(x):
        return 5
    with pytest.raises(ValidationError):
        Settable(test_func)

    manpar = ManualParameter('x')
    Settable(manpar)

    del manpar.unit
    with pytest.raises(ValidationError):
        Settable(manpar)

    std_par = Parameter('x', set_cmd=test_func)
    Settable(std_par)

    class NoName:
        def set(self):
            return
        unit = 'no_name'

    with pytest.raises(ValidationError):
        Settable(NoName())


def test_gettable():
    x = 5
    with pytest.raises(TypeError):
        Gettable(x)

    def test_func():
        return 5
    with pytest.raises(ValidationError):
        Gettable(test_func)

    manpar = ManualParameter('x')
    Gettable(manpar)

    del manpar.unit
    with pytest.raises(ValidationError):
        Gettable(manpar)

    std_par = Parameter('x', get_cmd=test_func)
    Gettable(std_par)
