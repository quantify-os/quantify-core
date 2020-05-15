import pytest
from qcodes import ManualParameter, Parameter
from quantify.measurement.core import Settable, Gettable


def test_settable():
    x = 5
    with pytest.raises(AttributeError):
        Settable(x)

    def test_func(x):
        return 5
    with pytest.raises(AttributeError):
        Settable(test_func)

    manpar = ManualParameter('x')
    Settable(manpar)

    del manpar.unit
    with pytest.raises(AttributeError, match="x does not have 'unit'"):
        Settable(manpar)

    std_par = Parameter('x', set_cmd=test_func)
    Settable(std_par)

    class NoName:
        def set(self):
            return
        unit = 'no_name'

    with pytest.raises(AttributeError, match=r".*does not have 'name'"):
        Settable(NoName())


def test_gettable():
    x = 5
    with pytest.raises(AttributeError):
        Gettable(x)

    def test_func():
        return 5
    with pytest.raises(AttributeError):
        Gettable(test_func)

    manpar = ManualParameter('x')
    Gettable(manpar)

    del manpar.unit
    with pytest.raises(AttributeError, match="x does not have 'unit'"):
        Gettable(manpar)

    std_par = Parameter('x', get_cmd=test_func)
    Gettable(std_par)
