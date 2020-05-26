import pytest
from qcodes import ManualParameter, Parameter
from quantify.measurement.types import Settable, Gettable, Datasource


def test_settable():
    x = 5
    with pytest.raises(AttributeError):
        Settable(x, Datasource.INTERNAL)

    def test_func(x):
        return 5
    with pytest.raises(AttributeError):
        Settable(test_func, Datasource.INTERNAL)

    manpar = ManualParameter('x')
    Settable(manpar, Datasource.INTERNAL)

    del manpar.unit
    with pytest.raises(AttributeError, match="x does not have 'unit'"):
        Settable(manpar, Datasource.INTERNAL)

    std_par = Parameter('x', set_cmd=test_func)
    Settable(std_par, Datasource.INTERNAL)

    class NoName:
        def set(self):
            return
        unit = 'no_name'

    with pytest.raises(AttributeError, match=r".*does not have 'name'"):
        Settable(NoName(), Datasource.INTERNAL)


def test_gettable():
    x = 5
    with pytest.raises(AttributeError):
        Gettable(x, Datasource.INTERNAL)

    def test_func():
        return 5
    with pytest.raises(AttributeError):
        Gettable(test_func, Datasource.INTERNAL)

    manpar = ManualParameter('x')
    Gettable(manpar, Datasource.INTERNAL)

    del manpar.unit
    with pytest.raises(AttributeError, match="x does not have 'unit'"):
        Gettable(manpar, Datasource.INTERNAL)

    std_par = Parameter('x', get_cmd=test_func)
    Gettable(std_par, Datasource.INTERNAL)
