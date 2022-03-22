# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=unused-argument
# pylint: disable=no-self-use

from typing import Any

import pytest
from jsonschema import ValidationError
from qcodes import ManualParameter, Parameter

from quantify_core.measurement.types import Gettable, Settable


def test_settable() -> None:
    x = 5
    with pytest.raises(ValidationError):
        Settable(x)

    def test_func(x: Any) -> int:
        return 5

    with pytest.raises(ValidationError):
        Settable(test_func)

    manpar = ManualParameter("x")
    Settable(manpar)

    manpar.unit = None
    with pytest.raises(ValidationError):
        Settable(manpar)

    std_par = Parameter("x", set_cmd=test_func)
    Settable(std_par)

    class NoName:
        def set(self) -> None:
            return

        unit = "no_name"

    with pytest.raises(ValidationError):
        Settable(NoName())


def test_attrs_as_property_decorator() -> None:
    class SettableWithProperties:
        def __init__(self) -> None:
            self.name = "bla"
            self._label = "x"
            self._unit = "V"

        @property
        def label(self) -> str:
            return self._label

        @property
        def batched(self) -> bool:
            return False

        @property
        def unit(self) -> str:
            return self._unit

        def set(self, value: Any) -> None:
            pass

    Settable(SettableWithProperties())


def test_gettable() -> None:
    x = 5
    with pytest.raises(ValidationError):
        Gettable(x)

    def test_func() -> int:
        return 5

    with pytest.raises(ValidationError):
        Gettable(test_func)

    manpar = ManualParameter("x")
    Gettable(manpar)

    manpar.unit = None
    with pytest.raises(ValidationError):
        Gettable(manpar)

    std_par = Parameter("x", get_cmd=test_func)
    Gettable(std_par)
