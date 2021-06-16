"""Unit tests for inspect_utils module."""
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from quantify.utilities import inspect_utils


def test_get_classes():
    # Arrange
    from quantify.data import types  # pylint: disable=import-outside-toplevel

    # Act
    classes = inspect_utils.get_classes(types)

    # Assert
    assert "TUID" in classes
    assert isinstance(classes["TUID"], type)
