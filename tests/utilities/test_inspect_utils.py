"""Unit tests for inspect_utils module."""
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from quantify_core.utilities import inspect_utils


def test_get_classes() -> None:
    # Arrange
    from quantify_core.data import types  # pylint: disable=import-outside-toplevel

    # Act
    classes = inspect_utils.get_classes(types)

    # Assert
    assert "TUID" in classes
    assert isinstance(classes["TUID"], type)


def test_get_functions() -> None:
    # Arrange
    expected = {
        "get_members_of_module": inspect_utils.get_members_of_module,
        "get_classes": inspect_utils.get_classes,
        "get_functions": inspect_utils.get_functions,
        "display_source_code": inspect_utils.display_source_code,
    }

    # Act
    functions = inspect_utils.get_functions(inspect_utils)

    # Assert
    assert functions == expected
