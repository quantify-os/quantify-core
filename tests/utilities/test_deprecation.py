# pylint: disable=anomalous-backslash-in-string
# pylint: disable=disallowed-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
import warnings

import pytest

from quantify_core.utilities.deprecation import deprecated


def test_deprecated_message():
    @deprecated("1.0", "Can't you just type foo?")
    def give_me_foo():
        return "foo"

    with pytest.warns(
        FutureWarning,
        match=r"Function .*give_me_foo\(\) is deprecated and will be removed .*-1.0. "
        "Can't you just type foo?",
    ):
        foo = give_me_foo()
    assert foo == "foo"

    class SomeClass:
        attr = "Blah"

        def __init__(self):
            self.foo = "foo"

        @deprecated("1.1", "Use SomeClass.foo directly.")
        def get_foo(self):
            return self.foo

    new_class_instance = SomeClass()

    with pytest.warns(
        FutureWarning,
        match=r"Function .*SomeClass.get_foo\(\) is deprecated and will be removed "
        "in .*-1.1. Use SomeClass.foo directly.",
    ):
        foo = new_class_instance.get_foo()

    assert foo == "foo"

    @deprecated("0.9", "Read our awesome new documentation!")
    class OldClass:
        def __init__(self, foo):
            self.foo = foo

    with pytest.warns(
        FutureWarning,
        match="Class .*OldClass is deprecated and will be removed "
        "in .*-0.9. Read our awesome new documentation!",
    ):
        old_class_instance = OldClass("foo")

    assert old_class_instance.foo == "foo"

    # This is actually very important: we don't want OldClass to suddenly become
    # a function or something like that
    assert isinstance(old_class_instance, OldClass)
    assert OldClass.__name__ == "OldClass"


def test_deprecated_alias():
    def new_function(x: int) -> int:
        return x + 5

    @deprecated("0.7", new_function)
    def old_function(_):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new_result = new_function(37)

    with pytest.warns(
        FutureWarning,
        match=r"Function .*old_function\(\) is deprecated and will be removed "
        "in .*-0.7. Use .*new_function\(\) instead.",
    ):
        old_result = old_function(37)

    assert new_result == old_result

    class NewClass:
        def __init__(self, val) -> None:
            self._val = val

        def val(self):
            return self._val

        @deprecated("0.7", val)
        def get_val(self):
            """Deprecated alias"""

    @deprecated("1.1", NewClass)
    class OldClass:
        pass

    val = "42"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new_instance = NewClass(val)

    with pytest.warns(
        FutureWarning,
        match="Class .*OldClass is deprecated and will be removed in .*-1.1. "
        "Use .*NewClass instead.",
    ):
        old_instance = OldClass(val)

    assert isinstance(old_instance, OldClass)
    assert isinstance(new_instance, NewClass)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert new_instance.val() == old_instance.val()

    with pytest.warns(
        FutureWarning,
        match=r"Function .*NewClass.get_val\(\) is deprecated and will be removed in "
        ".*-0.7. Use .*NewClass.val\(\) instead.",
    ):
        new_val = new_instance.get_val()

    # This will fail, and print NewClass.get_val() instead of OldClass.get_val().
    # I don't think it is a huge issue, because fixing this will take really a lot
    # of time ¯\_(ツ)_/¯
    # This is an edge case: deprecated method of a deprecated class will refer to
    # a new class. Preferably it should be fixed and message should look like:
    #
    #   "Function .*OldClass.get_val\(\) is deprecated and will be removed in "
    #   ".*-0.7. Use .*OldClass.val\(\) instead."
    #
    # Hopefully this will not confuse anyone because of its rareness ;)
    with pytest.warns(
        FutureWarning,
        match=r"Function .*NewClass.get_val\(\) is deprecated and will be removed in "
        ".*-0.7. Use .*NewClass.val\(\) instead.",
    ):
        old_val = old_instance.get_val()

    assert new_val == old_val
