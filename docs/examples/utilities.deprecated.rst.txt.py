# ---
# jupyter:
#   jupytext:
#     cell_markers: \"\"\"
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %%
rst_conf = {"jupyter_execute_options": [":hide-code:"]}

"""Examples for using quantify_core.utilities.deprecated() decorator."""

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=wrong-import-position
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods

import warnings
from quantify_core.utilities import deprecated

# %%
@deprecated("99.99", 'Initialize the "foo" literal directly.')
def get_foo():
    return "foo"


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    get_foo()  # issues deprecation warning.

assert len(w) == 1
assert w[0].category is DeprecationWarning
print(w[0].message)

# %%


class NewClass:
    """A very useful class"""

    def __init__(self, val):
        self._val = val

    def val(self):
        return self._val


@deprecated("99.99", NewClass)
class OldClass:
    pass


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = OldClass(42)  # type: ignore

assert len(w) == 1
assert w[0].category is DeprecationWarning
print(w[0].message)
print("obj.val() =", obj.val())  # type: ignore

# %%


class SomeClass:
    """A very useful class"""

    def __init__(self, val):
        self._val = val

    def val(self):
        return self._val

    @deprecated("7.77", val)
    def get_val(self):
        """Deprecated alias"""


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    val = SomeClass(42).get_val()  # issues deprecation warning.

assert len(w) == 1
assert w[0].category is DeprecationWarning
print(w[0].message)
print("obj.get_val() =", val)
