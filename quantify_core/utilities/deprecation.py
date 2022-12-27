# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Utilities used to maintain deprecation and reverse-compatibility of the code."""

import functools
import inspect
import os
import warnings
from typing import Callable, Type, Union


def _find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside quantify-core
    (tests notwithstanding).

    (adopted from pandas.util._exceptions.find_stack_level)
    """
    # pylint: disable=import-outside-toplevel,invalid-name

    import quantify_core

    pkg_dir = os.path.dirname(quantify_core.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir) and not fname.startswith(test_dir):
            frame = frame.f_back
            n += 1
        else:
            break
    return n


def deprecated(
    drop_version: str, message_or_alias: Union[str, Callable]
) -> Callable[[Callable], Callable]:
    """A decorator for deprecating classes and methods.

    For each deprecation we must provide a version when this function or class will be
    removed completely and an instruction to a user about how to port their existing
    code to a new software version. This is easily done using this decorator.

    If callable is passed instead of a message, this decorator assumes that
    the function or class has moved to another module and generates the standard
    instruction to use a new function or class. There is no need to re-implement
    the function logic in two places, since the implementation of new function or class
    is used in both new and old aliases.

    .. admonition:: Example
        :class: dropdown, tip

        .. include:: examples/utilities.deprecated.rst.txt

    Parameters
    ----------
    drop_version
        A version of the package when the deprecated function or class will be dropped.
    message_or_alias
        Either an instruction about how to port the software to a new version without
        the usage of deprecated calls (string), or the new drop-in replacement to the
        deprecated class or function (callable).
    """

    def deprecator(func_or_class: Union[Callable, Type]) -> Union[Callable, Type]:
        old_module = inspect.getmodule(func_or_class)
        if old_module is None:
            raise RuntimeError("Could not determine module of the deprecated object.")
        # We assume that the package name and the first part of module name are
        # the same, but package name has - instead of _, for example if root module name
        # is "quantify_core", package name is assumed to be "quantify-core".
        package = old_module.__name__.split(".", 1)[0].replace("_", "-")

        maybe_brackets = "" if isinstance(func_or_class, type) else "()"

        if callable(message_or_alias):
            new_module = inspect.getmodule(message_or_alias)
            if new_module is None:
                raise RuntimeError("Could not determine module of the moved object.")
            instruction = (
                f"Use {new_module.__name__}.{message_or_alias.__qualname__}"
                f"{maybe_brackets} instead."
            )
        else:
            instruction = message_or_alias

        message = (
            f"{'Class' if isinstance(func_or_class, type) else 'Function'} "
            f"{old_module.__name__}.{func_or_class.__qualname__}{maybe_brackets} is "
            f"deprecated and will be removed in {package}-{drop_version}. {instruction}"
        )

        if isinstance(func_or_class, type):
            if isinstance(message_or_alias, type):
                metaclass = type(message_or_alias)
                cls = metaclass(
                    func_or_class.__name__,
                    message_or_alias.__bases__,
                    dict(message_or_alias.__dict__),
                )
            else:
                cls = func_or_class

            orig_init = cls.__init__  # type: ignore

            @functools.wraps(orig_init)
            def __init__(self, *args, **kwargs):
                warnings.warn(message, FutureWarning, stacklevel=_find_stack_level())
                orig_init(self, *args, **kwargs)

            # Here we patch only __init__ method. For completeness, we should also patch
            # all the staticmethods and classmethods, but let it be left for the future,
            # if someone really needs it implemented.
            cls.__init__ = __init__  # type: ignore
            return cls

        if callable(message_or_alias):
            func = message_or_alias
        else:
            func = func_or_class

        # pylint: disable=too-few-public-methods
        class _FnDeprecator:
            def __init__(self):
                functools.update_wrapper(self, func)

            def __set_name__(self, owner, name):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    return self(*args, **kwargs)

                setattr(owner, name, wrapper)

            def __call__(self, *args, **kwargs):
                warnings.warn(message, FutureWarning, stacklevel=_find_stack_level())
                return func(*args, **kwargs)

        return _FnDeprecator()

    return deprecator
