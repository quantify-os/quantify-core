# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Python inspect helper functions."""
from types import ModuleType
import inspect
import sys
from typing import Any, Dict, Type


def get_classes(*modules: ModuleType) -> Dict[str, Type[Any]]:
    """
    Returns a dictionary of class names by class types of the
    modules given as arguments.

    .. code-block::

        from quantify.utilities import inspect_utils
        from my_module import foo

        class_dict: Dict[str, type] = inspect_utils.get_classes(foo)
        print(class_dict)
        // { 'Bar': my_module.foo.Bar }

    Parameters
    ----------
    modules :
        Variable length of modules.

    Returns
    -------
    :
        A dictionary containing the class names by class reference.
    """
    classes = list()
    for module in modules:
        module_name: str = module.__name__
        classes += inspect.getmembers(
            sys.modules[module_name],
            lambda member: inspect.isclass(member)
            and member.__module__ == module_name,  # pylint: disable=cell-var-from-loop
        )
    return dict(classes)
