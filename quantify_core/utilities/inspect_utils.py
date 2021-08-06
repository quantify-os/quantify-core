# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Python inspect helper functions."""
import inspect
import sys
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Tuple, Type, Union


def get_members_of_module(
    module: ModuleType, predicate: Callable[[Any], bool]
) -> List[Tuple[str, Union[type, FunctionType]]]:
    """
    Returns all members of a module that match the predicate function.

    Parameters
    ----------
    module :
        The module to inspect.
    predicate :
        The predicate function.

    Returns
    -------
    :
        The list of matching members of a module.
    """
    module_name: str = module.__name__
    members = inspect.getmembers(
        sys.modules[module_name],
        lambda member: predicate(member)
        and member.__module__ == module_name,  # pylint: disable=cell-var-from-loop
    )
    return members


def get_classes(*modules: ModuleType) -> Dict[str, Type[Any]]:
    """
    Returns a dictionary of class types by class names found in the modules provided
    in the 'modules' parameter.

    .. code-block::

        from quantify_core.utilities import inspect_utils
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
        A dictionary containing the class members of a module by name.
    """
    classes_list = list()
    for module in modules:
        classes_list.extend(get_members_of_module(module, inspect.isclass))

    return dict(classes_list)


def get_functions(*modules: ModuleType) -> Dict[str, Callable]:
    """
    Returns a dictionary of functions by function names found in the modules provided
    in the 'modules' parameter.

    .. code-block::

        from quantify_core.utilities import inspect_utils
        from my_module import foo

        function_dict: Dict[str, type] = inspect_utils.get_functions(foo)
        print(function_dict)
        // { 'get_name': my_module.foo.Bar.get_name }

    Parameters
    ----------
    modules :
        Variable length of modules.

    Returns
    -------
    :
        A dictionary containing the function members of a module by name.
    """
    function_list = list()
    for module in modules:
        function_list.extend(get_members_of_module(module, inspect.isfunction))

    return dict(function_list)
