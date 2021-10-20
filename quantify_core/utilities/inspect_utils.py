# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Python inspect helper functions."""
import inspect
import sys
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from IPython.display import Code, display


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

    .. jupyter-execute::

        from quantify_core.utilities import inspect_utils
        import quantify_core.analysis.base_analysis as ba

        class_dict: Dict[str, type] = inspect_utils.get_classes(ba)
        print(class_dict)


    Parameters
    ----------
    modules
        Variable length of modules.

    Returns
    -------
    :
        A dictionary containing the class members of a module by name.
    """
    classes_list = []
    for module in modules:
        classes_list.extend(get_members_of_module(module, inspect.isclass))

    return dict(classes_list)


def get_functions(*modules: ModuleType) -> Dict[str, Callable]:
    """
    Returns a dictionary of functions by function names found in the modules provided
    in the 'modules' parameter.

    .. jupyter-execute::

        from quantify_core.utilities import inspect_utils

        function_dict: Dict[str, type] = inspect_utils.get_functions(inspect_utils)
        print(function_dict)

    Parameters
    ----------
    modules
        Variable length of modules.

    Returns
    -------
    :
        A dictionary containing the function members of a module by name.
    """
    function_list = []
    for module in modules:
        function_list.extend(get_members_of_module(module, inspect.isfunction))

    return dict(function_list)


def display_source_code(obj: object, exec_display: bool = True) -> Optional[Code]:
    """Displays the source code of a python object in a IPython kernel.

    Parameters
    ----------
    obj
        The python object for which to display the code.
    exec_display
        If ``True`` executes :func:`IPython.display.display` instead of returning an
        :class:`IPython.display.Code` object.

    Returns
    -------
    :
        The source code if ``exec_display==False``.
    """

    source_code: str = inspect.getsource(obj)
    code = Code(source_code, language="python")

    if not exec_display:
        return code

    return display(code)  # returns None
