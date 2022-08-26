# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""General utilities."""
import json
import pathlib
from collections.abc import MutableMapping
from typing import Any, Iterator, Optional, Type, TypeVar, Union

from qcodes.utils.helpers import NumpyJSONEncoder


def delete_keys_from_dict(dictionary: dict, keys: set) -> dict:
    """
    Delete keys from dictionary recursively.

    Parameters
    ----------
    dictionary
        to be mutated
    keys
        a set of keys to strip from the dictionary
    Returns
    -------
    :
        a new dictionary that does not included the blacklisted keys
    """
    keys_set = set(keys)  # optimization for the "if key in keys" lookup.

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, MutableMapping):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)
            else:
                modified_dict[key] = value
    return modified_dict


def traverse_dict(obj: dict, convert_to_string: bool = True):
    """
    Traversal implementation which recursively visits each node in a dict.
    We modify this function so that at the lowest hierarchy,
    we convert the element to a string.

    From: `<https://nvie.com/posts/modifying-deeply-nested-structures/>`_
    """
    if isinstance(obj, dict):
        out_dict = {}
        for key, val in obj.items():
            out_dict[key] = traverse_dict(val)
        return out_dict

    if isinstance(obj, list):
        return [traverse_dict(elem) for elem in obj]

    return_obj = str(obj) if convert_to_string else obj
    return str(return_obj)


def get_keys_containing(obj: Union[dict, Any], key) -> set:
    """
    Returns a set with the keys that contain `key`

    Example:

    .. code-block:: python

        from quantify_core.utilities.general import get_keys_containing
        dict_obj = {"x0": [1, 2, 3], "y0": [4, 5, 6], "other_key": 79}
        get_keys_containing(dict_obj, "x")

        # Return:
        # {"x0"}

    Parameters
    -----------
    obj
        any object with a `.keys()` attribute, usually a dictionary
    key
        the search key, usually a string
    Returns
    -------
    :
        a new set containing the keys that match the search
    """
    return set(filter(lambda k: key in k, obj.keys()))


def save_json(directory: pathlib.Path, filename: str, data) -> None:
    """
    Utility function to save serializable data to disk.

    Parameters
    ----------
    directory
        The directory where the data needs to be written to
    filename
        The filename of the data
    data
        The serializable data which needs to be saved to disk

    """
    full_path_to_file = directory / filename
    with open(full_path_to_file, "w", encoding="utf-8") as file:
        json.dump(data, file, cls=NumpyJSONEncoder, indent=4)


def load_json(full_path: pathlib.Path) -> dict:
    """
    Utility function to load from a json file to dict.

    Parameters
    ----------
    full_path
        The full path to the json file which needs to be loaded

    Raises
    ------
    FileNotFoundError
        If no file can be found at `full_path`
    """
    with open(full_path, encoding="utf-8") as file:
        return json.load(file)


def load_json_safe(full_path: pathlib.Path) -> Optional[dict]:
    """
    Utility function to load from a json file to dict.

    Parameters
    ----------
    full_path
        The full path to the json file which needs to be loaded

    Returns
    :
        Content of file. If file not found, returns ``None``.
    """
    try:
        return load_json(full_path=full_path)
    except FileNotFoundError:
        return None


def load_json_schema(relative_to: Union[str, pathlib.Path], filename: str):
    """
    Load a JSON schema from file. Expects a 'schemas' directory in the same directory
    as `relative_to`.

    .. tip::

        Typical usage of the form
        `schema = load_json_schema(__file__, 'definition.json')`

    Parameters
    ----------
    relative_to
        the file to begin searching from
    filename
        the JSON file to load
    Returns
    -------
    dict
        the schema
    """
    path = pathlib.Path(relative_to).resolve().parent.joinpath("schemas", filename)
    with path.open(mode="r", encoding="utf-8") as file:
        return json.load(file)


def without(dict_in: dict, keys: list):
    """
    Utility that copies a dictionary excluding a specific list of keys.
    """
    if not isinstance(keys, list):
        keys = [keys]
    new_d = dict_in.copy()
    for key in keys:
        new_d.pop(key)
    return new_d


def call_if_has_method(obj: Any, method: str) -> None:
    """
    Calls the `method` of the `obj` if it has it
    """
    prepare_method = getattr(obj, method, lambda: None)
    prepare_method()


def last_modified(path: pathlib.Path) -> float:
    """Returns the timestamp of the last modification of a file.

    Parameters
    ----------
    path
        File path.
    """
    path = pathlib.Path(path)

    return path.stat().st_mtime


T = TypeVar("T")


def get_subclasses(base: Type[T], include_base: bool = False) -> Iterator[Type[T]]:
    """
    Obtain all subclasses of a class.
    From: `<https://stackoverflow.com/a/33607093>`_.

    Parameters
    ----------
    base
        base class for which subclasses will be returned.
    include_base
        include the base class in the iterator.

    Yields
    ------
    subclass : Type[T]
        Next subclass for a class.
    base : Type[T]
        Optionally, base class itself included in the iterator.
    """
    for subclass in base.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass

    if include_base:
        yield base
