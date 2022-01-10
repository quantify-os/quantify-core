# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""General utilities."""
from __future__ import annotations

import copy
import importlib
import json
import pathlib
import warnings
from collections.abc import MutableMapping
from typing import Any, Union

import numpy as np
import xxhash

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

    From https://nvie.com/posts/modifying-deeply-nested-structures/
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


def make_hash(obj: Any):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    From: https://stackoverflow.com/questions/5884066/hashing-a-dictionary
    """

    new_hash = xxhash.xxh64()
    if isinstance(obj, (set, tuple, list)):
        return tuple(make_hash(e) for e in obj)

    if isinstance(obj, np.ndarray):
        # numpy arrays behave funny for hashing
        new_hash.update(obj)
        val = new_hash.intdigest()
        new_hash.reset()
        return val

    if not isinstance(obj, dict):
        return hash(obj)

    new_o = copy.deepcopy(obj)
    for key, val in new_o.items():
        new_o[key] = make_hash(val)

    return hash(tuple(frozenset(sorted(new_o.items()))))


def import_func_from_string(function_string: str) -> Any:
    """A deprecated alias for :func:`~.import_python_object_from_string`."""
    warnings.warn(
        "This functions is deprecated. Use `import_python_object_from_string` instead.",
        category=DeprecationWarning,
    )
    return import_python_object_from_string(function_string)


def import_python_object_from_string(function_string: str) -> Any:
    """
    Based on https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
    """  # pylint: disable=line-too-long
    mod_name, func_name = function_string.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


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

    """
    with open(full_path, encoding="utf-8") as file:
        return json.load(file)


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
