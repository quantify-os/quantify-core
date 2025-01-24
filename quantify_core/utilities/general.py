# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""General utilities."""

import bz2
import gzip
import json
import lzma
import pathlib
from collections.abc import MutableMapping
from typing import Any, Iterator, Literal, Optional, Type, TypeVar, Union

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


def save_json(
    directory: pathlib.Path,
    filename: str,
    data: Any,
    compression: Literal[None, "bz2", "gzip", "lzma"] = None,
) -> None:
    """
    Utility function to save serializable data to disk, optionally with compression.

    Parameters
    ----------
    directory
        The directory where the data needs to be written to
    filename
        The filename of the data, excluding the compression extension
    data
        The serializable data which needs to be saved to disk
    compression
        The compression type to use. Can be one of 'bz2', 'gzip', 'lzma'
        Defaults to None, which means no compression

    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    ValueError
        If the filename is not a string
    ValueError
        If the compression type is not supported
    """
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")

    if compression and compression not in ("bz2", "gzip", "lzma"):
        raise ValueError(
            f"Unsupported compression type '{compression}'. "
            "Supported types are 'bz2', 'gzip', 'lzma'."
        )

    full_path = directory / filename

    if compression:
        compression_openers = {
            "bz2": bz2.open,
            "gzip": gzip.open,
            "lzma": lzma.open,
        }

        compression_suffixes = {
            "bz2": ".bz2",
            "gzip": ".gz",
            "lzma": ".xz",
        }

        with compression_openers[compression](
            str(full_path) + compression_suffixes[compression], "wt"
        ) as file:
            json.dump(data, file, cls=NumpyJSONEncoder, indent=4)
    else:
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(data, file, cls=NumpyJSONEncoder, indent=4)


def load_json(full_path: pathlib.Path) -> dict[str, Any]:
    """
    Load JSON data from a file with automatic compression detection.

    Parameters
    ----------
    full_path : pathlib.Path
        The full path to the json file

    Returns
    -------
    dict[str, Any]
        The loaded JSON data

    Raises
    ------
    FileNotFoundError
        If file not found in any supported format
    json.JSONDecodeError
        If JSON is invalid
    ValueError
        If compressed file is corrupted
    OSError
        If error reading file
    """

    def read_json(file_path: pathlib.Path, opener) -> Any:
        """Helper function to read JSON with given file opener.

        Parameters
        ----------
        file_path : pathlib.Path
            Path to the file to read
        opener : callable
            File opener function to use (open, gzip.open, etc.)

        Returns
        -------
        Any
            Decoded JSON data

        Raises
        ------
        json.JSONDecodeError, OSError, ValueError
            If reading or decoding fails
        """
        try:
            with opener(file_path, "rt", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(e.msg, e.doc, e.pos) from None
        except (OSError, ValueError) as e:
            raise type(e)(str(e)) from e

    full_path = pathlib.Path(full_path)
    compression_handlers = {
        ".bz2": bz2.open,
        ".gz": gzip.open,
        ".xz": lzma.open,
    }

    for suffix, opener in compression_handlers.items():
        if str(full_path).endswith(suffix):
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {full_path}")
            return read_json(full_path, opener)

    if full_path.exists():
        return read_json(full_path, open)

    for suffix, opener in compression_handlers.items():
        compressed_path = pathlib.Path(str(full_path) + suffix)
        if compressed_path.exists():
            return read_json(compressed_path, opener)

    raise FileNotFoundError(
        f"File not found in any format:\n"
        f"  {full_path}\n"
        + "\n".join(f"  {str(full_path)}{suffix}" for suffix in compression_handlers)
    )


def load_json_safe(full_path: pathlib.Path) -> Optional[dict[str, Any]]:
    """
    Safely loads JSON file contents, returns None if loading fails.

    Parameters
    ----------
    full_path : pathlib.Path
        The full path to the JSON file

    Returns
    -------
    Optional[dict[str, Any]]
        The loaded JSON data or None if loading fails

    Notes
    -----
    Catches TypeError, FileNotFoundError, json.JSONDecodeError, and OSError
    silently and returns None instead of raising exceptions.
    """
    try:
        return load_json(full_path=full_path)
    except (TypeError, FileNotFoundError, json.JSONDecodeError, OSError):
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
        new_d.pop(key, None)
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
