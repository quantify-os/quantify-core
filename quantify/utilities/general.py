import json
import pathlib
from collections.abc import MutableMapping


def delete_keys_from_dict(dictionary: dict, keys: set):
    """
    Delete keys from dictionary recursively.

    Args:
        dictionary (dict)
        keys (set)  a set of keys to strip from the dictionary.

    Return:
        dict: a new dictionary that does not included the blacklisted keys.

    function based on "https://stackoverflow.com/questions/3405715/"
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


def load_json_schema(relative_to, filename):
    """
    Load a JSON schema from file. Expects a 'schemas' directory in the same directory as `relative_to`.

    .. tip:: Typical usage of the form `schema = load_json_schema(__file__, 'definition.json')`

    Args:
        relative_to (str): the file to begin searching from
        filename (str): the JSON file to load

    Returns:
        dict: the schema
    """
    path = pathlib.Path(relative_to).resolve().parent.joinpath('schemas', filename)
    with path.open(mode='r') as f:
        return json.load(f)


class KeyboardFinish(KeyboardInterrupt):
    """
    Indicates the user has signalled to safely abort/finish the experiment.
    """
    pass
