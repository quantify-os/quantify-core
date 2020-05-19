import copy
import xxhash
import numpy as np
from collections.abc import MutableMapping


def delete_keys_from_dict(dictionary: dict, keys: set):
    """
    Delete keys from dictionary recursively.

    Args:
        dictionary (dict)
        keys (set)  a set of keys to strip from the dictionary.

    Return:
        modified_dict (dict) a new dictionary that does not included the blacklisted keys.

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


def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    from: https://stackoverflow.com/questions/5884066/hashing-a-dictionary
    """

    h = xxhash.xxh64()
    if isinstance(o, (set, tuple, list)):

        return tuple([make_hash(e) for e in o])

    elif isinstance(o, np.ndarray):
        # numpy arrays behave funny for hashing
        h.update(o)
        val = h.intdigest()
        h.reset()
        return val

    elif not isinstance(o, dict):
        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))
