from collections.abc import MutableMapping


def delete_keys_from_dict(dictionary: dict, keys: set):
    """
    Delete keys from dictionary recursively.

    Args:
        dictionary (dict)
        keys (set)  a set of keys to strip from the dictionary.

    Return:
        modified_dict (dict) a new dictionary that does not included the
        blacklisted keys.

    function based on "https://stackoverflow.com/questions/3405715/
    elegant-way-to-remove-fields-from-nested-dictionaries"
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
