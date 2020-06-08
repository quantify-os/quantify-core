import jsonschema
import json
import pathlib


def _load_schema(filename):
    path = pathlib.Path(__file__).resolve().parent.joinpath('schemas', filename)
    with path.open(mode='r') as f:
        return json.load(f)


class Settable:
    """
    Defines the Settable concept, which is considered complete if the given type satisfies the following:
    This object does not wrap the passed in object but simply verifies and returns it.

    .. jsonschema:: schemas/Settable.json#/attrs
    .. jsonschema:: schemas/Settable.json#/methods
    """

    schema = _load_schema('Settable.json')

    def __new__(cls, obj):
        jsonschema.validate(vars(obj), Settable.schema['attrs'])
        jsonschema.validate(dir(obj), Settable.schema['methods'])
        return obj


class Gettable:
    """
    Defines the Gettable concept, which is considered complete if the given type satisfies the following:
    This object does not wrap the passed in object but simply verifies and returns it.

    .. jsonschema:: schemas/Gettable.json#/attrs
    .. jsonschema:: schemas/Gettable.json#/methods

    """

    schema = _load_schema('Gettable.json')

    def __new__(cls, obj):
        jsonschema.validate(vars(obj), Gettable.schema['attrs'])
        jsonschema.validate(dir(obj), Gettable.schema['methods'])
        return obj


def is_software_controlled(obj):
    if hasattr(obj, 'soft'):
        return obj.soft
    return True
