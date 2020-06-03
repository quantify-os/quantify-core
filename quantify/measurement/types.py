import jsonschema
import json
import pathlib


def _load_schema(filename):
    path = pathlib.Path(__file__).resolve().parent.joinpath('schemas', filename)
    with path.open(mode='r') as f:
        return json.load(f)


class Settable:
    schema = _load_schema('Settable.json')

    """
    Defines the Settable concept, which is considered complete if the given type satisfies the following:

    .. jsonschema:: schemas/Settable.json

    """
    def __new__(cls, obj):
        jsonschema.validate(vars(obj), Settable.schema['required_attributes'])
        jsonschema.validate(dir(obj), Settable.schema['required_methods'])
        return obj


class Gettable:
    schema = _load_schema('Gettable.json')

    """
    Defines the Gettable concept, which is considered complete if the given type satisfies the following:

    .. jsonschema:: schemas/Gettable.json

    """
    def __new__(cls, obj):
        jsonschema.validate(vars(obj), Gettable.schema['required_attributes'])
        jsonschema.validate(dir(obj), Gettable.schema['required_methods'])
        return obj


def is_internally_controlled(obj):
    if hasattr(obj, 'internal'):
        return obj.internal
    return True
