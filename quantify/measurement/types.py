import jsonschema


class Settable:
    attr_schema = {
        "required": [
            "name",
            "label",
            "unit",
        ],
        "properties": {
            "name": {"type": "string"},
            "label": {"type": "string"},
            "unit": {"type": "string"},
        },
    }

    method_schema = {
        "required": [
            "set",
        ],
    }

    """
    Defines the Settable concept, which is considered complete if the given type satisfies the following:

    contains attributes
        - set(float)
        - name: str
        - label: str
        - unit: str

    optional attributes
        - internal (str): whether this parameter is internally or externally driven
        - prepare():
        - finish():
    """
    def __new__(cls, obj):
        jsonschema.validate(vars(obj), Settable.attr_schema)
        jsonschema.validate(dir(obj), Settable.method_schema)
        return obj


class Gettable:
    attr_schema = {
        "required": [
            "name",
            "label",
            "unit",
        ],
        "properties": {
            "name": {"oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]},
            "label": {"oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]},
            "unit": {"oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]}
        },
    }

    method_schema = {
        "required": [
            "get",
        ],
    }

    """
    Defines the Gettable concept, which is considered complete if the given type satisfies the following:

    contains attributes
        - get()
        - name: str
        - label: str
        - unit: str

    optional attributes
        - internal (str): whether this parameter is internally or externally driven
        - prepare():
        - finish():
    """
    def __new__(cls, obj):
        jsonschema.validate(vars(obj), Gettable.attr_schema)
        jsonschema.validate(dir(obj), Gettable.method_schema)
        return obj


def is_internally_controlled(obj):
    if hasattr(obj, 'internal'):
        return obj.internal
    return True
