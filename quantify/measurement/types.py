# -----------------------------------------------------------------------------
# Description:    Module containing the core types for use with the MeasurementControl.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import jsonschema
from quantify.utilities.general import load_json_schema


class Settable:
    """
    Defines the Settable concept, which is considered complete if the given type satisfies the following:
    This object does not wrap the passed in object but simply verifies and returns it.

    .. jsonschema:: schemas/Settable.json#/attrs
    .. jsonschema:: schemas/Settable.json#/methods
    """

    schema = load_json_schema(__file__, 'Settable.json')

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

    schema = load_json_schema(__file__, 'Gettable.json')

    def __new__(cls, obj):
        jsonschema.validate(vars(obj), Gettable.schema['attrs'])
        jsonschema.validate(dir(obj), Gettable.schema['methods'])
        return obj


def is_batched(obj):
    return getattr(obj, 'batched', False)
