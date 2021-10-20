# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Module containing the types for use with the analysis classes"""
from collections import UserDict

from jsonschema import validate

from quantify_core.utilities.general import load_json_schema


# pylint: disable=too-few-public-methods
# WARNING! Do not inherit from dict! if you do, `AnalysisSettings.update will skip the
# validation done in `__setitem__`.
class AnalysisSettings(UserDict):
    """
    Analysis settings with built-in schema validations.

    .. jsonschema:: schemas/AnalysisSettings.json#/configurations
    """

    schema = load_json_schema(__file__, "AnalysisSettings.json")["configurations"]
    schema_individual = dict(schema)
    schema_individual.pop("required")

    def __init__(self, settings: dict = None):
        """Initializes and validates the passed settings."""
        super().__init__()
        if settings:
            validate(settings, self.schema)
            for key, value in settings.items():
                super().__setitem__(key, value)

    def __setitem__(self, key, value):
        """Items are validated before assigning."""
        validate({key: value}, self.schema_individual)
        super().__setitem__(key, value)
