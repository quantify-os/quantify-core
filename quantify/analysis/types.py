# -----------------------------------------------------------------------------
# Description:    Module containing the types for use with the analysis classes
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from collections import OrderedDict
from jsonschema import validate
from quantify.utilities.general import load_json_schema


class AnalysisSettings(OrderedDict):
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
