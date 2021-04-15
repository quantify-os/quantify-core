# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Helpers for testing quantify."""
import pathlib


def get_test_data_dir():
    return pathlib.Path(__file__).parent.parent.parent.resolve() / "tests" / "test_data"
