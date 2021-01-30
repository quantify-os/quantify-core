# -----------------------------------------------------------------------------
# Description:  Helpers for testing quantify.
# Repository:   https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
import pathlib


def get_test_data_dir():
    return pathlib.Path(__file__).parent.parent.parent.resolve() / "tests" / "test_data"
