#!/usr/bin/env python
# pylint: disable=import-outside-toplevel
from setuptools import setup


def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os  # noqa: PLC0415
    from importlib.util import (  # noqa: PLC0415
        module_from_spec,
        spec_from_file_location,
    )

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


version, cmdclass = get_version_and_cmdclass(r"quantify_core")

setup(
    version=version,
    cmdclass=cmdclass,
)
