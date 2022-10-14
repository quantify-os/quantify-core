#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup


def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


version, cmdclass = get_version_and_cmdclass(r"quantify_core")

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md") as history_file:
    history = history_file.read()

with open("AUTHORS.md") as authors_file:
    authors = authors_file.read()

with open("requirements.txt") as installation_requirements_file:
    requirements = installation_requirements_file.read().splitlines()

with open("requirements_setup.txt") as setup_requirements_file:
    setup_requirements = setup_requirements_file.read().splitlines()

with open("requirements_dev.txt") as test_requirements_file:
    test_requirements = test_requirements_file.read().splitlines()

setup(
    author="The Quantify consortium consisting of Qblox and Orange Quantum Systems",
    author_email="maintainers@quantify-os.org",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="""Unified quantum computing, solid-state and pulse sequencing """
    """physical experimentation framework.""",
    install_requires=requirements,
    license="BSD-3 license",
    long_description=readme + "\n\n" + authors + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="quantify-core",
    name="quantify-core",
    packages=find_packages(include=["quantify_core", "quantify_core.*"]),
    package_data={
        "": ["*.json"],  # ensures JSON schema are included
        "quantify_core": ["py.typed"],
    },
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.com/quantify-os/quantify-core",
    version=version,
    cmdclass=cmdclass,
    zip_safe=False,
)
