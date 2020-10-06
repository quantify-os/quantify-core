#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('AUTHORS.rst') as authors_file:
    authors = authors_file.read()

with open('requirements_dev.txt') as test_reqs:
    test_requirements = test_reqs.read().splitlines()

requirements = [
    'qcodes',
    'numpy',
    'scipy',
    'xarray',
    'columnar',
    'xxhash',
    'matplotlib',
    'lmfit',
    'pyqt5==5.14.0',
    'pyqtgraph',
    'plotly',
    'jsonschema',
    'adaptive'
]

setup_requirements = ['pytest-runner', ]

version = '0.1.1'

setup(
    author="QBlox BV",
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Unified quantum computing, solid-state and pulse sequencing physical experimentation framework.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + authors + '\n\n' + history,
    include_package_data=True,
    keywords='quantify-core',
    name='quantify-core',
    packages=find_packages(include=['quantify', 'quantify.*']),
    package_data={'': ['*.json']},  # ensures JSON schema are included
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.com/qblox/packages/software/quantify',
    version='0.1.1',
    zip_safe=False,
)


from PyQt5 import QtCore
print("XXX")
print(QtCore.PYQT_VERSION_STR.split('.'))
print("XXX")
