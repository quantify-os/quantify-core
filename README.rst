=============
Quantify-core
=============

.. image:: https://img.shields.io/badge/slack-chat-green.svg
    :target: https://join.slack.com/t/quantify-hq/shared_invite/zt-vao45946-f_NaRc4mvYQDQE_oYB8xSw
    :alt: Slack

.. image:: https://gitlab.com/quantify-os/quantify-core/badges/develop/pipeline.svg
    :target: https://gitlab.com/quantify-os/quantify-core/pipelines/
    :alt: Pipelines

.. image:: https://img.shields.io/pypi/v/quantify-core.svg
    :target: https://pypi.org/pypi/quantify-core
    :alt: PyPi

.. image:: https://app.codacy.com/project/badge/Grade/32265e1e7d3f491fa028528aaf8bfa69
    :target: https://www.codacy.com/gl/quantify-os/quantify-core/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-core&amp;utm_campaign=Badge_Grade
    :alt: Code Quality

.. image:: https://app.codacy.com/project/badge/Coverage/32265e1e7d3f491fa028528aaf8bfa69
    :target: https://www.codacy.com/gl/quantify-os/quantify-core/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-core&amp;utm_campaign=Badge_Coverage
    :alt: Coverage

.. image:: https://readthedocs.com/projects/quantify-quantify-core/badge/?version=develop&token=2f68e7fc6a2426b5eb9b44bb2f764a9d75a9932f41c39efdf0a8a99bf33e6a34
    :target: https://quantify-quantify-core.readthedocs-hosted.com
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-BSD%204--Clause-blue.svg
    :target: https://gitlab.com/quantify-os/quantify-core/-/blob/master/LICENSE
    :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style

.. image:: https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat
    :target: http://unitary.fund
    :alt: Unitary Fund



.. figure:: https://orangeqs.com/logos/QUANTIFY_LANDSCAPE.svg
    :align: center
    :alt: Quantify logo

Quantify is a python based data acquisition platform focused on Quantum Computing and solid-state physics experiments.
It is build on top of `QCoDeS <https://qcodes.github.io/Qcodes/>`_ and is a spiritual successor of `PycQED <https://github.com/DiCarloLab-Delft/PycQED_py3>`_.
Quantify currently consists of `quantify-core <https://pypi.org/project/quantify-core/>`_ and `quantify-scheduler <https://pypi.org/project/quantify-scheduler/>`_.

Take a look at the documentation for quantify-core: `last release <https://quantify-quantify-core.readthedocs-hosted.com/>`_ (or `develop <https://quantify-quantify-core.readthedocs-hosted.com/en/develop/?badge=develop>`_).

Quantify-core is the core module that contains all basic functionality to control experiments. This includes:

* A framework to control instruments.
* A data-acquisition loop.
* Data storage and analysis.
* Parameter monitoring and live visualization of experiments.


.. caution::

    This is a pre-release **beta version**, changes and improvements are expected.

Overview
--------

Quantify evolves rapidly, nevertheless, he following `presentation <https://www.youtube.com/embed/koWIp12hD8Q?start=150&end=1126>`_ by Adriaan Rol gives
a good general overview of Quantify.


About
-----

Quantify-core is maintained by The Quantify consortium consisting of Qblox and Orange Quantum Systems.

.. |_| unicode:: 0xA0
   :trim:


.. figure:: https://cdn.sanity.io/images/ostxzp7d/production/f9ab429fc72aea1b31c4b2c7fab5e378b67d75c3-132x31.svg
    :width: 200px
    :target: https://qblox.com
    :align: left

.. figure:: https://orangeqs.com/OQS_logo_with_text.svg
    :width: 200px
    :target: https://orangeqs.com
    :align: left

|_|

|_|

The software is free to use under the conditions specified in the license.
