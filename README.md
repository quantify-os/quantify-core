# Quantify-core

[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://join.slack.com/t/quantify-hq/shared_invite/zt-vao45946-f_NaRc4mvYQDQE_oYB8xSw)
[![Pipelines](https://gitlab.com/quantify-os/quantify-core/badges/main/pipeline.svg)](https://gitlab.com/quantify-os/quantify-core/pipelines/)
[![PyPi](https://img.shields.io/pypi/v/quantify-core.svg)](https://pypi.org/pypi/quantify-core)
[![Code Quality](https://app.codacy.com/project/badge/Grade/32265e1e7d3f491fa028528aaf8bfa69)](https://www.codacy.com/gl/quantify-os/quantify-core/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-core&amp;utm_campaign=Badge_Grade)
[![Coverage](https://app.codacy.com/project/badge/Coverage/32265e1e7d3f491fa028528aaf8bfa69)](https://www.codacy.com/gl/quantify-os/quantify-core/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-core&amp;utm_campaign=Badge_Coverage)
[![Documentation Status](https://readthedocs.com/projects/quantify-quantify-core/badge/?version=latest)](https://quantify-quantify-core.readthedocs-hosted.com)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://gitlab.com/quantify-os/quantify-core/-/blob/main/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](http://unitary.fund)

![Quantify logo](https://orangeqs.com/logos/QUANTIFY_LANDSCAPE.svg)

Quantify is a python based data acquisition platform focused on Quantum Computing and
solid-state physics experiments. It is built on top of [QCoDeS](https://qcodes.github.io/Qcodes/)
and is a spiritual successor of [PycQED](https://github.com/DiCarloLab-Delft/PycQED_py3).
Quantify currently consists of [quantify-core](https://pypi.org/project/quantify-core/)
and [quantify-scheduler](https://pypi.org/project/quantify-scheduler/).

Take a look at the [latest documentation for quantify-core](https://quantify-quantify-core.readthedocs-hosted.com/)
or use the switch at the bottom of the left panel to read the documentation for older releases.

Quantify-core is the core module that contains all basic functionality to control experiments. This includes:

- A framework to control instruments.
- A data-acquisition loop.
- Data storage and analysis.
- Parameter monitoring and live visualization of experiments.

⚠️**Caution!**
This is a pre-release **beta version**, changes and improvements are expected.

## Overview

Quantify evolves rapidly, nevertheless, the following [presentation](https://www.youtube.com/embed/koWIp12hD8Q?start=150&end=1126)
by Adriaan Rol gives a good general overview of Quantify.

## About

Quantify-core is maintained by The Quantify consortium consisting of Qblox and Orange Quantum Systems.

[<img src="https://cdn.sanity.io/images/ostxzp7d/production/f9ab429fc72aea1b31c4b2c7fab5e378b67d75c3-132x31.svg" alt="Qblox logo" width=200px/>](https://qblox.com)
&nbsp;
&nbsp;
&nbsp;
&nbsp;
[<img src="https://orangeqs.com/OQS_logo_with_text.svg" alt="Orange Quantum Systems logo" width=200px/>](https://orangeqs.com)

&nbsp;

The software is free to use under the conditions specified in the license.
