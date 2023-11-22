# quantify-core

[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://quantify-os.org/slack.html#sec-slack)
[![Pipelines](https://gitlab.com/quantify-os/quantify-core/badges/main/pipeline.svg)](https://gitlab.com/quantify-os/quantify-core/-/pipelines/)
[![PyPi](https://img.shields.io/pypi/v/quantify-core.svg)](https://pypi.org/pypi/quantify-core)
[![Code Quality](https://app.codacy.com/project/badge/Grade/32265e1e7d3f491fa028528aaf8bfa69)](https://app.codacy.com/gl/quantify-os/quantify-core/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-core&amp;utm_campaign=Badge_Grade)
[![Coverage](https://app.codacy.com/project/badge/Coverage/32265e1e7d3f491fa028528aaf8bfa69)](https://app.codacy.com/gl/quantify-os/quantify-core/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-core&amp;utm_campaign=Badge_Coverage)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://gitlab.com/quantify-os/quantify-core/-/blob/main/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)
[![Documentation](https://img.shields.io/badge/documentation-grey)](https://quantify-os.org/docs/quantify-core)

![Quantify logo](https://gitlab.com/quantify-os/quantify-core/-/raw/main/docs/source/images/QUANTIFY_LANDSCAPE.svg)

Quantify is a Python-based data acquisition framework focused on Quantum Computing and
solid-state physics experiments.
The framework consists of [quantify-core](https://pypi.org/project/quantify-core/) ([git](https://gitlab.com/quantify-os/quantify-core/) | [docs](https://quantify-os.org/docs/quantify-core))
and [quantify-scheduler](https://pypi.org/project/quantify-scheduler/) ([git](https://gitlab.com/quantify-os/quantify-scheduler/) | [docs](https://quantify-os.org/docs/quantify-scheduler)).
It is built on top of [QCoDeS](https://qcodes.github.io/Qcodes/)
and is a spiritual successor of [PycQED](https://github.com/DiCarloLab-Delft/PycQED_py3).
`quantify-core` is the core module that contains all basic functionality to control experiments. This includes:

- A framework to control instruments.
- A data-acquisition loop.
- Data storage and analysis.
- Parameter monitoring and live visualization of experiments.

## Overview and Community

For a general overview of Quantify and connecting to its open-source community, see [quantify-os.org](https://quantify-os.org/).
Quantify is maintained by the Quantify Consortium consisting of Qblox and Orange Quantum Systems.

[<img src="https://gitlab.com/quantify-os/quantify-core/-/raw/main/docs/source/images/Qblox_logo.svg" alt="Qblox logo" width=200px/>](https://qblox.com)
&nbsp;
&nbsp;
&nbsp;
&nbsp;
[<img src="https://gitlab.com/quantify-os/quantify-core/-/raw/main/docs/source/images/OQS_logo_with_text.svg" alt="Orange Quantum Systems logo" width=200px/>](https://orangeqs.com)

&nbsp;

The software is free to use under the conditions specified in the [license](https://gitlab.com/quantify-os/quantify-core/-/raw/main/LICENSE).
