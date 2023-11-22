# About Quantify-core

```{image} /images/QUANTIFY_LANDSCAPE.svg
:class: only-light
```
```{image} /images/QUANTIFY_LANDSCAPE_DM.svg
:class: only-dark
```

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
