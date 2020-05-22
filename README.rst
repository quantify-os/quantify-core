========
quantify
========

Quantify is a python based data acquisition platform focused on  Quantum Computing and solid-state physics experiments.
It is build on top of `QCoDeS <https://qcodes.github.io/Qcodes/>`_ and is a spiritual successor of `PycQED <https://github.com/DiCarloLab-Delft/PycQED_py3>`_.

It differs from QCoDeS in that it not only provides a framework of instruments, parameters and data but also pulse sequencing and a library of standard experiments including analysis.

* Documentation: https://qblox-quantify.readthedocs-hosted.com/en/latest/


Features
--------

Quantify contains all basic functionality to control experiments. This includes:

* A framework for to control instruments + a library of common instruments.
* A measurement loop.
* A framework for data storage and analysis.
* Parameter monitoring and live visualization.
* Pulse sequencer (todo)
* A library of standard experiments and analysis

Take a look at our `Transmock demo <http://>`_ to see quantify in action!


.. note::

    Features are WIP. All features listed should be added before a v1.0 release.
