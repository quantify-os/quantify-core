.. highlight:: shell

============
Installation
============

Stable release
--------------

To install quantify, run this command in your terminal:

.. code-block:: console

    $ pip install quantify-scheduler

This is the preferred method to install quantify, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for quantify can be downloaded from the `GitLab repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://gitlab.com/quantify-os/quantify-scheduler

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .


.. _GitLab repo: https://gitlab.com/quantify-os/quantify-scheduler


Jupyter and plotly
-------------------

Quantify-scheduler uses the `ploty`_ graphing framework for some components, which can require some additional set-up
to run with a Jupyter environment - please see `this page for details.`_


.. _ploty: https://plotly.com/
.. _this page for details.: https://plotly.com/python/getting-started/#jupyter-notebook-support
