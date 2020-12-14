.. highlight:: shell

============
Installation
============

Stable release
--------------

On Windows
~~~~~~~~~~~~

Quantify-core has third party dependencies that need to be compiled.
The default build process can introduce hard to debug and environment specific problems.
For this reason we recommend using the `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ python distribution which comes with precompiled binaries for many popular libraries.

1. Install `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_.
2. Install `Git BASH <https://gitforwindows.org/>`_ to have a unix shell.
3. Add :code:`source /path/to/Anaconda3/etc/profile.d/conda.sh` in the `bash_profile <https://superuser.com/questions/602872/how-do-i-modify-my-git-bash-profile-in-windows>`_  to expose the anaconda in Git Bash.
4. Install jupyter-lab using :code:`conda install -c conda-forge jupyterlab`.
5. Create a conda environment, see also the `Conda cheat sheet <https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html>`_.
6. Install quantify-core using pypi :code:`pip install quantify-core`.
7. Add the conda environment as a kernel to jupyter.


.. code-block:: console

    $ conda install -c conda-forge jupyterlab                   # install jupyter lab
    $ conda create --name quantify-env python=3.8               # create the conda environment
    $ conda activate quantify-env                               # activates the conda environment
    $ pip install quantify-core                                 # install the package into
    $ pip install quantify-...                                  # optionally install other quantify modules
    $ python -m ipykernel install --user --name=quantify-env    # adds the environment as a kernel to start a notebook from in jupyter-lab.

Verify that the installation was succesful by running the test suite.

.. code-block::

    $ pip show quantify-core        # shows the path where quantify-core was installed as "location".
    $ pytest path_to_quantify_core  # run pytest on tests in the quantify-core repository.


.. note::

    We currently do not have a conda recipe for installation, instead we refer to the default pip installation within the conda environment.



On all other systems (Linux, MacOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install Quantify, run this command in your terminal:

.. code-block:: console

    $ pip install quantify-core

This is the preferred method to install Quantify, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Quantify can be downloaded from the `GitLab repo`_:

.. code-block:: console

    $ git clone https://gitlab.com/Quantify-os/Quantify-core.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .

If you are a developer you might want to install the package in the editable mode:

.. code-block:: console

    $ pip install -e .

.. _GitLab repo: https://gitlab.com/Quantify-os/Quantify-core


Potential issues
-------------------

PyQTgraph and PyQT5
~~~~~~~~~~~~~~~~~~~~~~~~

Quantify-core has a dependency on the PyQt5 package, which itself has a dependency on the Qt5 runtime.
On most systems, the standard installation process will correctly install Qt.
The Anaconda installation should resolve issues with installation on Windows.
You may need to consult a search engine if you have a more exotic system.


.. warning::

    We use the pyqtgraph library which contains an `issue with venv on Windows`_. Windows users should see the linked
    issue for details and prefer `virtualenv` over `python -m venv`.

.. _issue with venv on Windows: https://github.com/pyqtgraph/pyqtgraph/issues/1052
