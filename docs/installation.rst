.. highlight:: console

Installation
==============

Stable release
--------------

This is the preferred method to install Quantify, as it will always install the most recent stable release.
If you want to contribute to quantify, also check out :ref:`Setting up quantify for local development` in the contributing section.


All systems except Windows and macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm that you have a working python 3.7+ and run the following in your terminal of choice

.. code-block:: console

    $ python --version
    # Expected output similar to:
    # Python 3.7.6

To install Quantify

.. code-block:: console

    $ pip install quantify-core


If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


On Windows and macOS (Anaconda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantify-core has third party dependencies that need to be compiled.
The default build process can introduce hard to debug and environment-specific problems.
For this reason we recommend using the `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ python distribution which comes with precompiled binaries for many popular libraries.

1. Install `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ (default options during installation should work well in most setups).

#. (Windows only) Install `Git BASH <https://gitforwindows.org/>`_ to have a unix-like bash terminal (default options during installation should work well in most setups).

    .. tip::

        Users can right click any folder in windows and open Git BASH in that location.

    .. note::

        Be aware that a unix-like terminal on windows has some caveats. To avoid them, we recommend to run any python code using :ref:`JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_ (installation steps follows below).

#. (Windows only) Add :code:`source /path/to/Anaconda3/etc/profile.d/conda.sh` in the `.bashrc <https://superuser.com/a/602896>`_ (or in the `.bash_profile`) to expose the anaconda in bash terminal.

    .. tip::

        If you followed the default anaconda installation the path to it will be similar to
        :code:`/c/Users/<YOUR_USERNAME>/anaconda3/etc/profile.d/conda.sh` (Windows) or :code:`/usr/local/anaconda3/etc/profile.d/conda.sh` (macOS).

        Pro tip: you can drag and drop a file from the file explorer into the terminal and get the path of the file (instead of typing it manually).

    .. note::

        To confirm you have a functional installation of anaconda, run :code:`conda` in the terminal. This will print the conda help message which is an indication of a working installation.

#. Create a conda environment, see also the `Conda cheat sheet <https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html>`_.

    .. code-block:: console

        $ conda create --name quantify-env python=3.8   # create the conda environment, you can replace `quantify-env` if you wish
        $ conda activate quantify-env                   # activates the conda environment

    .. tip::

        You can add :code:`conda activate quantify-env` at the end of the `.bashrc` (or `.bash_profile`) if you wish for this environment to be activated automatically in the terminal when it is opened.


#. Install jupyter-lab in the new environment using

    .. code-block:: console

        $ conda install -c conda-forge jupyterlab  # install jupyter lab


#. Install quantify-core from pypi.

    .. tip::

        If you are an early adopter or interested to contribute to Quantify you should :ref:`install it from source <From source>`.

    .. code-block:: console

        $ pip install quantify-core  # install the package into
        $ pip install quantify-...   # optionally install other quantify modules

    .. tip::

        To update quantify to the latest version:

        .. code-block:: console

            $ pip install --upgrade quantify-core

        If you require a specific version, e.g. 0.3.0, run:

        .. code-block:: console

            $ pip install --upgrade quantify-core==0.3.0

#. Add the conda environment as a kernel to jupyter.

    .. code-block:: console

        # adds the environment as an available kernel for jupyter notebook within  jupyter-lab.
        $ python -m ipykernel install --user --name=quantify-env  --display-name="Python 3 Quantify Env"

    .. note::

        We currently do not have a conda recipe for installation, instead we refer to the default pip installation within the conda environment.

From source
------------

The source code of Quantify can be downloaded from the `GitLab repo <https://gitlab.com/Quantify-os/Quantify-core>`_ or installed from your terminal:

.. code-block:: console

    $ git clone https://gitlab.com/Quantify-os/Quantify-core.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .

.. tip::

    If you are a developer or wish to contribute you might want to install the package in the editable mode:

    .. code-block:: console

        $ pip install -e .

    See also :ref:`Setting up quantify for local development` in the contributing section.


Troubleshooting
-------------------

If for some reason you are not able to install or use Quantify using the prescribed ways indicated above, make sure you have working python environment (e.g. you are able to run an `IPyhon` terminal). Follow the next steps that aim at installing Quantify from source and running its tests.

0. Uninstall Quantify

    .. code-block:: console

        $ pip uninstall quantify-core

#. Install from source (run line by line)

    .. code-block:: console

        $ git clone https://gitlab.com/Quantify-os/Quantify-core.git; cd quantify-core
        $ pip install .
        $ pip install pytest
        $ pytest

#. The tests will either pass or not. In any case, please report your experience and which test do not pass by creating a `New issue` on the `issue tracker <https://gitlab.com/quantify-os/quantify-core/-/issues>`_, your efforts are much appreciated and will help us to understand the problems you might be facing.


Potential issues
~~~~~~~~~~~~~~~~~~~~~~~~

PyQtGraph and PyQt5
^^^^^^^^^^^^^^^^^^^^^^^^^

Quantify-core has a dependency on the PyQt5 package, which itself has a dependency on the Qt5 runtime.
On most systems, the standard installation process will correctly install Qt.
The Anaconda installation should resolve issues with installation on Windows.
You may need to consult a search engine if you have a more exotic system.
