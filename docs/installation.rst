.. highlight:: shell

Installation
==============

Stable release
--------------

This is the preferred method to install Quantify, as it will always install the most recent stable release.
If you want to contribute to quantify, also check out :ref:`Setting up quantify for local development` in the contributing section.


All systems except Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install Quantify::

    $ pip install quantify-core


If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


On Windows (Anaconda)
~~~~~~~~~~~~~~~~~~~~~~~

Quantify-core has third party dependencies that need to be compiled.
The default build process can introduce hard to debug and environment-specific problems.
For this reason we recommend using the `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ python distribution which comes with precompiled binaries for many popular libraries.

1. Install `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ (default options during installation should work well in most setups).

#. Install `Git BASH <https://gitforwindows.org/>`_ to have a unix-like shell (default options during installation should work well in most setups).

    .. tip::

        Users can right click any folder in windows and open Git BASH in that location.

    .. note::

        Be aware that a unix-like shell on windows almost always comes with some caveats. The most likely for you to encounter is that running a python interactive shell requires running :code:`winpty python` or :code:`python -i` instead of just :code:`python`. If possible avoid that altogether and run an IPython shell instead: :code:`ipython`.

#. Add :code:`source /path/to/Anaconda3/etc/profile.d/conda.sh` in the `.bashrc <https://superuser.com/a/602896>`_ (or in the .bash_profile) to expose the anaconda in Git BASH.

    .. tip::
        If you followed the default anaconda installation the path to it will be similar to
        :code:`/c/Users/<YOUR_USERNAME>/anaconda3/etc/profile.d/conda.sh`.

        Pro tip: you can drag and drop a file from the File Explorer into the Git BASH and get the path of the file (instead of typing it manually).

    .. note::

        To confirm you have a functional installation of anaconda, run :code:`conda` in Git BASH. This will print the conda help message which is an indication of a working installation.

#. Create a conda environment, see also the `Conda cheat sheet <https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html>`_.

    .. code-block:: console

        $ conda create --name quantify-env python=3.8   # create the conda environment, you can replace `quantify-env` if you wish
        $ conda activate quantify-env                   # activates the conda environment

    .. tip::

        You can add :code:`conda activate quantify-env` at the end of the .bashrc (or .bash_profile) if you wish for this environment to be activated automatically in the Git BASH shell when it is opened.


#. Install jupyter-lab in the new environment using

    .. code-block:: console

        $ conda install -c conda-forge jupyterlab  # install jupyter lab


#. Install quantify-core using pypi :code:`pip install quantify-core`.

    .. code-block:: console

        $ pip install quantify-core  # install the package into
        $ pip install quantify-...   # optionally install other quantify modules


#. Add the conda environment as a kernel to jupyter.

    .. code-block:: console

        $ python -m ipykernel install --user --name=quantify-env  --display-name="Python 3 Quantify Env"  # adds the environment as an available kernel for jupyter notebook within  jupyter-lab.

    .. note::

        We currently do not have a conda recipe for installation, instead we refer to the default pip installation within the conda environment.




From source
------------

The sources for Quantify can be downloaded from the `GitLab repo <https://gitlab.com/Quantify-os/Quantify-core>`_:

.. code-block:: console

    $ git clone https://gitlab.com/Quantify-os/Quantify-core.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .

If you are a developer you might want to install the package in the editable mode:

.. code-block:: console

    $ pip install -e .

See also :ref:`Setting up quantify for local development` in the contributing section.


Troubleshooting
-------------------

If for some reason you are not able to install or use Quantify using the prescribed ways indicated above, first make sure you first have working python environment (e.g. you are able to run an `IPyhon` shell). Follow the next steps that aim at installing quantify from source and running its tests.

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
