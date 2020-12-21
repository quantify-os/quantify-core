
**Before instantiating any instruments or starting a measurement** we change the directory in which the experiments data will be saved. For this Quantify provides the :meth:`~quantify.data.handling.get_datadir`/:meth:`~quantify.data.handling.set_datadir` functions.


.. tip::

    It is **highly recommended to**:

    **(a)** change the default directory when starting the python kernel (after importing Quantify); and

    **(b)** settle for a single common data directory for all notebooks/experiments within your measurement setup/PC (e.g. *D:/Data*).

    Quantify provides utilities to find/search and extract data, which expects all your experiment containers to be located within the same directory (under the corresponding date subdirectory).


.. jupyter-execute::

    # Always set the directory at the start of the python kernel
    # And stick to a single common data directory for all
    # notebooks/experiments within your measurement setup/PC
    import os # path utilities
    from quantify.data.handling import get_datadir, set_datadir

    # we set the datadir to the default one FOR TUTORIAL PURPOSES ONLY!!!
    # we highly recommend to change it! See note above.

    datadir = get_datadir() # CHANGE ME!!!
    set_datadir(datadir)

    print("Data will be saved in \n" + os.path.abspath(get_datadir()))
