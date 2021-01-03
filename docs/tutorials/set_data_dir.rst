
**Before instantiating any instruments or starting a measurement** we change the directory in which the experiments are saved using the :meth:`~quantify.data.handling.set_datadir`[/:meth:`~quantify.data.handling.get_datadir`] functions.


.. tip::

    We **highly recommended to** settle for a single common data directory for all notebooks/experiments within your measurement setup/PC (e.g. *D:\\Data*, or */d/Data* for a unix-like shell on Windows).
    The utilities to find/search/extract data only work if all the experiment containers are located within the same directory.

.. jupyter-execute::
    :hide-code:

    from quantify.data import handling
    # FOR TUTORIAL PURPOSES ONLY!!!
    # DO NOT RUN THIS CELL, YOU RISK TO LOSE YOUR DATA!
    import quantify.data.handling as dh
    datadir = dh._default_datadir

.. jupyter-execute::

    # Always set the directory at the start of the python kernel
    # And stick to a single common data directory for all
    # notebooks/experiments within your measurement setup/PC
    import os # path utilities
    from quantify.data.handling import get_datadir, set_datadir

    # datadir = "/path/to/your/datadir" # UNCOMMENT AND CHANGE ME!!!
    set_datadir(datadir)
    print("Data will be saved in \n" + os.path.abspath(get_datadir()))
