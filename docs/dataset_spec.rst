
==========================
DataStorage specification
==========================

Quantify experiment output storage is based on ideas from PycQED, QCoDeS and xarray. Conceptually it is divided into two parts:

- Dataset, stored in dataset.hdf5. This file contains the raw data with associated attributes, such as axis labels and units.
- Metadata, stored in snapshot.json. This file contains information such as when the experiment was run, what devices were in use and so on.

As previously described, an individual experiment is represented on disk by these two files in a directory named relative to the start time of the experiment.

Dataset
~~~~~~~~~

The Dataset is implemented using the xarray :class:`~xarray.DataSet` class. Most concepts are translated

Quantify arranges data along n-many X axis an one Y axis. The X axis represents the setpoints provided, which are tiled in the case of multiple dimensions.
The Y axis represents the output of the Gettable. Both are numbered ascending from 0, and each stores information described by the :class:`~quantify.measurement.Settable`
and :class:`~quantify.measurement.Gettable` classes, such as titles and units. The Dataset object also stores some further metadata, such as the TUID of the experiment which it was generated from.

For example, consider an experiment varying time and amplitude against a Cosine function.
The resulting dataset will look something like the following:

.. jupyter-execute::

    from qcodes import ManualParameter, Parameter
    from quantify.measurement.control import MeasurementControl
    import numpy as np

    t = ManualParameter('t', initial_value=1, unit='s', label='Time')
    amp = ManualParameter('amp', initial_value=1, unit='V', label='Amplitude')

    def CosFunc():
        return amp() * np.cos(2 * np.pi * 1e6 * t())

    sig = Parameter(name='sig', label='Signal level', unit='V', get_cmd=CosFunc)

    MC = MeasurementControl('cosine')
    MC.settables([t, amp])
    MC.setpoints_grid([np.linspace(0, 5, 20), np.linspace(-1, 1, 5)])
    MC.gettables(sig)
    MC.run('my experiment')
