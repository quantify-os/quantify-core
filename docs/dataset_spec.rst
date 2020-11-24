
==========================
DataStorage specification
==========================

Quantify experiment output storage is based on ideas from PycQED, QCoDeS and xarray. Conceptually it is divided into two parts:

- Dataset, stored in `dataset.hdf5`. This file contains the raw data with associated attributes, such as axis labels and units.
- Metadata, stored in `snapshot.json`. This file contains information such as when the experiment was run, what devices were in use and so on.

As previously described in :ref:`Data storage & Analysis`, an individual experiment is represented on disk by these two files in a directory named relative to the start time of the experiment. Experiments are grouped in directories by date. And the root directory of all experiments being used by Quantify can be retrieved/set with :meth:`~quantify.data.handling.get_datadir`/:meth:`~quantify.data.handling.set_datadir`.

Dataset
~~~~~~~~~

The Dataset is implemented using the xarray :class:`xarray.Dataset` class.

.. note:: To support both gridded and non-gridded data, we use Xarray using only `datavariables` and without any `coordinates`  or `dimensions` (as described below). This is necessary as in the non-gridded case the dataset will be a perfect sparse array, usability of which is cumbersome. This does mean that some of Xarray's more advanced functionality, such as the in-built graphing or query system, are unavailable without further processing.

Quantify arranges data along two types of axes: X and Y.
In each dataset there will be *n* X axes and *m* Y axes. For example, the dataset produced in an experiment where we sweep 2 parameters (settables) and measure 3 other parameters (all 3 returned by a Gettable), we will have *n* = 2 and *m* = 3.
Each X axis represents a dimension of the setpoints provided. The Y axes represent the output of the Gettable.
Each axis type are numbered ascending from 0 (e.g. x0, x1, y0, y1, y2), and each stores information described by the :class:`~quantify.measurement.Settable` and
:class:`~quantify.measurement.Gettable` classes, such as titles and units. The Dataset object also stores some further metadata,
such as the :class:`~quantify.data.types.TUID` of the experiment which it was generated from.

For example, consider an experiment varying time and amplitude against a Cosine function.
The resulting dataset will look similar to the following:

.. jupyter-execute::

    from qcodes import ManualParameter, Parameter
    from quantify.measurement.control import MeasurementControl
    import numpy as np

    t = ManualParameter('t', initial_value=1, unit='s', label='Time')
    amp = ManualParameter('amp', initial_value=1, unit='V', label='Amplitude')

    def CosFunc():
        return amp() * np.cos(2 * np.pi * 1e6 * t())

    sig = Parameter(name='sig', label='Signal level', unit='V', get_cmd=CosFunc)

    MC = MeasurementControl('MC')
    MC.settables([t, amp])
    MC.setpoints_grid([np.linspace(0, 5, 20), np.linspace(-1, 1, 5)])
    MC.gettables(sig)
    MC.run('my experiment')
