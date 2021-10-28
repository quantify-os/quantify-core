"""
The visualization module contains tools for real-time visualization as
well as utilities to help in plotting.

.. list-table::
    :header-rows: 1
    :widths: auto

    * - Import alias
      - Maps to
    * - :class:`!quantify_core.visualization.InstrumentMonitor`
      - :class:`.InstrumentMonitor`
    * - :class:`!quantify_core.visualization.PlotMonitor_pyqt`
      - :class:`.PlotMonitor_pyqt`
"""

from .instrument_monitor import InstrumentMonitor
from .pyqt_plotmon import PlotMonitor_pyqt

# Commented out because it messes up Sphinx and sphinx extensions
# __all__ = ["PlotMonitor_pyqt", "InstrumentMonitor"]
