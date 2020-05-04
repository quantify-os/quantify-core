import pytest
import xarray as xr
import numpy as np
from quantify.visualization import PlotMonitor_pyqt

import os
import quantify
from quantify import set_datadir

test_datadir = os.path.join(os.path.split(
    quantify.__file__)[0], '..', 'tests', 'data')


class TestPlotMonitor_pyqt:

    @classmethod
    def setup_class(cls):
        cls.plotmon = PlotMonitor_pyqt(name='plotmon')
        # ensures the default datadir is used which is excluded from git
        set_datadir(test_datadir)

    @classmethod
    def teardown_class(cls):
        cls.plotmon.close()
        set_datadir(None)

    def test_attributes_created_during_init(self):
        hasattr(self.plotmon, 'main_QtPlot')
        hasattr(self.plotmon, 'secondary_QtPlot')


    def test_basic_1D_plot(self):
        # Test 1D plotting using an example dataset
        self.plotmon.tuid('20200430-170837-315f36')
        self.plotmon.update()

        x = self.plotmon.curves[0]['config']['x']
        y = self.plotmon.curves[0]['config']['y']

        x_exp = np.linspace(0, 5, 50)
        y_exp = np.cos(np.pi*x_exp)
        np.testing.assert_allclose(x, x_exp)
        np.testing.assert_allclose(y, y_exp)