"""Tests for Rabi analysis module"""
from pathlib import Path
from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import Rabi_analysis as Ra
from quantify.utilities._tests_helpers import get_test_data_dir


class TestRabiAnalysis:
    """Test the analysis class for a Rabi oscillation experiment"""

    @classmethod
    def setup_class(cls):
        """Setup test objects"""
        dh.set_datadir(get_test_data_dir())

        cls.tuids = "20210419-153127-883-fa4508"
        cls.a_objs = Ra.RabiAnalysis(tuid=cls.tuids)
        cls.values = {"Pi-pulse amp": 498.8e-3}

    def test_raw_data_not_in_processed_dataset(self):
        """Check that the required data is in the dataset"""
        tuid = self.tuids
        container = Path(dh.locate_experiment_container(tuid))
        file_path = container / "analysis_RabiAnalysis" / "processed_dataset.hdf5"
        dataset = dh.load_dataset_from_path(file_path)

        assert "x0" in dataset.dims.keys()
        assert "x0" in dataset.coords.keys()
        assert "y0" not in dataset.data_vars.keys()
        assert "y1" not in dataset.data_vars.keys()
        assert "Magnitude" in dataset.data_vars.keys()

    def test_figures_generated(self):
        """test that the right figures get created"""
        a_obj = self.a_objs
        assert set(a_obj.figs_mpl.keys()) == {
            "Rabi_oscillation",
        }

    def test_quantities_of_interest(self):
        """Test that the quantities of interest have the correct values"""
        a_obj = self.a_objs
        values = self.values
        assert set(a_obj.quantities_of_interest.keys()) == {
            "Pi-pulse amp",
            "fit_msg",
            "fit_res",
        }

        assert isinstance(a_obj.quantities_of_interest["Pi-pulse amp"], Variable)
        # Tests that the fitted values are correct (to within 5 standard deviations)
        assert a_obj.quantities_of_interest["Pi-pulse amp"].nominal_value == approx(
            values["Pi-pulse amp"],
            abs=5 * a_obj.quantities_of_interest["Pi-pulse amp"].std_dev,
        )
