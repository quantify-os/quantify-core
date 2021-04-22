from pathlib import Path

import pytest
from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import Rabi_analysis as Ra
from quantify.utilities._tests_helpers import get_test_data_dir


class TestRabiAnalysis:
    @classmethod
    def setup_class(cls):
        dh.set_datadir(get_test_data_dir())

        cls.tuids = ["20210419-153127-883-fa4508"]
        cls.a_objs = [Ra.RabiAnalysis(tuid=tuid) for tuid in cls.tuid_time_scan]
        cls.values = [{"amp180": 498.8e-3}]

    def test_raw_data_not_in_processed_dataset(self):
        for tuid in self.tuid_time_scan:
            container = Path(dh.locate_experiment_container(tuid))
            file_path = container / "analysis_RabiAnalysis" / "processed_dataset.hdf5"
            dataset = dh.load_dataset_from_path(file_path)

            assert "x0" in dataset.dims.keys()
            assert "x0" in dataset.coords.keys()
            assert "y0" not in dataset.data_vars.keys()
            assert "y1" not in dataset.data_vars.keys()
            assert "Magnitude" in dataset.data_vars.keys()

    def test_figures_generated(self):
        # test that the right figures get created.
        for a_obj in self.a_time_scan:
            assert set(a_obj.figs_mpl.keys()) == {
                "Rabi_oscillation",
            }

    def test_quantities_of_interest(self):
        for a_obj, values in zip(self.a_time_scan, self.values_time_scan):
            assert set(a_obj.quantities_of_interest.keys()) == {
                "amp180",
            }

            assert isinstance(a_obj.quantities_of_interest["amp180"], Variable)
            # Tests that the fitted values are correct (to within 5 standard deviations)
            assert a_obj.quantities_of_interest["amp180"].nominal_value == approx(
                values["amp180"], abs=5 * a_obj.quantities_of_interest["amp180"].std_dev
            )
