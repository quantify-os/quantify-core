from pathlib import Path

import pytest
from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import t1_analysis as ta
from quantify.utilities._tests_helpers import get_test_data_dir


class TestQubitT1Analysis:
    @classmethod
    def setup_class(cls):
        dh.set_datadir(get_test_data_dir())

        tuids = ["20210322-205253-758-6689"]
        cls.tuids = tuids
        cls.a_objs = [ta.QubitT1Analysis(tuid=tuid) for tuid in cls.tuids]

        cls.T1s = [1.07e-5]

        cls.ref_0s = [4.2e-5]

        cls.ref_1s = [7.6e-5]

    def test_raw_data_not_in_processed_dataset(self):
        for tuid in self.tuids:
            container = Path(dh.locate_experiment_container(tuid))
            file_path = (
                container / "analysis_QubitT1Analysis" / "processed_dataset.hdf5"
            )
            dataset = dh.load_dataset_from_path(file_path)

            assert "x0" in dataset.dims.keys()
            assert "x0" in dataset.coords.keys()
            assert "y0" not in dataset.data_vars.keys()
            assert "y1" not in dataset.data_vars.keys()
            assert "Magnitude" in dataset.data_vars.keys()

    def test_figures_generated(self):
        # test that the right figures get created.
        for a_obj in self.a_objs:
            assert set(a_obj.figs_mpl.keys()) == {
                "T1_decay",
            }

    def test_quantities_of_interest(self):
        for a_obj, T1, ref_0, ref_1 in zip(
            self.a_objs, self.T1s, self.ref_0s, self.ref_1s
        ):
            assert set(a_obj.quantities_of_interest.keys()) == {
                "T1",
                "ref_0",
                "ref_1",
                "fit_msg",
                "fit_res",
            }

            assert isinstance(a_obj.quantities_of_interest["T1"], Variable)
            # Tests that the fitted values are correct (to within 5 standard deviations)
            assert a_obj.quantities_of_interest["T1"].nominal_value == approx(
                T1, abs=5 * a_obj.quantities_of_interest["T1"].std_dev
            )
            assert a_obj.quantities_of_interest["ref_0"].nominal_value == approx(
                ref_0, abs=5 * a_obj.quantities_of_interest["ref_0"].std_dev
            )
            assert a_obj.quantities_of_interest["ref_1"].nominal_value == approx(
                ref_1, abs=5 * a_obj.quantities_of_interest["ref_1"].std_dev
            )
