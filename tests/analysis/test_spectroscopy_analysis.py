from pathlib import Path

from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import spectroscopy_analysis as sa
from quantify.utilities._tests_helpers import get_test_data_dir


class TestResonatorSpectroscopyAnalysis:
    @classmethod
    def setup_class(cls):
        dh.set_datadir(get_test_data_dir())

        cls.tuid = "20210308-121759-988-76bb50"
        cls.a_obj = sa.ResonatorSpectroscopyAnalysis(tuid=cls.tuid)

    def test_raw_data_not_in_processed_dataset(self):
        container = Path(dh.locate_experiment_container(self.tuid))
        file_path = (
            container
            / "analysis_ResonatorSpectroscopyAnalysis"
            / "processed_dataset.hdf5"
        )
        dataset = dh.load_dataset_from_path(file_path)

        assert "x0" in dataset.dims.keys()
        assert "x0" in dataset.coords.keys()
        assert "y0" not in dataset.data_vars.keys()
        assert "y1" not in dataset.data_vars.keys()
        assert "S21" in dataset.data_vars.keys()

    def test_figures_generated(self):
        # test that the right figures get created.
        assert set(self.a_obj.figs_mpl.keys()) == {
            "S21-RealImag",
            "S21-MagnPhase",
            "S21-complex",
        }

    def test_quantities_of_interest(self):
        assert set(self.a_obj.quantities_of_interest.keys()) == {
            "Qi",
            "Qe",
            "Ql",
            "Qc",
            "fr",
            "fit_res",
            "fit_msg",
        }

        fitted_freq = self.a_obj.quantities_of_interest["fr"]
        assert isinstance(fitted_freq, Variable)
        assert self.a_obj.quantities_of_interest["fr"].nominal_value == approx(
            4.5403e09
        )
        assert self.a_obj.quantities_of_interest["Ql"].nominal_value == approx(
            5977.42438
        )
        assert self.a_obj.quantities_of_interest["Qe"].nominal_value == approx(
            6395.20588
        )
