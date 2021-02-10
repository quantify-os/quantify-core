from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import spectroscopy_analysis as sa
from quantify.utilities._tests_helpers import get_test_data_dir


class TestResonatorSpectroscopyAnalysis:
    @classmethod
    def setup_class(cls):
        dh.set_datadir(get_test_data_dir())

        tuid = "20210118-202044-211-58ddb0"
        cls.a = sa.ResonatorSpectroscopyAnalysis(tuid=tuid)

    def test_figures_generated(self):
        # test that the right figures get created.
        assert set(self.a.figs_mpl.keys()) == {"S21"}

    def test_quantities_of_interest(self):
        assert set(self.a.quantities_of_interest.keys()) == {
            "Qi",
            "Qe",
            "Ql",
            "Qc",
            "fr",
            "fit_res",
        }

        fitted_freq = self.a.quantities_of_interest['fr']
        assert isinstance(fitted_freq, Variable)
        assert self.a.quantities_of_interest['fr'].nominal_value == approx(7649998552)
