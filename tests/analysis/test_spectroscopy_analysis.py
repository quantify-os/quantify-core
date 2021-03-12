from pathlib import Path

import pytest
from pytest import approx
from uncertainties.core import Variable
import quantify.data.handling as dh
from quantify.analysis import spectroscopy_analysis as sa
from quantify.utilities._tests_helpers import get_test_data_dir


class TestResonatorSpectroscopyAnalysis:
    @classmethod
    def setup_class(cls):
        dh.set_datadir(get_test_data_dir())

        tuid_list = dh.get_tuids_containing(
            "Resonator_id", t_start="20210305", t_stop="20210306"
        )

        cls.tuids = tuid_list
        cls.a_objs = [sa.ResonatorSpectroscopyAnalysis(tuid=tuid) for tuid in cls.tuids]

        cls.frs = [
            4482627786.933104,
            4482670162.566818,
            4540287828.70407,
            4576421867.293702,
        ]
        cls.Qls = [
            4983.385483402395,
            4192.005581230714,
            4910.617635185228,
            6437.377871269456,
        ]
        cls.Qes = [
            6606.202849302761,
            7317.398211359418,
            5216.566199947343,
            7759.742961176549,
        ]

    @pytest.mark.skip(reason="blocked by #161, see `base_analisys.AnalysisSteps`")
    def test_raw_data_not_in_processed_dataset(self):
        for tuid in self.tuids:
            container = Path(dh.locate_experiment_container(tuid))
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
        for a_obj in self.a_objs:
            assert set(a_obj.figs_mpl.keys()) == {
                "S21-RealImag",
                "S21-MagnPhase",
                "S21-complex",
            }

    def test_quantities_of_interest(self):
        for a_obj, fr, Ql, Qe in zip(self.a_objs, self.frs, self.Qls, self.Qes):
            assert set(a_obj.quantities_of_interest.keys()) == {
                "Qi",
                "Qe",
                "Ql",
                "Qc",
                "fr",
                "fit_res",
                "fit_msg",
            }

            fitted_freq = a_obj.quantities_of_interest["fr"]
            assert isinstance(fitted_freq, Variable)
            # Tests that the fitted values are correct (to within 5 standard deviations)
            assert a_obj.quantities_of_interest["fr"].nominal_value == approx(
                fr, abs=5 * a_obj.quantities_of_interest["fr"].std_dev
            )
            assert a_obj.quantities_of_interest["Ql"].nominal_value == approx(
                Ql, abs=5 * a_obj.quantities_of_interest["Ql"].std_dev
            )
            assert a_obj.quantities_of_interest["Qe"].nominal_value == approx(
                Qe, abs=5 * a_obj.quantities_of_interest["Qe"].std_dev
            )
