# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm
from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import format_value_string


class T1Analysis(ba.BaseAnalysis):
    """
    Analysis class for a qubit T1 experiment,
    which fits an exponential decay and extracts the T1 time.
    """

    def process_data(self):

        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.
        # y1 = phase in deg, this unit should always be correct
        assert self.dataset_raw.y1.units == "deg"

        self.dataset["Magnitude"] = self.dataset_raw.y0
        self.dataset.Magnitude.attrs["name"] = "Magnitude"
        self.dataset.Magnitude.attrs["units"] = self.dataset_raw.y0.units
        self.dataset.Magnitude.attrs["long_name"] = "Magnitude"

        self.dataset["x0"] = self.dataset_raw.x0
        self.dataset = self.dataset.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset = self.dataset.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a :class:`~quantify.analysis.fitting_models.ExpDecayModel` to the data.
        """

        mod = fm.ExpDecayModel()

        magn = np.array(self.dataset["Magnitude"])
        delay = np.array(self.dataset.x0)
        guess = mod.guess(magn, delay=delay)
        fit_res = mod.fit(magn, params=guess, t=delay)

        self.fit_results.update({"exp_decay_func": fit_res})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """

        fit_res = self.fit_results["exp_decay_func"]
        fit_warning = ba.check_lmfit(fit_res)

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True
            unit = self.dataset.Magnitude.units
            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$T1$", fit_res.params["tau"], end_char="\n", unit="s"
            )
            text_msg += format_value_string(
                "amplitude", fit_res.params["amplitude"], end_char="\n", unit=unit
            )
            text_msg += format_value_string(
                "offset", fit_res.params["offset"], unit=unit
            )
        else:
            text_msg = fit_warning
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["T1"] = ba.lmfit_par_to_ufloat(
            fit_res.params["tau"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """
        Create a figure showing the exponential decay and fit.
        """

        fig_id = "T1_decay"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        qpl.plot_textbox(ax, ba.wrap_text(self.quantities_of_interest["fit_msg"]))

        self.dataset.Magnitude.plot(ax=ax, marker=".", linestyle="")

        qpl.plot_fit(
            ax=ax,
            fit_res=self.fit_results["exp_decay_func"],
            plot_init=False,
        )

        qpl.set_ylabel(ax, "Magnitude", self.dataset.Magnitude.units)
        qpl.set_xlabel(ax, self.dataset.x0.long_name, self.dataset.x0.units)

        qpl.set_suptitle_from_dataset(fig, self.dataset_raw, "S21")
