# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Analysis module for a Rabi Oscillation experiment"""
import numpy as np
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm
from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import format_value_string


class RabiAnalysis(ba.BaseAnalysis):
    """
    Fits a cosine curve to Rabi oscillation data and finds the qubit drive
    amplitude required to implement a pi-pulse.
    """

    def process_data(self):
        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.

        self.dataset["Magnitude"] = self.dataset_raw.y0
        self.dataset.Magnitude.attrs["name"] = "Magnitude"
        self.dataset.Magnitude.attrs["units"] = self.dataset_raw.y0.units
        self.dataset.Magnitude.attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset["x0"] = self.dataset_raw.x0
        self.dataset = self.dataset.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset = self.dataset.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a :class:`~quantify.analysis.fitting_models.RabiModel` to the data.
        """
        mod = fm.RabiModel()

        magnitude = np.array(self.dataset["Magnitude"])
        drive_amp = np.array(self.dataset.x0)
        guess = mod.guess(magnitude, drive_amp=drive_amp)
        fit_res = mod.fit(magnitude, params=guess, x=drive_amp)

        self.fit_results.update({"Rabi_oscillation": fit_res})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """
        fit_res = self.fit_results["Rabi_oscillation"]
        fit_warning = ba.check_lmfit(fit_res)

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True

            text_msg = "Summary\n"
            text_msg += format_value_string(
                "Pi-pulse amplitude", fit_res.params["amp180"], unit="V", end_char="\n"
            )
            text_msg += format_value_string(
                "Oscillation amplitude",
                fit_res.params["amplitude"],
                unit="V",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Offset", fit_res.params["offset"], unit="V", end_char="\n"
            )
        else:
            text_msg = fit_warning
            self.quantities_of_interest["fit_success"] = False

        fpars = fit_res.params
        self.quantities_of_interest["Pi-pulse amp"] = ba.lmfit_par_to_ufloat(
            fpars["amp180"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        self.create_fig_rabi_oscillation()

    def create_fig_rabi_oscillation(self):
        """Plot Rabi oscillation figure"""

        fig_id = "Rabi_oscillation"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        qpl.plot_textbox(ax, ba.wrap_text(self.quantities_of_interest["fit_msg"]))

        self.dataset.Magnitude.plot(ax=ax, marker=".", linestyle="")

        qpl.plot_fit(
            ax=ax,
            fit_res=self.fit_results["Rabi_oscillation"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        qpl.set_ylabel(ax, r"Output voltage", self.dataset.Magnitude.units)
        qpl.set_xlabel(ax, self.dataset.x0.long_name, self.dataset.x0.units)

        qpl.set_suptitle_from_dataset(fig, self.dataset_raw, "S21")
