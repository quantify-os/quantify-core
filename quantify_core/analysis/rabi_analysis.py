# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from quantify_core.analysis import base_analysis as ba
from quantify_core.analysis import fitting_models as fm
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import format_value_string


class RabiAnalysis(ba.BaseAnalysis):
    """
    Fits a cosine curve to Rabi oscillation data and finds the qubit drive
    amplitude required to implement a pi-pulse.
    """

    def process_data(self):
        """
        Populates the :code:`.dataset_processed`.
        """
        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.

        self.dataset_processed["Magnitude"] = self.dataset.y0
        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.Magnitude.attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset_processed["x0"] = self.dataset.x0
        self.dataset_processed = self.dataset_processed.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a :class:`~quantify_core.analysis.fitting_models.RabiModel` to the data.
        """
        model = fm.RabiModel()

        magnitude = self.dataset_processed["Magnitude"].values
        drive_amplitude = self.dataset_processed.x0.values
        guess = model.guess(magnitude, drive_amp=drive_amplitude)
        fit_result = model.fit(magnitude, params=guess, x=drive_amplitude)

        self.fit_results.update({"Rabi_oscillation": fit_result})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """
        fit_result = self.fit_results["Rabi_oscillation"]
        fit_warning = ba.wrap_text(ba.check_lmfit(fit_result))

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True

            text_msg = "Summary\n"
            text_msg += format_value_string(
                "Pi-pulse amplitude",
                fit_result.params["amp180"],
                unit="V",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Oscillation amplitude",
                fit_result.params["amplitude"],
                unit="V",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Offset", fit_result.params["offset"], unit="V", end_char="\n"
            )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["Pi-pulse amplitude"] = ba.lmfit_par_to_ufloat(
            fit_result.params["amp180"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """Creates Rabi oscillation figure"""

        fig_id = "Rabi_oscillation"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        qpl.plot_textbox(ax, self.quantities_of_interest["fit_msg"])

        self.dataset_processed.Magnitude.plot(ax=ax, marker=".", linestyle="")

        qpl.plot_fit(
            ax=ax,
            fit_res=self.fit_results["Rabi_oscillation"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        qpl.set_ylabel(ax, r"Output voltage", self.dataset_processed.Magnitude.units)
        qpl.set_xlabel(
            ax, self.dataset_processed.x0.long_name, self.dataset_processed.x0.units
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset, "S21")
