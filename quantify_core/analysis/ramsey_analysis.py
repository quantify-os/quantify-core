# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from quantify_core.analysis import base_analysis as ba
from quantify_core.analysis import fitting_models as fm
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import format_value_string


class RamseyAnalysis(ba.BaseAnalysis):
    """
    Fits a decaying cosine curve to Ramsey data (possibly with artificial detuning)
    and finds the true detuning, qubit frequency and T2* time.
    """

    # Override the run method so that we can add the new optional arguments
    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def run(self, artificial_detuning: float = 0, qubit_frequency: float = None):
        """
        Parameters
        ----------
        artificial_detuning:
            The detuning in Hz that will be emulated by adding an extra phase in
            software.
        qubit_frequency:
            The initial recorded value of the qubit frequency in software (before
            accurate fitting is done) in Hz.

        Returns
        -------
        :class:`~quantify_core.analysis.ramsey_analysis.RamseyAnalysis`:
            The instance of this analysis.

        """  # NB the return type need to be specified manually to avoid circular import
        self.artificial_detuning = artificial_detuning
        self.qubit_frequency = qubit_frequency
        return super().run()

    def process_data(self):
        """
        Populates the :code:`.dataset_processed`.
        """
        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.
        # y1 = phase in deg, this unit should always be correct
        assert self.dataset.y1.units == "deg"

        magnitude = self.dataset.y0
        # TODO solve NaNs properly when #176 has a solution, pylint: disable=fixme
        valid_measurement = np.logical_not(np.isnan(magnitude))
        self.dataset_processed["Magnitude"] = magnitude[valid_measurement]
        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.Magnitude.attrs["long_name"] = r"Magnitude, $|S_{21}|$"

        self.dataset_processed["x0"] = self.dataset.x0[valid_measurement]
        self.dataset_processed = self.dataset_processed.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a :class:`~quantify_core.analysis.fitting_models.DecayOscillationModel` to the
        data.
        """
        model = fm.DecayOscillationModel()

        magnitude = self.dataset_processed["Magnitude"].values
        time = self.dataset_processed.x0.values
        guess = model.guess(magnitude, time=time)
        fit_result = model.fit(magnitude, params=guess, t=time)

        self.fit_results.update({"Ramsey_decay": fit_result})

    def analyze_fit_results(self):
        """
        Extract the real detuning and qubit frequency based on the artificial detuning
        and fitted detuning.
        """
        fit_warning = ba.check_lmfit(self.fit_results["Ramsey_decay"])

        fit_parameters = self.fit_results["Ramsey_decay"].params

        self.quantities_of_interest["T2*"] = ba.lmfit_par_to_ufloat(
            fit_parameters["tau"]
        )
        self.quantities_of_interest["fitted_detuning"] = ba.lmfit_par_to_ufloat(
            fit_parameters["frequency"]
        )
        self.quantities_of_interest["detuning"] = (
            self.quantities_of_interest["fitted_detuning"] - self.artificial_detuning
        )

        if self.qubit_frequency is not None:
            self.quantities_of_interest["qubit_frequency"] = (
                self.qubit_frequency - self.quantities_of_interest["detuning"]
            )

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True

            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$T_2^*$",
                self.quantities_of_interest["T2*"],
                unit="s",
                end_char="\n\n",
            )
            text_msg += format_value_string(
                "artificial detuning",
                self.artificial_detuning,
                unit="Hz",
                end_char="\n",
            )
            text_msg += format_value_string(
                "fitted detuning",
                self.quantities_of_interest["fitted_detuning"],
                unit="Hz",
                end_char="\n",
            )
            text_msg += format_value_string(
                "actual detuning",
                self.quantities_of_interest["detuning"],
                unit="Hz",
                end_char="\n",
            )

            if self.qubit_frequency is not None:
                text_msg += "\n"
                text_msg += format_value_string(
                    "initial qubit frequency",
                    self.qubit_frequency,
                    unit="Hz",
                    end_char="\n",
                )
                text_msg += format_value_string(
                    "fitted qubit frequency",
                    self.quantities_of_interest["qubit_frequency"],
                    unit="Hz",
                )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """Plot Ramsey decay figure"""

        fig_id = "Ramsey_decay"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        qpl.plot_textbox(ax, self.quantities_of_interest["fit_msg"])

        self.dataset_processed.Magnitude.plot(ax=ax, marker=".", linestyle="")

        qpl.plot_fit(
            ax=ax,
            fit_res=self.fit_results["Ramsey_decay"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        qpl.set_ylabel(ax, r"Output voltage", self.dataset_processed.Magnitude.units)
        qpl.set_xlabel(
            ax,
            self.dataset_processed.x0.long_name,
            self.dataset_processed.x0.units,
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset, "S21")
