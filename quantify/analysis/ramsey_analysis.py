# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Analysis module for a Rabi Oscillation experiment"""
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from uncertainties import ufloat
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm
from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import format_value_string


class RamseyAnalysis(ba.BaseAnalysis):
    """
    Fits a decaying cosine curve to Ramsey data (possibly with artificial detuning)
    and finds the true detuning, qubit frequency and T2* time.
    """

    def __init__(
        self,
        dataset_raw: xr.Dataset = None,
        tuid: str = None,
        label: str = "",
        settings_overwrite: dict = None,
        artificial_detuning: float = 0,
        qubit_frequency: float = None,
    ):
        self.artificial_detuning = artificial_detuning
        self.qubit_frequency = qubit_frequency

        super().__init__(
            dataset_raw=dataset_raw,
            label=label,
            tuid=tuid,
            settings_overwrite=settings_overwrite,
        )

    def process_data(self):
        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.
        # y1 = phase in deg, this unit should always be correct
        assert self.dataset_raw["y1"].attrs["units"] == "deg"

        mag = self.dataset_raw["y0"]
        valid_meas = np.logical_not(np.isnan(mag))
        self.dataset["Magnitude"] = mag[valid_meas]
        self.dataset["Magnitude"].attrs["name"] = "Magnitude"
        self.dataset["Magnitude"].attrs["units"] = self.dataset_raw["y0"].attrs["units"]
        self.dataset["Magnitude"].attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset["x0"] = self.dataset_raw["x0"][valid_meas]
        self.dataset = self.dataset.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset = self.dataset.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        mod = fm.DecayOscModel()

        magnitude = np.array(self.dataset["Magnitude"])
        time = np.array(self.dataset["x0"])
        guess = mod.guess(magnitude, time=time)
        fit_res = mod.fit(magnitude, params=guess, t=time)

        self.fit_res.update({"Ramsey_decay": fit_res})

    def analyze_fit_results(self):
        """
        Extract the real detuning and qubit frequency based on the artificial detuning
        and fitted detuning
        """
        fit_warning = ba.check_lmfit(self.fit_res["Ramsey_decay"])
        if fit_warning is not None:
            fit_warning = "\n".join(
                wrap(fit_warning, width=35, replace_whitespace=True)
            )

        fpars = self.fit_res["Ramsey_decay"].params

        self.quantities_of_interest["T2*"] = ba.lmfit_par_to_ufloat(fpars["tau"])
        self.quantities_of_interest["fitted_detuning"] = ba.lmfit_par_to_ufloat(
            fpars["frequency"]
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
                r"$T2^*$", self.quantities_of_interest["T2*"], unit="s", end_char="\n\n"
            )
            text_msg += format_value_string(
                "artificial detuning",
                ufloat(self.artificial_detuning, 0),
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
                    ufloat(self.qubit_frequency, 0),
                    unit="Hz",
                    end_char="\n",
                )
                text_msg += format_value_string(
                    "fitted qubit frequency",
                    self.quantities_of_interest["qubit_frequency"],
                    unit="Hz",
                )
        else:
            text_msg = fit_warning
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        self.create_fig_ramsey_decay()

    def create_fig_ramsey_decay(self):
        """Plot Ramsey decay figure"""

        fig_id = "Ramsey_decay"
        fig, axs = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = axs

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs, self.quantities_of_interest["fit_msg"])

        self.dataset.Magnitude.plot(ax=axs, marker=".", linestyle="")

        qpl.plot_fit(
            ax=axs,
            fit_res=self.fit_res["Ramsey_decay"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        qpl.set_ylabel(axs, r"Output voltage", self.dataset["Magnitude"].units)
        qpl.set_xlabel(axs, self.dataset["x0"].long_name, self.dataset["x0"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\n"
            f"tuid: {self.dataset_raw.attrs['tuid']}"
        )
