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
        assert self.dataset_raw["y1"].attrs["units"] == "deg"

        self.dataset["Magnitude"] = self.dataset_raw["y0"]
        self.dataset["Magnitude"].attrs["name"] = "Magnitude"
        self.dataset["Magnitude"].attrs["units"] = self.dataset_raw["y0"].attrs["units"]
        self.dataset["Magnitude"].attrs["long_name"] = "Magnitude"

        self.dataset["x0"] = self.dataset_raw["x0"]
        self.dataset = self.dataset.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset = self.dataset.swap_dims({"dim_0": "x0"})

    def run_fitting(self):

        mod = fm.ExpDecayModel()

        magn = np.array(self.dataset["Magnitude"])
        delay = np.array(self.dataset["x0"])
        guess = mod.guess(magn, delay=delay)
        fit_result = mod.fit(magn, params=guess, t=delay)
        fit_warning = ba.check_lmfit(fit_result)

        self.fit_result.update({"exp_decay_func": fit_result})

        fpars = fit_result.params
        self.quantities_of_interest["T1"] = ba.lmfit_par_to_ufloat(fpars["tau"])

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True
            unit = self.dataset["Magnitude"].attrs["units"]
            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$T1$", fit_result.params["tau"], end_char="\n", unit="s"
            )
            text_msg += format_value_string(
                "amplitude", fit_result.params["amplitude"], end_char="\n", unit=unit
            )
            text_msg += format_value_string(
                "offset", fit_result.params["offset"], unit=unit
            )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """
        Create a figure showing the exponential decay and fit.
        """

        fig_id = "T1_decay"
        fig, axs = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = axs

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs, self.quantities_of_interest["fit_msg"])

        self.dataset.Magnitude.plot(ax=axs, marker=".", linestyle="")

        qpl.plot_fit(
            ax=axs,
            fit_res=self.fit_result["exp_decay_func"],
            plot_init=False,
        )

        qpl.set_ylabel(axs, "Magnitude", self.dataset["Magnitude"].units)
        qpl.set_xlabel(axs, self.dataset["x0"].long_name, self.dataset["x0"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\ntuid: "
            f"{self.dataset_raw.attrs['tuid']}"
        )
