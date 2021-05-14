# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm
from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import format_value_string


class EchoAnalysis(ba.BaseAnalysis):
    """
    Analysis class for a qubit spin-echo experiment,
    which fits an exponential decay and extracts the T2_echo time.
    """

    def process_data(self):
        """
        Populates the :code:`.dataset_processed`.
        """
        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.
        # y1 = phase in deg, this unit should always be correct
        assert self.dataset["y1"].attrs["units"] == "deg"

        self.dataset_processed["Magnitude"] = self.dataset["y0"]
        self.dataset_processed["Magnitude"].attrs["name"] = "Magnitude"
        self.dataset_processed["Magnitude"].attrs["units"] = self.dataset["y0"].attrs[
            "units"
        ]
        self.dataset_processed["Magnitude"].attrs["long_name"] = "Magnitude"

        self.dataset_processed["x0"] = self.dataset["x0"]
        self.dataset_processed = self.dataset_processed.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a :class:`~quantify.analysis.fitting_models.ExpDecayModel` to the
        data.
        """
        mod = fm.ExpDecayModel()

        magn = np.array(self.dataset_processed["Magnitude"])
        delay = np.array(self.dataset_processed["x0"])
        guess = mod.guess(magn, delay=delay)
        fit_result = mod.fit(magn, params=guess, t=delay)
        fit_warning = ba.check_lmfit(fit_result)

        self.fit_results.update({"exp_decay_func": fit_result})

        fpars = fit_result.params
        self.quantities_of_interest["t2_echo"] = ba.lmfit_par_to_ufloat(fpars["tau"])

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True
            unit = self.dataset_processed["Magnitude"].attrs["units"]
            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$T_{2,\mathrm{Echo}}$",
                fit_result.params["tau"],
                end_char="\n",
                unit="s",
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

        fig_id = "Echo_decay"
        fig, axs = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = axs

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs, self.quantities_of_interest["fit_msg"])

        self.dataset_processed.Magnitude.plot(ax=axs, marker=".", linestyle="")

        qpl.plot_fit(
            ax=axs,
            fit_res=self.fit_results["exp_decay_func"],
            plot_init=False,
        )

        qpl.set_ylabel(axs, "Magnitude", self.dataset_processed["Magnitude"].units)
        qpl.set_xlabel(
            axs,
            self.dataset_processed["x0"].long_name,
            self.dataset_processed["x0"].units,
        )

        fig.suptitle(
            f"S21 {self.dataset.attrs['name']}\ntuid: " f"{self.dataset.attrs['tuid']}"
        )
