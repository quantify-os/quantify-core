# -----------------------------------------------------------------------------
# Description:    Module containing T1 analysis for a single qubit.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm
from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import format_value_string


class QubitT1Analysis(ba.BaseAnalysis):
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

        mod = fm.T1Model()

        magn = np.array(self.dataset["Magnitude"])
        delay = np.array(self.dataset["x0"])
        guess = mod.guess(magn, delay=delay)
        fit_res = mod.fit(magn, params=guess, t=delay)

        self.model = mod

        self.fit_res.update({"exp_decay_func": fit_res})

        fpars = fit_res.params
        self.quantities_of_interest["T1"] = ufloat(
            fpars["tau"].value, fpars["tau"].stderr
        )
        self.quantities_of_interest["ref_0"] = ufloat(
            fpars["A"].value, fpars["A"].stderr
        )
        self.quantities_of_interest["ref_1"] = ufloat(
            fpars["B"].value, fpars["B"].stderr
        )

        unit = self.dataset["Magnitude"].attrs["units"]
        text_msg = "Summary\n"
        text_msg += format_value_string(
            r"$T1$", fit_res.params["tau"], end_char="\n", unit="s"
        )
        text_msg += format_value_string(
            r"$ref_0$", fit_res.params["A"], end_char="\n", unit=unit
        )
        text_msg += format_value_string(r"$ref_1$", fit_res.params["B"], unit=unit)
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        self.create_fig_T1_decay()

    def create_fig_T1_decay(self):

        fig_id = "T1_decay"
        fig, axs = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = axs

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs, self.quantities_of_interest["fit_msg"])

        self.dataset.Magnitude.plot(ax=axs, marker=".")

        qpl.plot_fit(
            ax=axs,
            fit_res=self.fit_res["exp_decay_func"],
            plot_init=False,
        )

        qpl.set_ylabel(axs, "Magnitude", self.dataset["Magnitude"].units)
        qpl.set_xlabel(axs, self.dataset["x0"].long_name, self.dataset["x0"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
        )
