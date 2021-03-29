# -----------------------------------------------------------------------------
# Description:    Module containing spectroscopy analysis.
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


class RabiAnalysis(ba.BaseAnalysis):
    def process_data(self):

        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.
        # y1 = phase in deg, this unit should always be correct
        # assert self.dataset_raw["y1"].attrs["units"] == "deg"

        self.dataset["Magnitude"] = self.dataset_raw["y0"]
        self.dataset["Magnitude"].attrs["name"] = "Magnitude"
        self.dataset["Magnitude"].attrs["units"] = self.dataset_raw["y0"].attrs["units"]
        self.dataset["Magnitude"].attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset["x0"] = self.dataset_raw["x0"]
        self.dataset = self.dataset.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset = self.dataset.swap_dims({"dim_0": "x0"})

    def run_fitting(self):

        mod = fm.RabiModel()

        Magnitude = np.array(self.dataset["Magnitude"])
        x0 = np.array(self.dataset["x0"])
        guess = mod.guess(Magnitude)
        fit_res = mod.fit(Magnitude, params=guess, t=x0)

        self.model = mod

        self.fit_res.update({"Rabi_oscillation": fit_res})

        fpars = fit_res.params
        self.quantities_of_interest["omega"] = ufloat(
            fpars["omega"].value, fpars["omega"].stderr
        )
        self.quantities_of_interest["t_pi"] = ufloat(
            fpars["t_pi"].value, fpars["t_pi"].stderr
        )

        text_msg = "Summary\n"
        text_msg += format_value_string(
            r"$\Omega$", fit_res.params["omega"], unit="Hz", end_char="\n"
        )
        text_msg += format_value_string(
            r"$t_\pi$", fit_res.params["t_pi"], unit="s", end_char="\n"
        )
        text_msg += format_value_string(
            r"$A$", fit_res.params["A"], unit="V", end_char="\n"
        )
        text_msg += format_value_string(
            r"$\mathrm{offset}$", fit_res.params["offset"], unit="V", end_char="\n"
        )
        text_msg += format_value_string(r"$\phi$", fit_res.params["phi"])
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        self.create_fig_Rabi_oscillation()

    def create_fig_Rabi_oscillation(self):

        fig_id = "Rabi_oscillation"
        fig, axs = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = axs

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs, self.quantities_of_interest["fit_msg"])

        self.dataset.Magnitude.plot(ax=axs, marker=".")

        qpl.plot_fit(
            ax=axs,
            fit_res=self.fit_res["Rabi_oscillation"],
            plot_init=True,
            range_casting="real",
        )

        qpl.set_ylabel(axs, r"Magnitude", self.dataset["Magnitude"].units)
        qpl.set_xlabel(axs, self.dataset["x0"].long_name, self.dataset["x0"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
        )
