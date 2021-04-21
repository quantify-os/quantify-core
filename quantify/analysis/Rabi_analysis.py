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

        magnitude = np.array(self.dataset["Magnitude"])
        drive_amp = np.array(self.dataset["x0"])
        guess = mod.guess(magnitude, drive_amp=drive_amp)
        fit_res = mod.fit(magnitude, params=guess, x=drive_amp)

        self.fit_res.update({"Rabi_oscillation": fit_res})

        fpars = fit_res.params
        self.quantities_of_interest["Pi-pulse amp"] = ufloat(
            fpars["amp180"].value, fpars["amp180"].stderr
        )

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
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        self.create_fig_rabi_oscillation()

    def create_fig_rabi_oscillation(self):

        fig_id = "Rabi_oscillation"
        fig, axs = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = axs

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs, self.quantities_of_interest["fit_msg"])

        self.dataset.Magnitude.plot(ax=axs, marker=".", linestyle="")

        qpl.plot_fit(
            ax=axs,
            fit_res=self.fit_res["Rabi_oscillation"],
            plot_init=False,
            range_casting="real",
        )

        qpl.set_ylabel(axs, r"Output voltage", self.dataset["Magnitude"].units)
        qpl.set_xlabel(axs, self.dataset["x0"].long_name, self.dataset["x0"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
        )
