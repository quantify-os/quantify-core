import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm
from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import format_value_string


class ResonatorSpectroscopyAnalysis(ba.BaseAnalysis):
    def process_data(self):

        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.
        # y1 = phase in deg, this unit should always be correct
        assert self.dataset_raw["y1"].attrs["units"] == "deg"

        S21 = self.dataset_raw["y0"] * np.cos(
            np.deg2rad(self.dataset_raw["y1"])
        ) + 1j * self.dataset_raw["y0"] * np.sin(np.deg2rad(self.dataset_raw["y1"]))
        self.dataset["S21"] = S21
        self.dataset["S21"].attrs["name"] = "S21"
        self.dataset["S21"].attrs["units"] = self.dataset_raw["y0"].attrs["units"]
        self.dataset["S21"].attrs["long_name"] = "Transmission, $S_{21}$"

        self.dataset["x0"] = self.dataset_raw["x0"]
        self.dataset = self.dataset.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset = self.dataset.swap_dims({"dim_0": "x0"})

    def run_fitting(self):

        mod = fm.ResonatorModel()

        S21 = np.array(self.dataset["S21"])
        freq = np.array(self.dataset["x0"])
        guess = mod.guess(S21, f=freq)
        fit_res = mod.fit(S21, params=guess, f=freq)

        self.model = mod

        self.fit_res.update({"hanger_func_complex_SI": fit_res})

        fp = fit_res.params
        self.quantities_of_interest["Qi"] = ufloat(fp["Qi"].value, fp["Qi"].stderr)
        self.quantities_of_interest["Qe"] = ufloat(fp["Qe"].value, fp["Qe"].stderr)
        self.quantities_of_interest["Ql"] = ufloat(fp["Ql"].value, fp["Ql"].stderr)
        self.quantities_of_interest["Qc"] = ufloat(fp["Qc"].value, fp["Qc"].stderr)
        self.quantities_of_interest["fr"] = ufloat(fp["fr"].value, fp["fr"].stderr)

        text_msg = "Summary\n"
        text_msg += format_value_string(r"$Q_I$", fit_res.params["Qi"], end_char="\n")
        text_msg += format_value_string(r"$Q_C$", fit_res.params["Qc"], end_char="\n")
        text_msg += format_value_string(r"$f_{res}$", fit_res.params["fr"], unit="Hz")
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        self.create_fig_S21_real_imag()
        self.create_fig_S21_magn_phase()
        self.create_fig_S21_complex()

    def create_fig_S21_real_imag(self):

        fig_id = "S21-RealImag"
        fig, axs = plt.subplots(2, 1, sharex=True)
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id + "_Re"] = axs[0]
        self.axs_mpl[fig_id + "_Im"] = axs[1]

        # Add a textbox with the fit_message
        box_props = dict(boxstyle="round", pad=0.4, facecolor="white", alpha=0.5)
        axs[0].text(
            1.05,
            0.95,
            self.quantities_of_interest["fit_msg"],
            transform=axs[0].transAxes,
            bbox=box_props,
            verticalalignment="top",
        )

        self.dataset.S21.real.plot(ax=axs[0], marker=".")
        self.dataset.S21.imag.plot(ax=axs[1], marker=".")

        qpl.plot_fit(
            ax=axs[0],
            fit_res=self.fit_res["hanger_func_complex_SI"],
            plot_init=True,
            range_casting="real",
        )

        qpl.plot_fit(
            ax=axs[1],
            fit_res=self.fit_res["hanger_func_complex_SI"],
            plot_init=True,
            range_casting="imag",
        )

        qpl.set_ylabel(axs[0], r"Re$(S_{21})$", self.dataset["S21"].units)
        qpl.set_ylabel(axs[1], r"Im$(S_{21})$", self.dataset["S21"].units)
        axs[0].set_xlabel("")
        qpl.set_xlabel(axs[1], self.dataset["x0"].long_name, self.dataset["x0"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
        )

    def create_fig_S21_magn_phase(self):

        fig_id = "S21-MagnPhase"
        fig, axs = plt.subplots(2, 1, sharex=True)
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id + "_Magn"] = axs[0]
        self.axs_mpl[fig_id + "_Phase"] = axs[1]

        # Add a textbox with the fit_message
        box_props = dict(boxstyle="round", pad=0.4, facecolor="white", alpha=0.5)
        axs[0].text(
            1.05,
            0.95,
            self.quantities_of_interest["fit_msg"],
            transform=axs[0].transAxes,
            bbox=box_props,
            verticalalignment="top",
        )

        axs[0].plot(self.dataset["x0"], np.abs(self.dataset.S21), marker=".")
        axs[1].plot(
            self.dataset["x0"], np.angle(self.dataset.S21, deg=True), marker="."
        )

        qpl.plot_fit(
            ax=axs[0],
            fit_res=self.fit_res["hanger_func_complex_SI"],
            plot_init=True,
            range_casting="abs",
        )

        qpl.plot_fit(
            ax=axs[1],
            fit_res=self.fit_res["hanger_func_complex_SI"],
            plot_init=True,
            range_casting="angle",
        )

        qpl.set_ylabel(axs[0], r"$|S_{21}|$", self.dataset["S21"].units)
        qpl.set_ylabel(axs[1], r"$\angle S_{21}$", "deg")
        axs[0].set_xlabel("")
        qpl.set_xlabel(axs[1], self.dataset["x0"].long_name, self.dataset["x0"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
        )

    def create_fig_S21_complex(self):

        fig_id = "S21-complex"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        box_props = dict(boxstyle="round", pad=0.4, facecolor="white", alpha=0.5)
        ax.text(
            1.05,
            0.95,
            self.quantities_of_interest["fit_msg"],
            transform=ax.transAxes,
            bbox=box_props,
            verticalalignment="top",
        )

        ax.plot(self.dataset.S21.real, self.dataset.S21.imag, marker=".")

        qpl.plot_fit_complex_plane(
            ax=ax,
            fit_res=self.fit_res["hanger_func_complex_SI"],
            plot_init=True,
        )

        qpl.set_xlabel(ax, r"Re$(S_{21})$", self.dataset["S21"].units)
        qpl.set_ylabel(ax, r"Im$(S_{21})$", self.dataset["S21"].units)

        fig.suptitle(
            f"S21 {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
        )
