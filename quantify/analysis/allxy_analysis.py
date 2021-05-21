# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Analysis module for a AllXY experiment"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
from quantify.visualization import mpl_plotting as qpl


class AllXYAnalysis(ba.BaseAnalysis):
    """
    Normalizes the data from an AllXY experiment and plots against an ideal curve
    """

    def process_data(self):
        points = self.dataset["x0"]
        self.analysis_result["calibration points"] = points
        raw_data = self.dataset["y0"]
        number_points = len(raw_data)

        # Raise an exception if we do not have the correct number of points for a
        # complete ALLXY experiment
        if number_points % 21 != 0:
            raise ValueError(
                "Invalid dataset. The number of calibration points in an "
                "AllXY experiment must be a multiple of 21"
            )

        repeats = int(round(number_points / 21))

        ### Creating the ideal data ###
        self.analysis_result["ideal_data"] = np.concatenate(
            (
                0 * np.ones(5 * repeats),
                0.5 * np.ones(12 * repeats),
                np.ones(4 * repeats),
            )
        )
        self.analysis_result["experiment numbers"] = np.arange(0, 21, 1)

        ### calibration points for normalization ###
        # II is set as 0 cal point
        zero = np.sum(raw_data[points == 0]) / len(raw_data[points == 0])
        # average of XI and YI is set as the 1 cal point
        one = np.sum(raw_data[np.logical_or(points == 17, points == 18)]) / len(
            raw_data[np.logical_or(points == 17, points == 18)]
        )

        data_normalized = (raw_data - zero) / (one - zero)

        ### Analyzing Data ###
        data_error = data_normalized - self.analysis_result["ideal_data"]
        deviation = np.mean(abs(data_error))

        self.analysis_result["normalized_data"] = data_normalized
        self.analysis_result["deviation"] = deviation
        self.quantities_of_interest["deviation"] = deviation.item()

    def create_figures(self):
        data = self.analysis_result

        fig, ax = plt.subplots()
        fig_id = "AllXY"
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        labels = [
            "II",
            "XX",
            "YY",
            "XY",
            "YX",
            "xI",
            "yI",
            "xy",
            "yx",
            "xY",
            "yX",
            "Xy",
            "Yx",
            "xX",
            "Xx",
            "yY",
            "Yy",
            "XI",
            "YI",
            "xx",
            "yy",
        ]

        ax.plot(data["calibration points"], data["normalized_data"], "o-")
        ax.plot(data["calibration points"], data["ideal_data"], label="Ideal data")
        deviation_text = r"Deviation: %#.2g" % data["deviation"]
        ax.text(1, 1, deviation_text, fontsize=11)
        ax.xaxis.set_ticks(data["experiment numbers"])
        ax.set_xticklabels(labels, rotation=60)
        ax.set(ylabel="Normalized readout signal")
        ax.legend(loc=4)
        qpl.set_suptitle_from_dataset(fig, self.dataset, "Normalized")
