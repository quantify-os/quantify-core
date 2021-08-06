# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Analysis module for a AllXY experiment"""
from __future__ import annotations
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from quantify_core.analysis import base_analysis as ba
from quantify_core.visualization import mpl_plotting as qpl


class AllXYAnalysis(ba.BaseAnalysis):
    """
    Normalizes the data from an AllXY experiment and plots against an ideal curve
    """

    def process_data(self):
        points = self.dataset["x0"]
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
        ideal_data = xr.DataArray(
            data=np.concatenate(
                (
                    0 * np.ones(5 * repeats),
                    0.5 * np.ones(12 * repeats),
                    np.ones(4 * repeats),
                )
            ),
            name="ideal_data",
            dims="dim_0",
        )

        experiment_numbers = xr.DataArray(
            data=np.arange(0, 21, 1),
            name="experiment_numbers",
            dims="dim_1",  # has less points than "dim_0", so we use different dims
        )

        ### calibration points for normalization ###
        # II is set as 0 cal point
        zero = np.sum(raw_data[points == 0]) / len(raw_data[points == 0])
        # average of XI and YI is set as the 1 cal point
        one = np.sum(raw_data[np.logical_or(points == 17, points == 18)]) / len(
            raw_data[np.logical_or(points == 17, points == 18)]
        )

        normalized_data = xr.DataArray(
            data=(raw_data - zero) / (one - zero),
            name="normalized_data",
            dims="dim_0",
        )

        self.dataset_processed = xr.merge(
            [self.dataset_processed, experiment_numbers, ideal_data, normalized_data]
        )

        ### Analyzing Data ###
        deviation = np.mean(abs(normalized_data - ideal_data)).item()
        self.quantities_of_interest["deviation"] = deviation

    def create_figures(self):

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

        ax.plot(self.dataset.x0, self.dataset_processed.normalized_data, "o-")
        ax.plot(
            self.dataset.x0,
            self.dataset_processed.ideal_data,
            label="Ideal data",
        )
        deviation = self.quantities_of_interest["deviation"]
        ax.text(1, 1, f"Deviation: {deviation:#.2g}", fontsize=11)
        ax.xaxis.set_ticks(self.dataset_processed.experiment_numbers)
        ax.set_xticklabels(labels, rotation=60)
        ax.set(ylabel="Normalized readout signal")
        ax.legend(loc=4)
        qpl.set_suptitle_from_dataset(fig, self.dataset, "Normalized")
