# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Analysis module for a AllXY experiment"""
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm
from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import format_value_string


class AllXYAnalysis(ba.BaseAnalysis):
    """
    Normalises the data from an AllXY experiment and plots against an ideal curve
    Parameters
    ------------
    repeats:
        Specifies how many repeats of each data point where taken. If None, this
        is infered from the length of the dataset.
    """

    # Overwrite the run method so that we can add the new optional arguments
    # pylint: disable=attribute-defined-outside-init
    # def run(self, repeats: float = None):
    #     # pylint: disable=arguments-differ

    #     self.repeats = repeats
    #     return super().run()

    # # pylint: disable=attribute-defined-outside-init
    # def run_until(
    #     self,
    #     interrupt_before: Union[str, ba.AnalysisSteps],
    #     repeats: float = None,
    #     **kwargs,
    # ):
    #     # pylint: disable=arguments-differ

    #     self.repeats = repeats
    #     return super().run_until(interrupt_before=interrupt_before, kwargs=kwargs)

    def process_data(self):
        points = self.dataset_raw["x0"]
        self.analysis_result["calibration points"] = points
        raw_data = self.dataset_raw["y0"]
        number_points = len(raw_data)

        # Raise an exception if there are too few points for a complete ALLXY
        # experiment
        if number_points % 21 != 0:
            raise ValueError(
                "Invalid datset. The number of calibration points in an "
                "ALLXY experiment must be a multiple of 21"
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

        ### callibration points for normalization ###
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
        ax.set(ylabel=r"$F$ $|1 \rangle$")
        ax.legend(loc=4)
        fig.suptitle(
            f"Normalised {self.dataset_raw.attrs['name']}\n"
            f"tuid: {self.dataset_raw.attrs['tuid']}"
        )

        fig_id = "AllXY"
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax
