# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from quantify.analysis import base_analysis as ba
from quantify.visualization.plot_interpolation import interpolate_heatmap
from quantify.visualization.SI_utilities import format_value_string
from quantify.visualization import mpl_plotting as qpl


class MixerOffsetAnalysis(ba.BaseAnalysis):
    """
    An analysis class for a an interpolating mixer offset optimisation. Extracts the
    optimal mixer offset values and plots figures of the spectral power versus mixer
    offset on each channel as well as an interpolated 2D plot for both channnels
    """

    def process_data(self):
        unit = self.dataset["y0"].units

        offset_min_0 = float(
            self.dataset["x0"][np.argmin(self.dataset["y0"].values)].values
        )
        offset_min_1 = float(
            self.dataset["x1"][np.argmin(self.dataset["y0"].values)].values
        )
        self.quantities_of_interest["offset_channel_0"] = offset_min_0
        self.quantities_of_interest["offset_channel_1"] = offset_min_1

        text_msg = "Summary\n"
        # TODO: get rid of these ufloats one the MR on format_value_string is merged
        text_msg += format_value_string(
            "Mixer offset channel 0", ufloat(offset_min_0, 0), end_char="\n", unit=unit
        )
        text_msg += format_value_string(
            "Mixer offset channel 1", ufloat(offset_min_1, 0), end_char="\n", unit=unit
        )
        self.quantities_of_interest["plot_msg"] = text_msg

    def create_figures(self):
        self._fig_interpolating2D()

    def _fig_interpolating2D(self):

        fig_id = "mixer_offset_2D"

        xvals0 = self.dataset["x0"].values
        xvals1 = self.dataset["x1"].values
        yvals = self.dataset["y0"].values

        fig, ax = plt.subplots()
        # Interpolated 2D heatmap
        extent = (min(xvals0), max(xvals0), min(xvals1), max(xvals1))
        interpolated_datset = interpolate_heatmap(
            xvals0,
            xvals1,
            yvals,
            interp_method="linear",
        )
        mappable = ax.imshow(
            interpolated_datset[2],
            extent=extent,
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(mappable, ax=ax, label="Signal power [dBm]")

        # Scatter plot of measured datapoints
        ax.scatter(xvals0, xvals1, s=2, c="red", alpha=1)

        ax.set_xlabel(f'Mixer offset channel 0 [{self.dataset["x0"].units}]')
        ax.set_ylabel(f'Mixer offset channel 1 [{self.dataset["x0"].units}]')
        fig.suptitle(
            f"Mixer {self.dataset.attrs['name']}\n"
            f"tuid: {self.dataset.attrs['tuid']}"
        )
        qpl.plot_textbox(ax, self.quantities_of_interest["plot_msg"], x=1.32)

        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax
