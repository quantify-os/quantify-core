# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
from quantify.visualization.plot_interpolation import interpolate_heatmap
from quantify.visualization.SI_utilities import format_value_string
from quantify.visualization import mpl_plotting as qpl


class InterpolationAnalysis2D(ba.BaseAnalysis):
    """
    An analysis class which generates a 2D interpolating plot
    """

    # Override the run method so that we can add the new optional arguments
    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def run(self, minimize: bool = True):
        """
        Parameters
        ----------
        minimize:
            Boolean which determines whether to minimise or maximise the function.
            True for minimize.
            False for maximize.

        Returns
        -------
        :class:`~quantify.analysis.interpolation_analysis2D.InterpolationAnalysis2D`:
            The instance of this analysis.

        """  # NB the return type need to be specified manually to avoid circular import
        self.minimize = minimize
        return super().run()

    def process_data(self):
        unit = self.dataset["y0"].units

        if self.minimize:
            optimum_0 = float(
                self.dataset["x0"][np.argmin(self.dataset["y0"].values)].values
            )
            optimum_1 = float(
                self.dataset["x1"][np.argmin(self.dataset["y0"].values)].values
            )
        else:
            optimum_0 = float(
                self.dataset["x0"][np.argmax(self.dataset["y0"].values)].values
            )
            optimum_1 = float(
                self.dataset["x1"][np.argmax(self.dataset["y0"].values)].values
            )
        self.quantities_of_interest[self.dataset["x0"].attrs["name"]] = optimum_0
        self.quantities_of_interest[self.dataset["x1"].attrs["name"]] = optimum_1

        text_msg = "Summary\n"

        text_msg += format_value_string(
            self.dataset["x0"].attrs["name"],
            optimum_0,
            end_char="\n",
            unit=unit,
        )
        text_msg += format_value_string(
            self.dataset["x1"].attrs["name"],
            optimum_1,
            end_char="\n",
            unit=unit,
        )
        self.quantities_of_interest["plot_msg"] = text_msg

    def create_figures(self):

        fig_id = "2D_interpolating"

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

        qpl.set_xlabel(ax, self.dataset["x0"].attrs["name"], self.dataset["x0"].units)
        qpl.set_ylabel(ax, self.dataset["x1"].attrs["name"], self.dataset["x1"].units)
        fig.suptitle(
            f"{self.dataset.attrs['name'] } 2D interpolating analysis\n"
            f"tuid: {self.dataset.attrs['tuid']}"
        )
        qpl.plot_textbox(ax, self.quantities_of_interest["plot_msg"], x=1.32)

        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax
