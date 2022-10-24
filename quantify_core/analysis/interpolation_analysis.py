# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
import matplotlib.pyplot as plt

from quantify_core.analysis import base_analysis as ba
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.plot_interpolation import interpolate_heatmap


class InterpolationAnalysis2D(ba.BaseAnalysis):
    """
    An analysis class which generates a 2D interpolating plot for each yi variable in
    the dataset.
    """

    def create_figures(self):
        """Create a 2D interpolating figure for each yi."""

        for y_variable in self.dataset.data_vars:
            variable_name = self.dataset[y_variable].attrs["long_name"]
            unit = self.dataset[y_variable].units
            fig_id = f"{variable_name} interpolating"

            x_values_0 = self.dataset["x0"].values
            x_values_1 = self.dataset["x1"].values
            y_values = self.dataset[y_variable].values

            fig, ax = plt.subplots()
            # Interpolated 2D heatmap
            extent = (
                min(x_values_0),
                max(x_values_0),
                min(x_values_1),
                max(x_values_1),
            )
            interpolated_datset = interpolate_heatmap(
                x_values_0,
                x_values_1,
                y_values,
                interp_method="linear",
            )
            mappable = ax.imshow(
                interpolated_datset[2],
                extent=extent,
                aspect="auto",
                origin="lower",
            )
            cbar = fig.colorbar(mappable, ax=ax)

            # Scatter plot of measured datapoints
            ax.plot(
                x_values_0,
                x_values_1,
                marker=".",
                linewidth=0.5,
                linestyle="",
                markerfacecolor="red",
                markeredgecolor="red",
                markersize=3,
                c="white",
                alpha=1,
            )

            qpl.set_xlabel(
                self.dataset["x0"].attrs["long_name"], self.dataset["x0"].units, ax
            )
            qpl.set_ylabel(
                self.dataset["x1"].attrs["long_name"], self.dataset["x1"].units, ax
            )
            qpl.set_cbarlabel(cbar, variable_name, unit)
            qpl.set_suptitle_from_dataset(
                fig, self.dataset, f"{variable_name} interpolating analysis:"
            )

            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax
