# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
from quantify.visualization.plot_interpolation import interpolate_heatmap
from quantify.visualization import mpl_plotting as qpl


class InterpolationAnalysis2D(ba.BaseAnalysis):
    """
    An analysis class which generates a 2D interpolating plot.
    """

    def create_figures(self):

        for y_var in self.dataset.data_vars:
            variable_name = self.dataset[y_var].attrs["name"]
            unit = self.dataset[y_var].units
            fig_id = f"{variable_name} interpolating"

            xvals0 = self.dataset["x0"].values
            xvals1 = self.dataset["x1"].values
            yvals = self.dataset[y_var].values

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
            fig.colorbar(mappable, ax=ax, label=f"{variable_name} [{unit}]")

            # Scatter plot of measured datapoints
            ax.plot(
                xvals0,
                xvals1,
                marker=".",
                linewidth=0.5,
                markerfacecolor="red",
                markeredgecolor="red",
                markersize=3,
                c="white",
                alpha=1,
            )

            qpl.set_xlabel(
                ax, self.dataset["x0"].attrs["name"], self.dataset["x0"].units
            )
            qpl.set_ylabel(
                ax, self.dataset["x1"].attrs["name"], self.dataset["x1"].units
            )
            qpl.set_suptitle_from_dataset(
                fig, self.dataset, f"{variable_name} interpolating analysis:"
            )
            # qpl.plot_textbox(ax, self.quantities_of_interest["plot_msg"], x=1.32)

            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax
