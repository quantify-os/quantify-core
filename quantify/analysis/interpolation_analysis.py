# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
from quantify.visualization.plot_interpolation import interpolate_heatmap
from quantify.visualization.SI_utilities import (
    format_value_string,
    adjust_axeslabels_SI,
)
from quantify.visualization import mpl_plotting as qpl


class OptimizationAnalysis(ba.BaseAnalysis):
    """
    An analysis class which extracts the optimal quantities from an N-dimensional
    interpolating experiment.
    """

    # Override the run method so that we can add the new optional arguments
    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def run(self, minimize: bool = True):
        """
        Parameters
        ----------
        minimize:
            Boolean which determines whether to report the minimum or the maximum.
            True for minimize.
            False for maximize.

        Returns
        -------
        :class:`~quantify.analysis.interpolation_analysis.OptimizationAnalysis`:
            The instance of this analysis.

        """  # NB the return type need to be specified manually to avoid circular import
        self.minimize = minimize
        return super().run()

    def process_data(self):
        text_msg = "Summary\n"

        # Go through every y variable and find the optimal point
        for y_var in self.dataset.data_vars:
            text_msg += "\n"
            if self.minimize:
                optimum_text = "mimimum"
            else:
                optimum_text = "maximum"
            variable_name = self.dataset[y_var].attrs["name"]
            text_msg += f"{variable_name} {optimum_text}:\n"

            # Find the optimum for each x coordinate
            for x_var in self.dataset.coords:
                if self.minimize:
                    optimum = float(
                        self.dataset[x_var][
                            np.argmin(self.dataset[y_var].values)
                        ].values
                    )
                else:
                    optimum = float(
                        self.dataset[x_var][
                            np.argmax(self.dataset[y_var].values)
                        ].values
                    )

                self.quantities_of_interest[self.dataset[x_var].attrs["name"]] = optimum

                text_msg += format_value_string(
                    self.dataset[x_var].attrs["name"],
                    optimum,
                    end_char="\n",
                    unit=self.dataset[x_var].units,
                )

        self.quantities_of_interest["plot_msg"] = text_msg

    def create_figures(self):
        """
        Plot each of the x variables against each of the y variables
        """

        figs, axs = iteration_plots(self.dataset, self.quantities_of_interest)
        self.figs_mpl.update(figs)
        self.axs_mpl.update(axs)

        figs, axs = x_vs_y_plots(self.dataset, self.quantities_of_interest)
        self.figs_mpl.update(figs)
        self.axs_mpl.update(axs)


def iteration_plots(dataset, quantities_of_interest):
    """
    For every x and y varible, plot a graph of that variable versus the iteration index.
    """

    figs = {}
    axs = {}
    all_variables = list(dataset.coords.items()) + list(dataset.data_vars.items())
    for var, var_vals in all_variables:
        var_name = dataset[var].attrs["name"]

        fig, ax = plt.subplots()
        fig_id = f"Line plot {var_name} vs iteration"

        ax.plot(var_vals, marker=".", linewidth="0.5", markersize="4.5")
        adjust_axeslabels_SI(ax)

        qpl.set_ylabel(ax, var_name, dataset[var].units)
        qpl.set_xlabel(ax, "iteration index")

        qpl.set_suptitle_from_dataset(fig, dataset, f"{var_name} vs iteration number:")

        qpl.plot_textbox(ax, quantities_of_interest["plot_msg"])

        # add the figure and axis to the dicts for saving
        figs[fig_id] = fig
        axs[fig_id] = ax

    return figs, axs


def x_vs_y_plots(dataset, quantities_of_interest):
    """
    Plot every x coordinate against every y variable
    """

    figs = {}
    axs = {}
    for xi, xvals in dataset.coords.items():
        x_name = dataset[xi].attrs["name"]
        for yi, yvals in dataset.data_vars.items():
            y_name = dataset[yi].attrs["name"]
            fig, ax = plt.subplots()
            fig_id = f"Line plot {x_name} vs {y_name}"

            ax.plot(xvals, yvals, marker=".", linewidth="0.5", markersize="4.5")
            adjust_axeslabels_SI(ax)

            qpl.set_xlabel(ax, x_name, dataset[xi].units)
            qpl.set_ylabel(ax, y_name, dataset[yi].units)

            qpl.set_suptitle_from_dataset(
                fig, dataset, f"{x_name} vs {y_name} optimization:"
            )

            qpl.plot_textbox(ax, quantities_of_interest["plot_msg"])

            # add the figure and axis to the dicts for saving
            figs[fig_id] = fig
            axs[fig_id] = ax

    return figs, axs


class InterpolationAnalysis2D(ba.BaseAnalysis):
    """
    An analysis class for the 2D case which generates a 2D interpolating plot.
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
            ax.scatter(xvals0, xvals1, s=2, c="red", alpha=1)

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
