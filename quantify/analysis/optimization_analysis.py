# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import matplotlib.pyplot as plt
from quantify.analysis import base_analysis as ba
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
        minimize
            Boolean which determines whether to report the minimum or the maximum.
            True for minimize.
            False for maximize.

        Returns
        -------
        :class:`~quantify.analysis.optimization_analysis.OptimizationAnalysis`:
            The instance of this analysis.

        """  # NB the return type need to be specified manually to avoid circular import
        self.minimize = minimize
        return super().run()

    def process_data(self):
        """
        Finds the optimal (minimum or maximum) for y0 and saves the xi and y0
        values in the :code:`quantities_of_interest`.
        """
        text_msg = "Summary\n"

        arg_optimum_func = np.argmin if self.minimize else np.argmax
        optimum_func = np.min if self.minimize else np.max
        optimum_text = "mimimum" if self.minimize else "maximum"

        # Go through every y variable and find the optimal point
        y_var_name = "y0"

        text_msg += "\n"
        variable_name = self.dataset[y_var_name].attrs["name"]
        text_msg += f"{variable_name} {optimum_text}:\n"

        # Find the optimum for each x coordinate
        for x_var in self.dataset.coords:
            optimum = float(
                self.dataset[x_var][
                    arg_optimum_func(self.dataset[y_var_name].values)
                ].values
            )

            self.quantities_of_interest[self.dataset[x_var].attrs["name"]] = optimum

            text_msg += format_value_string(
                self.dataset[x_var].attrs["name"],
                optimum,
                end_char="\n",
                unit=self.dataset[x_var].units,
            )

        # Find the corresponding optimal y value
        optimum = float(optimum_func(self.dataset[y_var_name].values))

        self.quantities_of_interest[self.dataset[y_var_name].attrs["name"]] = optimum

        text_msg += format_value_string(
            self.dataset[y_var_name].attrs["name"],
            optimum,
            end_char="\n",
            unit=self.dataset[y_var_name].units,
        )

        self.quantities_of_interest["plot_msg"] = text_msg

    def create_figures(self):
        """
        Plot each of the x variables against each of the y variables.
        """

        figs, axs = iteration_plots(self.dataset, self.quantities_of_interest)
        self.figs_mpl.update(figs)
        self.axs_mpl.update(axs)


def iteration_plots(dataset, quantities_of_interest):
    """
    For every x and y variable, plot a graph of that variable vs the iteration index.
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
