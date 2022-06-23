# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""
Module containing an education example of an analysis subclass.

See :ref:`analysis-framework-tutorial` that guides you through the process of building
this analysis.
"""

import matplotlib.pyplot as plt

import quantify_core.analysis.base_analysis as ba
from quantify_core.analysis.fitting_models import CosineModel
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import (
    adjust_axeslabels_SI,
    format_value_string,
)


class CosineAnalysis(ba.BaseAnalysis):
    """
    Exemplary analysis subclass that fits a cosine to a dataset.
    """

    def process_data(self):
        """
        In some cases, you might need to process the data, e.g., reshape, filter etc.,
        before starting the analysis. This is the method where it should be done.

        See :meth:`~quantify_core.analysis.spectroscopy_analysis.ResonatorSpectroscopyAnalysis.process_data`
        for an implementation example.
        """  # pylint: disable=line-too-long

    def run_fitting(self):
        """
        Fits a :class:`~quantify_core.analysis.fitting_models.CosineModel` to the data.
        """
        # create a fitting model based on a cosine function
        model = CosineModel()
        guess = model.guess(self.dataset.y0.values, x=self.dataset.x0.values)
        result = model.fit(
            self.dataset.y0.values, x=self.dataset.x0.values, params=guess
        )
        self.fit_results.update({"cosine": result})

    def create_figures(self):
        """
        Creates a figure with the data and the fit.
        """
        fig, ax = plt.subplots()
        fig_id = "cos_fit"
        self.figs_mpl.update({fig_id: fig})
        self.axs_mpl.update({fig_id: ax})

        self.dataset.y0.plot(ax=ax, x="x0", marker="o", linestyle="")
        qpl.plot_fit(ax, self.fit_results["cosine"])
        qpl.plot_textbox(ax, ba.wrap_text(self.quantities_of_interest["fit_msg"]))

        adjust_axeslabels_SI(ax)
        qpl.set_suptitle_from_dataset(fig, self.dataset, "x0-y0")
        ax.legend()

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`quantities_of_interest`.
        """
        fit_result = self.fit_results["cosine"]
        fit_warning = ba.check_lmfit(fit_result)

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True
            unit = self.dataset.y0.units
            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$f$", fit_result.params["frequency"], end_char="\n", unit="Hz"
            )
            text_msg += format_value_string(
                r"$A$", fit_result.params["amplitude"], unit=unit
            )
        else:
            text_msg = fit_warning
            self.quantities_of_interest["fit_success"] = False

        # save values and fit uncertainty
        for parameter_name in ["frequency", "amplitude"]:
            self.quantities_of_interest[parameter_name] = ba.lmfit_par_to_ufloat(
                fit_result.params[parameter_name]
            )
        self.quantities_of_interest["fit_msg"] = text_msg
