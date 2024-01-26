# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing an analysis class for two-state readout calibration."""
from __future__ import annotations

import math

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import quantify_core.data.handling as dh
from quantify_core.analysis import base_analysis as ba
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import format_value_string


class ReadoutCalibrationAnalysis(ba.BaseAnalysis):
    # pylint: disable=line-too-long
    """
    Find threshold and angle which discriminates qubit state.


    .. admonition:: Example

        .. jupyter-execute::

            import os


            import quantify_core.data.handling as dh
            from quantify_core.analysis.readout_calibration_analysis import (
                ReadoutCalibrationAnalysis,
            )

            # load example data
            test_data_dir = "../tests/test_data"
            dh.set_datadir(test_data_dir)
            ReadoutCalibrationAnalysis(tuid="20230509-152441-841-faef49").run().display_figs_mpl()

    """

    # pylint: disable=no-member
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=broad-exception-caught

    def process_data(self) -> None:
        """Process the data so that the analysis can make assumptions on the format."""
        self.dataset_processed = dh.to_gridded_dataset(self.dataset)

    def run_fitting(self) -> None:
        """Fit a state discriminator to the readout calibration data."""
        points = np.zeros((self.dataset_processed.x0.data.size, 2), dtype=float)
        points[:, 0] = self.dataset_processed.y0.data  # real (I)
        points[:, 1] = self.dataset_processed.y1.data  # imag (Q)

        # make a dummy model just to store the params with analysis infrastructure
        lin_model = lmfit.models.LinearModel()
        result = lin_model.fit([0, 1], x=[0, 1])
        # overwrite parameters
        result.params = lmfit.Parameters()

        try:
            self._lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
            self._lda.fit(points, self.dataset_processed.x0.data)

            intercept = -self._lda.intercept_[0] / self._lda.coef_[0][1]
            slope = -self._lda.coef_[0][0] / self._lda.coef_[0][1]

            result.params.add("intercept", value=intercept, vary=False)
            result.params.add("slope", value=slope, vary=False)
            result.params.add(
                "acq_threshold",
                # compute distance from origin to line
                value=abs(intercept) / math.sqrt(slope**2 + 1),
                vary=False,
            )
            result.params.add(
                "acq_rotation_rad",
                # compute CW angle from y-axis
                value=-math.pi / 2 - math.atan(slope),
                vary=False,
            )

            result.success = True

            result.errorbars = {  # type: ignore
                "intercept": False,
                "slope": False,
                "acq_threshold": False,
                "acq_rotation_rad": False,
            }

        except Exception as lda_error:
            result.success = False
            result.message = "Error during fit:\n" + str(lda_error)  # type: ignore

        self.fit_results.update({"linear_discriminator": result})

    def _get_points(self) -> tuple:
        points = np.zeros((self.dataset_processed.x0.data.size, 2), dtype=float)
        points[:, 0] = self.dataset_processed.y0.data  # real (I)
        points[:, 1] = self.dataset_processed.y1.data  # imag (Q)

        y_pred = self._lda.predict(points)
        true_pos = self.dataset_processed.x0.data == y_pred
        true_pos0, true_pos1 = (
            true_pos[self.dataset_processed.x0.data == 0],
            true_pos[self.dataset_processed.x0.data == 1],
        )
        x0, x1 = (
            points[self.dataset_processed.x0.data == 0],
            points[self.dataset_processed.x0.data == 1],
        )

        return (x0[true_pos0], x0[~true_pos0], x1[true_pos1], x1[~true_pos1])

    def analyze_fit_results(self) -> None:
        """Check the fit success and populate :code:`.quantities_of_interest`."""
        self.quantities_of_interest = {
            "fit_success": self.fit_results["linear_discriminator"].success
        }

        if not self.quantities_of_interest["fit_success"]:
            self.quantities_of_interest["fit_msg"] = self.fit_results[
                "linear_discriminator"
            ].message
            return

        # Set fitted parameters as quantities of interest
        for parameter, value in self.fit_results["linear_discriminator"].params.items():
            self.quantities_of_interest[parameter] = ba.lmfit_par_to_ufloat(value)

        # Get centroid from LDA model
        self.quantities_of_interest["avg_ket0"] = (
            self._lda.means_[0][0] + self._lda.means_[0][1] * 1j
        )
        self.quantities_of_interest["avg_ket1"] = (
            self._lda.means_[1][0] + self._lda.means_[1][1] * 1j
        )

        # Compute fidelity estimates
        (x0_tp, x0_fp, x1_tp, x1_fp) = self._get_points()
        self.quantities_of_interest["fid_est_0"] = x0_tp.shape[0] / (
            x0_tp.shape[0] + x0_fp.shape[0]
        )
        self.quantities_of_interest["fid_est_1"] = x1_tp.shape[0] / (
            x1_tp.shape[0] + x1_fp.shape[0]
        )

        text_msg = "Sample means:\n"
        text_msg += format_value_string(
            r"$|0\rangle$",
            self.quantities_of_interest["avg_ket0"],
            unit=self.dataset_processed.y0.units,
            end_char="\n",
        )
        text_msg += format_value_string(
            r"$|1\rangle$",
            self.quantities_of_interest["avg_ket1"],
            unit=self.dataset_processed.y0.units,
            end_char="\n\n",
        )
        text_msg += "Discriminator:\n"
        text_msg += format_value_string(
            "Threshold",
            self.quantities_of_interest["acq_threshold"],
            unit=self.dataset_processed.y0.units,
            end_char="\n",
        )
        text_msg += format_value_string(
            r"$\theta$ from y-axis",
            self.quantities_of_interest["acq_rotation_rad"],
            unit="rad",
            end_char="\n\n",
        )
        text_msg += "Fidelity estimates:\n"
        text_msg += format_value_string(
            r"$|0\rangle$",
            self.quantities_of_interest["fid_est_0"] * 100,
            unit="%",
            end_char="\n",
        )
        text_msg += format_value_string(
            r"$|1\rangle$",
            self.quantities_of_interest["fid_est_1"] * 100,
            unit="%",
            end_char="\n",
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self) -> None:
        """
        Generate figures of interest.

        matplotlib figures and axes objects are added to
        the ``.figs_mpl`` and ``.axs_mpl`` dictionaries, respectively.
        """
        if not self.quantities_of_interest["fit_success"]:
            print(self.quantities_of_interest["fit_msg"])
            return

        fig, ax = plt.subplots(1, 1, figsize=(11, 6))

        # plot grid for visual help
        ax.grid(alpha=1 / 30, color="black")
        ax.axhline(0.0, alpha=1 / 15, color="black")
        ax.axvline(0.0, alpha=1 / 15, color="black")
        ax.set_xlabel(f"I value [{self.dataset_processed.y0.units}]")
        ax.set_ylabel(f"Q value [{self.dataset_processed.y1.units}]")

        # Put a text box summarizing the QOI
        qpl.plot_textbox(
            ax=ax,
            transform=ax.transAxes,
            text=self.quantities_of_interest["fit_msg"],
        )

        (x0_tp, x0_fp, x1_tp, x1_fp) = self._get_points()

        # class 0: dots
        ax.scatter(x0_tp[:, 0], x0_tp[:, 1], marker=".", color="red")
        ax.scatter(
            x0_fp[:, 0], x0_fp[:, 1], marker="x", s=20, color="#990000"
        )  # dark red

        # class 1: dots
        ax.scatter(x1_tp[:, 0], x1_tp[:, 1], marker=".", color="blue")
        ax.scatter(
            x1_fp[:, 0], x1_fp[:, 1], marker="x", s=20, color="#000099"
        )  # dark blue

        xdef = np.linspace(
            self.dataset_processed.y0.data.min(),
            self.dataset_processed.y0.data.max(),
            200,
        )

        ax.plot(
            xdef,
            self.quantities_of_interest["slope"].nominal_value * xdef
            + self.quantities_of_interest["intercept"].nominal_value,
            color="black",
        )

        # means
        ax.plot(
            self._lda.means_[0][0],
            self._lda.means_[0][1],
            ".",
            color="red",
            markersize=30,
            mew=3,
            markeredgecolor="black",
        )
        ax.plot(
            self._lda.means_[1][0],
            self._lda.means_[1][1],
            ".",
            color="blue",
            markersize=30,
            mew=3,
            markeredgecolor="black",
        )

        ax.axis(
            [
                None,
                None,
                1.05 * self.dataset_processed.y1.min(),
                1.05 * self.dataset_processed.y1.max(),
            ]
        )
        qpl.set_suptitle_from_dataset(fig, self.dataset)  # type: ignore

        fig.tight_layout()

        self.figs_mpl["rca"] = fig  # type: ignore
        self.axs_mpl["rca"] = ax  # type: ignore
