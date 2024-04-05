# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing an analysis class for the conditional oscillation experiment."""


import lmfit
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray

import quantify_core.data.handling as dh
from quantify_core.analysis import base_analysis as ba
from quantify_core.analysis import fitting_models as fm
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import format_value_string


def _add_center(
    param_name: str, data: NDArray, params: lmfit.parameter.Parameters
) -> None:
    params.add(param_name, value=data.mean(), vary=False)
    params[param_name].stderr = data.std() / np.sqrt(data.shape[0])


def _center_and_fit_sinus(y: NDArray, x: NDArray) -> lmfit.model.ModelResult:
    # Center the data
    y_center = y.mean()
    y_centered = y - y_center

    # Fit a sinusoidal model to centered data (ON)
    sine_model = lmfit.models.SineModel()
    guess = sine_model.guess(y_centered, x=x)
    result = sine_model.fit(y_centered, x=x, params=guess)

    # Also add the offset with estimate standard error
    _add_center("center", data=y, params=result.params)

    return result


class ConditionalOscillationAnalysis(ba.BaseAnalysis):
    """
    Analysis class for the conditional oscillation experiment.

    For a reference to the conditional oscillation experiment, please
    see section D in the supplemental material of
    this paper: https://arxiv.org/abs/1903.02492

    .. admonition:: Example

        .. jupyter-execute::

            from quantify_core.analysis.conditional_oscillation_analysis import (
                ConditionalOscillationAnalysis
            )
            import quantify_core.data.handling as dh

            # load example data
            test_data_dir = "../tests/test_data"
            dh.set_datadir(test_data_dir)

            # run analysis and plot results
            analysis = (
                ConditionalOscillationAnalysis(tuid="20230509-165523-132-dcfea7")
                .run()
                .display_figs_mpl()
            )

    """

    # pylint: disable=no-member
    # pylint: disable=invalid-name
    # pylint: disable=broad-exception-caught
    # pylint: disable=attribute-defined-outside-init

    def process_data(self) -> None:
        """Process the data so that the analysis can make assumptions on the format."""
        ds = dh.to_gridded_dataset(self.dataset)

        # y0 : Magnitude    (OFF)   low-frequency qubit
        # y1 : Phase        (OFF)   low-frequency qubit
        # y2 : Magnitude    (OFF)   high-frequency qubit
        # y3 : Phase        (OFF)   high-frequency qubit
        # y4 : Magnitude    (ON)    low-frequency qubit
        # y5 : Phase        (ON)    low-frequency qubit
        # y6 : Magnitude    (ON)    high-frequency qubit
        # y7 : Phase        (ON)    high-frequency qubit

        # In general this measurement can be 2D or 3D, but this analysis
        # class strictly deals with the 1D case.
        self.dataset_processed = xr.Dataset(
            {
                "mag_lf_off": (("phi",), ds.y0.data.flat),
                "mag_hf_off": (("phi",), ds.y2.data.flat),
                "mag_lf_on": (("phi",), ds.y4.data.flat),
                "mag_hf_on": (("phi",), ds.y6.data.flat),
            },
            coords={"phi": ds.x0.data.flat},
        )

        # Copy units
        self.dataset_processed.phi.attrs["units"] = self.dataset.x0.units
        for data_var, y in zip(
            ["mag_lf_off", "mag_hf_off", "mag_lf_on", "mag_hf_on"],
            ["y0", "y2", "y4", "y6"],
        ):
            self.dataset_processed.data_vars[data_var].attrs["units"] = (
                self.dataset.data_vars[y].units
            )

    def run_fitting(self) -> None:
        """Fit two sinusoidal model to the off/on experiments."""
        self.fit_results = {}
        try:
            result_off = _center_and_fit_sinus(
                self.dataset_processed.mag_lf_off.data,
                self.dataset_processed.phi.data,
            )
            result_on = _center_and_fit_sinus(
                self.dataset_processed.mag_lf_on.data,
                self.dataset_processed.phi.data,
            )
            self.fit_results.update({"sin_off": result_off})
            self.fit_results.update({"sin_on": result_on})
            self._fit_success = bool(result_off.success) and bool(result_on.success)

        except Exception as fit_error:
            # we use these private members to avoid errors during fit failure
            self._fit_error = "Error during fit:\n" + str(fit_error)
            self._fit_success = False

    def analyze_fit_results(self) -> None:
        """Check fit success and populates :code:`.quantities_of_interest`."""
        self.quantities_of_interest: dict = {"fit_success": self._fit_success}

        if not self._fit_success:
            self.quantities_of_interest["fit_msg"] = self._fit_error
            return

        def _store_params(data: NDArray, mode: str) -> None:
            result = self.fit_results["sin_" + str(mode)]

            _add_center("offset", data, result.params)

            # Store all parameters as quantities of interest
            for p, value in result.params.items():
                self.quantities_of_interest[str(mode) + "_" + str(p)] = (
                    ba.lmfit_par_to_ufloat(value)
                )

        # Extract leakage estimator
        # and store in parameters struct of model
        _add_center(
            "hf_level",
            self.dataset_processed.mag_hf_on.data,
            self.fit_results["sin_on"].params,
        )
        _add_center(
            "hf_level",
            self.dataset_processed.mag_hf_off.data,
            self.fit_results["sin_off"].params,
        )

        _store_params(self.dataset_processed.mag_lf_off.data, mode="off")
        _store_params(self.dataset_processed.mag_lf_on.data, mode="on")

        # Extract conditional phase in degrees
        self.quantities_of_interest["phi_2q_deg"] = (
            self.quantities_of_interest["on_shift"]
            - self.quantities_of_interest["off_shift"]
        )
        self.quantities_of_interest["phi_2q_deg"] *= 180.0 / np.pi  # convert to degrees
        self.quantities_of_interest["phi_2q_deg"] = np.mod(
            self.quantities_of_interest["phi_2q_deg"], 360.0
        )

        self.quantities_of_interest["leak"] = ba.lmfit_par_to_ufloat(
            self.fit_results["sin_on"].params["hf_level"]
        ) - ba.lmfit_par_to_ufloat(
            self.fit_results["sin_off"].params["hf_level"]
        )  # type: ignore

        # Make fit message
        text_msg = "Summary:\n\n"
        text_msg += format_value_string(
            r"$\phi_{2q}$",
            self.quantities_of_interest["phi_2q_deg"],
            unit=self.dataset_processed.phi.units,
            end_char="\n",
        )
        text_msg += format_value_string(
            "Leakage",
            self.quantities_of_interest["leak"],
            unit=self.dataset_processed.mag_lf_off.units,
            end_char="\n",
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self) -> None:
        """
        Generate figures of interest.

        matplolib figures and axes objects are added to
        the .figs_mpl and .axs_mpl dictionaries., respectively.
        """
        if not self.quantities_of_interest["fit_success"]:
            print(self.quantities_of_interest["fit_msg"])
            return

        fig, axs = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

        # plot lf measurement for conditional phase
        self.dataset_processed.mag_lf_off.plot(
            ax=axs[0], marker=".", label="low freq. qb off", color="C0"
        )
        self.dataset_processed.mag_lf_on.plot(
            ax=axs[0], marker=".", label="low freq. qb on", color="C1"
        )

        # plot hf measurement for leakage estimator
        self.dataset_processed.mag_hf_off.plot(
            ax=axs[1], marker=".", label="high freq. qb off", color="C0"
        )
        self.dataset_processed.mag_hf_on.plot(
            ax=axs[1], marker=".", label="high freq. qb on", color="C1"
        )

        # Interpolate with the model for nice curve
        interp_x = np.arange(
            start=self.dataset_processed.phi.data.min(),
            stop=self.dataset_processed.phi.data.max(),
            step=np.diff(self.dataset_processed.phi.data)[0] / 2.0,
        )

        for var in ("on", "off"):
            axs[0].plot(
                interp_x,
                fm.cos_func(
                    x=interp_x,
                    frequency=self.quantities_of_interest[
                        var + "_frequency"
                    ].nominal_value
                    / (2 * np.pi),
                    amplitude=self.quantities_of_interest[
                        var + "_amplitude"
                    ].nominal_value,
                    offset=self.quantities_of_interest[var + "_offset"].nominal_value,
                    phase=self.quantities_of_interest[var + "_shift"].nominal_value
                    - np.pi / 2,
                ),
                color="red",
                ls="-",
            )

        axs[1].axhline(
            self.dataset_processed.mag_hf_off.data.mean(), color="red", ls="--"
        )
        axs[1].axhline(
            self.dataset_processed.mag_hf_on.data.mean(), color="red", ls="--"
        )

        for ax in axs:
            ax.grid(alpha=1 / 25, color="black")
            ax.legend(loc="upper left")

        qpl.set_suptitle_from_dataset(fig, self.dataset)  # type: ignore
        fig.tight_layout()  # type: ignore

        # Put a text box summarizing the QOI in the middle of the figure
        qpl.plot_textbox(
            ax=axs[1],
            text=self.quantities_of_interest["fit_msg"],
            # x=1.34, # make some space for colorbar
        )

        self.figs_mpl["figure"] = fig  # type: ignore
        self.axs_mpl["axis"] = axs  # type: ignore
