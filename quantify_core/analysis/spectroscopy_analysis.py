# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import quantify_core.data.handling as dh
from quantify_core.analysis import base_analysis as ba
from quantify_core.analysis import fitting_models as fm
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import format_value_string


class QubitFluxSpectroscopyAnalysis(ba.BaseAnalysis):
    # pylint: disable=line-too-long
    """
    Analysis class for qubit flux spectroscopy.

    .. admonition:: Example

        .. jupyter-execute::

            from quantify_core.analysis.spectroscopy_analysis import QubitFluxSpectroscopyAnalysis
            import quantify_core.data.handling as dh

            # load example data
            test_data_dir = "../tests/test_data"
            dh.set_datadir(test_data_dir)

            # run analysis and plot results
            analysis = (
                QubitFluxSpectroscopyAnalysis(tuid="20230309-235354-353-9c94c5")
                .run()
                .display_figs_mpl()
            )

    """

    # pylint: disable=invalid-name
    # pylint: disable=no-member
    # pylint: arguments-differ

    def process_data(self) -> None:
        """Process the data so that the analysis can make assumptions on the format."""
        ds = dh.to_gridded_dataset(self.dataset)
        self.dataset_processed = xr.Dataset(
            {
                "Magnitude": (("Frequency", "Offset"), ds.y0.data),
                "Phase": (("Frequency", "Offset"), np.mod(ds.y1.data, 360.0)),
            },
            coords={
                "Frequency": ds.x0.data,
                "Offset": ds.x1.data,
            },
        )

        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = ds.y0.units
        self.dataset_processed.Magnitude.attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset_processed.Phase.attrs["name"] = "Phase"
        self.dataset_processed.Phase.attrs["units"] = ds.y1.units
        self.dataset_processed.Phase.attrs["long_name"] = (
            r"Phase, $\angle S_{21}$ mod 360°"
        )

        self.dataset_processed.Frequency.attrs["name"] = "Frequency"
        self.dataset_processed.Frequency.attrs["units"] = ds.x0.units
        self.dataset_processed.Frequency.attrs["long_name"] = (
            "Frequency of excitation pulse"
        )

        self.dataset_processed.Offset.attrs["name"] = "Offset"
        self.dataset_processed.Offset.attrs["units"] = ds.x1.units
        self.dataset_processed.Offset.attrs["long_name"] = "External flux offset"

    def run_fitting(self) -> None:
        """Fits a QuadraticModel model to the frequency response vs. flux offset."""

        # Calculate the mean and standard deviation along each column
        # Calculate the absolute difference from the mean for each element
        # Create a boolean mask where elements are more than 3*sigma away from the mean

        s21 = self.dataset_processed.Magnitude.data * np.exp(
            1j * np.deg2rad(self.dataset_processed.Phase.data)
        )

        column_means = np.mean(s21, axis=0)
        column_stddevs = np.std(s21, axis=0)

        abs_diff = np.abs(s21 - column_means)
        mask = abs_diff > 3 * column_stddevs

        # Find the middle index for each column where the condition is met
        # You can use np.where to find the indices where the condition is True
        row_indices, col_indices = np.where(mask)

        # Split the data into x and y arrays
        x_values = self.dataset_processed.Offset[col_indices].data
        y_values = self.dataset_processed.Frequency[row_indices].data

        # Calculate the unique x values and their corresponding mean y values
        unique_x = np.unique(x_values)
        mean_y = np.array([np.mean(y_values[x_values == x]) for x in unique_x])

        # Fit a model to data
        quad_model = lmfit.models.QuadraticModel()
        guess = quad_model.guess(mean_y, x=unique_x)
        result = quad_model.fit(mean_y, x=unique_x, params=guess)

        self.fit_results.update({"poly2": result})

    def analyze_fit_results(self) -> None:
        """Check the fit success and populate :code:`.quantities_of_interest`."""
        self.quantities_of_interest = {}

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        fit_warning = ba.wrap_text(ba.check_lmfit(self.fit_results["poly2"]))
        if fit_warning is not None:
            self.quantities_of_interest["fit_success"] = False
            self.quantities_of_interest["fit_msg"] = ba.wrap_text(fit_warning)
            return

        # Set fitted parameters as quantities of interest
        for parameter, value in self.fit_results["poly2"].params.items():
            self.quantities_of_interest[parameter] = ba.lmfit_par_to_ufloat(value)

        a = self.quantities_of_interest["a"]
        b = self.quantities_of_interest["b"]
        c = self.quantities_of_interest["c"]

        off_0_unc = -b / (2.0 * a)
        frq_0_unc = a * (off_0_unc**2) + b * off_0_unc + c

        self.quantities_of_interest["sweetspot"] = off_0_unc
        self.quantities_of_interest["sweetspot_freq"] = frq_0_unc

        text_msg = "Summary:\n"
        text_msg += format_value_string(
            "Sweetspot",
            off_0_unc,
            unit=self.dataset_processed.Offset.units,
            end_char="\n",
        )
        text_msg += format_value_string(
            "Freq.",
            frq_0_unc,
            unit="Hz",
            end_char="\n",
        )
        self.quantities_of_interest["fit_success"] = True
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self) -> None:
        """Generate plot of magnitude and phase images, with superposed model fit."""
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        # Plot using xarrays plotting function
        self.dataset_processed.Magnitude.plot(ax=ax)  # type: ignore

        if self.quantities_of_interest["fit_success"]:
            # Interpolate with the model for nice curve
            interp_offsets = np.arange(
                start=self.dataset_processed.Offset.data.min(),
                stop=self.dataset_processed.Offset.data.max(),
                step=np.diff(self.dataset_processed.Offset)[0] / 2.0,
            )

            a = self.quantities_of_interest["a"]
            b = self.quantities_of_interest["b"]
            c = self.quantities_of_interest["c"]

            # Plot model interpolation
            ax.plot(
                interp_offsets,
                a.nominal_value * (interp_offsets**2)
                + b.nominal_value * interp_offsets
                + c.nominal_value,
                color="red",
                ls="-",
            )

            # Plot also the zero which we found
            ax.axvline(
                self.quantities_of_interest["sweetspot"].nominal_value
                - self.quantities_of_interest["sweetspot"].std_dev,
                color="red",
                ls="--",
                alpha=0.5,
            )
            ax.axvline(
                self.quantities_of_interest["sweetspot"].nominal_value,
                color="red",
                ls="-",
            )
            ax.axvline(
                self.quantities_of_interest["sweetspot"].nominal_value
                + self.quantities_of_interest["sweetspot"].std_dev,
                color="red",
                ls="--",
                alpha=0.5,
            )

        # Put a text box summarizing the QOI
        qpl.plot_textbox(
            ax=ax,
            transform=ax.transAxes,
            text=self.quantities_of_interest["fit_msg"],
            x=1.3,
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset)  # type: ignore
        ax.set_title(self.dataset_processed.Magnitude.attrs["name"])

        fig.tight_layout()  # type: ignore

        self.figs_mpl["qfs"] = fig  # type: ignore
        self.axs_mpl["qfs"] = (ax,)  # type: ignore


# Custom analysis class for QubitSpectroscopy
class QubitSpectroscopyAnalysis(ba.BaseAnalysis):
    """
    Analysis for a qubit spectroscopy experiment.

    Fits a Lorentzian function to qubit spectroscopy
    data and finds the 0-1 transistion frequency.
    """

    # pylint: disable=invalid-name
    # pylint: disable=no-member

    def process_data(self) -> None:
        """Populate the :code:`.dataset_processed`."""
        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.

        self.dataset_processed["Magnitude"] = self.dataset.y0
        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.Magnitude.attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset_processed["x0"] = self.dataset.x0
        self.dataset_processed = self.dataset_processed.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def run_fitting(self) -> None:
        """Fit a Lorentzian function to the data."""
        mod = fm.LorentzianModel()

        magnitude = self.dataset_processed["Magnitude"].values
        frequency = self.dataset_processed.x0.values
        guess = mod.guess(magnitude, x=frequency)
        fit_result = mod.fit(magnitude, params=guess, x=frequency)

        self.fit_results.update({"Lorentzian_peak": fit_result})

    def analyze_fit_results(self) -> None:
        """Check fit success and populates :code:`.quantities_of_interest`."""
        fit_result = self.fit_results["Lorentzian_peak"]
        fit_warning = ba.wrap_text(ba.check_lmfit(fit_result))

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True

            text_msg = "Summary\n"
            text_msg += format_value_string(
                "Frequency 0-1",
                fit_result.params["x0"],
                unit="Hz",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Peak width",
                fit_result.params["width"],
                unit="Hz",
                end_char="\n",
            )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["frequency_01"] = ba.lmfit_par_to_ufloat(
            fit_result.params["x0"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self) -> None:
        """Create qubit spectroscopy figure."""
        fig_id = "QubitSpectroscopyAnalysis"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig  # type: ignore
        self.axs_mpl[fig_id] = ax  # type: ignore

        self.dataset_processed.Magnitude.plot(ax=ax, marker=".", linestyle="")

        qpl.plot_fit(
            ax=ax,
            fit_res=self.fit_results["Lorentzian_peak"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset, "S21")  # type: ignore

        fig.tight_layout()  # type: ignore
        # Add a textbox with the fit_message
        qpl.plot_textbox(ax, self.quantities_of_interest["fit_msg"])  # type: ignore


class ResonatorSpectroscopyAnalysis(ba.BaseAnalysis):
    """
    Analysis for a spectroscopy experiment of a hanger resonator.
    """

    def process_data(self):
        """
        Verifies that the data is measured as magnitude and phase and casts it to
        a dataset of complex valued transmission :math:`S_{21}`.
        """

        # y0 = amplitude, no check for the amplitude unit as the name/label is
        # often different.
        # y1 = phase in deg, this unit should always be correct
        assert self.dataset.y1.units == "deg"

        S21 = self.dataset.y0 * np.cos(
            np.deg2rad(self.dataset.y1)
        ) + 1j * self.dataset.y0 * np.sin(np.deg2rad(self.dataset.y1))
        self.dataset_processed["S21"] = S21
        self.dataset_processed.S21.attrs["name"] = "S21"
        self.dataset_processed.S21.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.S21.attrs["long_name"] = "Transmission, $S_{21}$"

        self.dataset_processed["x0"] = self.dataset.x0
        self.dataset_processed = self.dataset_processed.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a :class:`~quantify_core.analysis.fitting_models.ResonatorModel` to the data.
        """

        model = fm.ResonatorModel()

        S21 = self.dataset_processed.S21.values
        frequency = self.dataset_processed.x0.values
        guess = model.guess(S21, f=frequency)

        fit_result = model.fit(S21, params=guess, f=frequency)

        self.fit_results.update({"hanger_func_complex_SI": fit_result})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """

        fit_result = self.fit_results["hanger_func_complex_SI"]
        fit_warning = ba.wrap_text(ba.check_lmfit(fit_result))

        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True
            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$Q_I$", fit_result.params["Qi"], unit="SI_PREFIX_ONLY", end_char="\n"
            )
            text_msg += format_value_string(
                r"$Q_C$", fit_result.params["Qc"], unit="SI_PREFIX_ONLY", end_char="\n"
            )
            text_msg += format_value_string(
                r"$f_{res}$", fit_result.params["fr"], unit="Hz"
            )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        for parameter in ["Qi", "Qe", "Ql", "Qc", "fr"]:
            self.quantities_of_interest[parameter] = ba.lmfit_par_to_ufloat(
                fit_result.params[parameter]
            )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """
        Plots the measured and fitted transmission :math:`S_{21}` as the I and Q
        component vs frequency, the magnitude and phase vs frequency,
        and on the complex I,Q plane.
        """
        self._create_fig_s21_real_imag()
        self._create_fig_s21_magn_phase()
        self._create_fig_s21_complex()

    def _create_fig_s21_real_imag(self):
        fig_id = "S21-RealImag"
        fig, axs = plt.subplots(2, 1, sharex=True)
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id + "_Re"] = axs[0]
        self.axs_mpl[fig_id + "_Im"] = axs[1]

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs[0], self.quantities_of_interest["fit_msg"])

        self.dataset_processed.S21.real.plot(ax=axs[0], marker=".")
        self.dataset_processed.S21.imag.plot(ax=axs[1], marker=".")

        qpl.plot_fit(
            ax=axs[0],
            fit_res=self.fit_results["hanger_func_complex_SI"],
            plot_init=False,
            range_casting="real",
        )

        qpl.plot_fit(
            ax=axs[1],
            fit_res=self.fit_results["hanger_func_complex_SI"],
            plot_init=False,
            range_casting="imag",
        )

        qpl.set_ylabel(r"Re$(S_{21})$", self.dataset_processed.S21.units, axs[0])
        qpl.set_ylabel(r"Im$(S_{21})$", self.dataset_processed.S21.units, axs[1])
        axs[0].set_xlabel("")
        qpl.set_xlabel(
            self.dataset_processed.x0.long_name, self.dataset_processed.x0.units, axs[1]
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset, "S21")

    def _create_fig_s21_magn_phase(self):
        fig_id = "S21-MagnPhase"
        fig, axs = plt.subplots(2, 1, sharex=True)
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id + "_Magn"] = axs[0]
        self.axs_mpl[fig_id + "_Phase"] = axs[1]

        # Add a textbox with the fit_message
        qpl.plot_textbox(axs[0], self.quantities_of_interest["fit_msg"])

        axs[0].plot(
            self.dataset_processed.x0, np.abs(self.dataset_processed.S21), marker="."
        )
        axs[1].plot(
            self.dataset_processed.x0,
            np.angle(self.dataset_processed.S21, deg=True),
            marker=".",
        )

        qpl.plot_fit(
            ax=axs[0],
            fit_res=self.fit_results["hanger_func_complex_SI"],
            plot_init=False,
            range_casting="abs",
        )

        qpl.plot_fit(
            ax=axs[1],
            fit_res=self.fit_results["hanger_func_complex_SI"],
            plot_init=False,
            range_casting="angle",
        )

        qpl.set_ylabel(r"$|S_{21}|$", self.dataset_processed.S21.units, axs[0])
        qpl.set_ylabel(r"$\angle S_{21}$", "deg", axs[1])
        axs[0].set_xlabel("")
        qpl.set_xlabel(
            self.dataset_processed.x0.long_name, self.dataset_processed.x0.units, axs[1]
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset, "S21")

    def _create_fig_s21_complex(self):
        fig_id = "S21-complex"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        qpl.plot_textbox(ax, self.quantities_of_interest["fit_msg"])

        ax.plot(
            self.dataset_processed.S21.real, self.dataset_processed.S21.imag, marker="."
        )

        qpl.plot_fit_complex_plane(
            ax=ax,
            fit_res=self.fit_results["hanger_func_complex_SI"],
            plot_init=False,
        )

        qpl.set_xlabel(r"Re$(S_{21})$", self.dataset_processed.S21.units, ax)
        qpl.set_ylabel(r"Im$(S_{21})$", self.dataset_processed.S21.units, ax)

        qpl.set_suptitle_from_dataset(fig, self.dataset, "S21")


class ResonatorFluxSpectroscopyAnalysis(ba.BaseAnalysis):
    """
    Analysis class for resonator flux spectroscopy.

    .. admonition:: Example

        .. jupyter-execute::

            from quantify_core.analysis.spectroscopy_analysis import (
                ResonatorFluxSpectroscopyAnalysis
            )
            import quantify_core.data.handling as dh

            # load example data
            test_data_dir = "../tests/test_data"
            dh.set_datadir(test_data_dir)

            # run analysis and plot results
            analysis = (
                ResonatorFluxSpectroscopyAnalysis(tuid="20230308-235659-059-cf471e")
                .run()
                .display_figs_mpl()
            )

    """

    # pylint: disable=no-member
    # pylint: disable=broad-exception-caught
    # pylint: disable=attribute-defined-outside-init

    def process_data(self) -> None:
        """Process the data so that the analysis can make assumptions on the format."""
        ds = dh.to_gridded_dataset(self.dataset)
        self.dataset_processed = xr.Dataset(
            {
                "Magnitude": (("Frequency", "Offset"), ds.y0.data),
                "Phase": (("Frequency", "Offset"), np.mod(ds.y1.data, 360.0)),
            },
            coords={
                "Frequency": ds.x0.data,
                "Offset": ds.x1.data,
            },
        )

        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = ds.y0.units
        self.dataset_processed.Magnitude.attrs["long_name"] = "Magnitude, $|S_{21}|$"

        self.dataset_processed.Phase.attrs["name"] = "Phase"
        self.dataset_processed.Phase.attrs["units"] = ds.y1.units
        self.dataset_processed.Phase.attrs["long_name"] = (
            r"Phase, $\angle S_{21}$ mod 360°"
        )

        self.dataset_processed.Frequency.attrs["name"] = "Frequency"
        self.dataset_processed.Frequency.attrs["units"] = ds.x0.units
        self.dataset_processed.Frequency.attrs["long_name"] = (
            "Frequency of readout pulse"
        )

        self.dataset_processed.Offset.attrs["name"] = "Offset"
        self.dataset_processed.Offset.attrs["units"] = ds.x1.units
        self.dataset_processed.Offset.attrs["long_name"] = "External flux offset"

    def run_fitting(self) -> None:
        """Fits a sinusoidal model to the frequency response vs. flux offset."""

        # Calculate the mean and standard deviation along each column
        # Calculate the absolute difference from the mean for each element
        # Create a boolean mask where elements are more than 3*sigma away from the mean

        try:
            column_means = np.mean(self.dataset_processed.Magnitude.data, axis=0)
            abs_diff = np.abs(self.dataset_processed.Magnitude.data - column_means)
            mask = abs_diff > 3 * np.std(self.dataset_processed.Magnitude.data, axis=0)

            # Find the middle index for each column where the condition is met
            # You can use np.where to find the indices where the condition is True
            row_indices, col_indices = np.where(mask)

            # Split the data into x and y arrays
            x_values = self.dataset_processed.Offset[col_indices].data
            y_values = self.dataset_processed.Frequency[row_indices].data

            # Calculate the unique x values and their corresponding mean y values
            unique_x = np.unique(x_values)
            mean_y = np.array([np.mean(y_values[x_values == x]) for x in unique_x])

            # Fit a sinusoidal model to centered data
            center = mean_y.mean()
            sine_model = lmfit.models.SineModel()
            guess = sine_model.guess(mean_y - center, x=unique_x)
            result = sine_model.fit(mean_y - center, x=unique_x, params=guess)

            # Also add the offset with estimate standard error
            result.params.add("center", value=center, vary=False)
            result.params["center"].stderr = mean_y.std() / np.sqrt(len(mean_y))

            self.fit_results.update({"sin": result})
            self._fit_success = bool(result.success)

        except Exception as fit_error:
            # we use these private members to avoid errors during fit failure
            self._fit_error = "Error during fit:\n" + str(fit_error)
            self._fit_success = False

    def analyze_fit_results(self) -> None:
        """Check the fit success and populate :code:`.quantities_of_interest`."""
        self.quantities_of_interest = {"fit_success": self._fit_success}

        if not self._fit_success:
            self.quantities_of_interest["fit_msg"] = self._fit_error
            return

        # Set fitted sinus parameters as quantities of interest
        for parameter, value in self.fit_results["sin"].params.items():
            self.quantities_of_interest[parameter] = ba.lmfit_par_to_ufloat(value)

        # Scale frequency of the sinusoid
        self.quantities_of_interest["frequency"] /= 2.0 * np.pi

        text_msg = "Summary:\n"
        for parameter, value in self.fit_results["sin"].params.items():
            # Build text box
            text_msg += format_value_string(
                parameter,
                self.quantities_of_interest[parameter],
                unit=dict(
                    amplitude=None,
                    frequency=self.dataset_processed.Offset.units,
                    shift=self.dataset_processed.Offset.units,
                    center=self.dataset_processed.Frequency.units,
                )[parameter],
                end_char="\n",
            )

        # Find some zeros of the derivative of the fitted sinusoidal model
        root_indices = np.arange(-20, 20, 1)
        roots = (
            -2.0 * self.quantities_of_interest["shift"]
            - 2 * np.pi * root_indices
            + np.pi
        ) / (4.0 * np.pi * self.quantities_of_interest["frequency"])

        # Don't extrapolate sweetspots
        roots = np.asarray(
            sorted(
                filter(
                    lambda root: (
                        (root.nominal_value <= self.dataset_processed.Offset.max())
                        and (root.nominal_value >= self.dataset_processed.Offset.min())
                    ),
                    roots,
                )
            )
        )

        # Save all visible sweetspots, since we have computed them
        text_msg += "\nSweetspots:\n"
        for root_index, root in enumerate(roots):
            # Save sweetspot
            self.quantities_of_interest[f"sweetspot_{root_index}"] = root

            # Build text box at the same time
            text_msg += format_value_string(
                f"sweetspot {root_index}",
                root,
                unit=self.dataset_processed.Offset.units,
                end_char="\n",
            )

        self.quantities_of_interest["fit_msg"] = text_msg[:-1]

    def create_figures(self) -> None:
        """Generate plot of magnitude and phase images, with superposed model fit."""

        if not self.quantities_of_interest["fit_success"]:
            print(self.quantities_of_interest["fit_msg"])
            return

        fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

        # Plot using xarrays plotting function
        self.dataset_processed.Magnitude.plot(ax=ax[0])  # type: ignore
        self.dataset_processed.Phase.plot(ax=ax[1], cmap="jet")

        if self.quantities_of_interest["fit_success"]:
            # Interpolate with the model for nice curve
            interp_offsets = np.arange(
                start=self.dataset_processed.Offset.data.min(),
                stop=self.dataset_processed.Offset.data.max(),
                step=np.diff(self.dataset_processed.Offset)[0] / 2.0,
            )

            # Plot model interpolation
            ax[0].plot(
                interp_offsets,
                fm.cos_func(
                    x=interp_offsets,
                    frequency=self.quantities_of_interest["frequency"].nominal_value,
                    amplitude=self.quantities_of_interest["amplitude"].nominal_value,
                    offset=self.quantities_of_interest["center"].nominal_value,
                    phase=self.quantities_of_interest["shift"].nominal_value
                    - np.pi / 2,
                ),
                color="red",
            )

            # Plot also the zeros which we found that are visible
            for _, root_ufloat in filter(
                lambda item: (item[0].startswith("sweetspot_")),
                self.quantities_of_interest.items(),
            ):
                ax[0].axvline(
                    root_ufloat.nominal_value - root_ufloat.std_dev,
                    color="red",
                    ls="--",
                    alpha=0.5,
                )
                ax[0].axvline(
                    root_ufloat.nominal_value,
                    color="red",
                    ls="-",
                )
                ax[0].axvline(
                    root_ufloat.nominal_value + root_ufloat.std_dev,
                    color="red",
                    ls="--",
                    alpha=0.5,
                )

        # Put a text box summarizing the QOI
        qpl.plot_textbox(
            ax=ax[1],
            transform=ax[1].transAxes,
            text=self.quantities_of_interest["fit_msg"],
            x=1.4,
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset)  # type: ignore
        ax[0].set_title(self.dataset_processed.Magnitude.attrs["name"])
        ax[1].set_title(self.dataset_processed.Phase.attrs["name"])

        fig.tight_layout()  # type: ignore

        self.figs_mpl["rfs"] = fig  # type: ignore
        self.axs_mpl["rfs"] = (ax,)  # type: ignore
