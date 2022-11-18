# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing analyses for common single qubit timedomain experiments."""
from __future__ import annotations

from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from quantify_core.analysis import base_analysis as ba
from quantify_core.analysis import fitting_models as fm
from quantify_core.analysis.calibration import (
    has_calibration_points,
    rotate_to_calibrated_axis,
)
from quantify_core.data.types import TUID
from quantify_core.visualization.mpl_plotting import (
    plot_fit,
    plot_textbox,
    set_suptitle_from_dataset,
    set_xlabel,
    set_ylabel,
)
from quantify_core.visualization.SI_utilities import format_value_string


class SingleQubitTimedomainAnalysis(ba.BaseAnalysis):
    """
    Base Analysis class for single-qubit timedomain experiments.
    """

    # pylint: disable=attribute-defined-outside-init, arguments-differ, line-too-long
    def run(self, calibration_points: Union[bool, Literal["auto"]] = "auto"):
        r"""
        Parameters
        ----------
        calibration_points
            Indicates if the data analyzed includes calibration points. If set to
            :code:`True`, will interpret the last two data points in the dataset as
            :math:`|0\rangle` and :math:`|1\rangle` respectively. If ``"auto"``, will
            use :func:`~.has_calibration_points` to determine if the data contains
            calibration points.

        Returns
        -------
        :class:`~.SingleQubitTimedomainAnalysis`:
            The instance of this analysis.
        """  # NB the return type need to be specified manually to avoid circular import
        if not (calibration_points == "auto" or isinstance(calibration_points, bool)):
            raise ValueError(
                f"Incorrect input. calibration_points={calibration_points} "
                "must be on of False, True or 'auto'."
            )

        self.calibration_points = calibration_points
        return super().run()

    def process_data(self):
        """
        Processes the data so that the analysis can make assumptions on the format.

        Populates self.dataset_processed.S21 with the complex (I,Q) valued transmission,
        and if calibration points are present for the 0 and 1 state, populates
        self.dataset_processed.pop_exc with the excited state population.
        """
        if not hasattr(self.dataset, "y1"):
            self.dataset_processed["S21"] = self.dataset.y0

        elif self.dataset.y1.units == "deg":
            self.dataset_processed["S21"] = self.dataset.y0 * np.exp(
                1j * np.deg2rad(self.dataset.y1)
            )
        else:
            self.dataset_processed["S21"] = self.dataset.y0 + 1j * self.dataset.y1

        self.dataset_processed.S21.attrs["name"] = "S21"
        self.dataset_processed.S21.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.S21.attrs["long_name"] = "Transmission $S_{21}$"

        if self.calibration_points == "auto":
            self.calibration_points = has_calibration_points(
                self.dataset_processed.S21.values
            )

        if self.calibration_points:
            self._rotate_to_calibrated_axis()

    def _rotate_to_calibrated_axis(self, ref_idx_0: int = -2, ref_idx_1: int = -1):
        ref_val_0 = self.dataset_processed.S21[ref_idx_0]
        ref_val_1 = self.dataset_processed.S21[ref_idx_1]
        pop_exc = np.real(
            rotate_to_calibrated_axis(
                data=self.dataset_processed.S21,
                ref_val_0=ref_val_0,
                ref_val_1=ref_val_1,
            )
        )
        self.dataset_processed["pop_exc"] = pop_exc
        self.dataset_processed.pop_exc.attrs["name"] = "pop_exc"
        self.dataset_processed.pop_exc.attrs["units"] = ""
        self.dataset_processed.pop_exc.attrs["long_name"] = r"$|1\rangle$ population"

        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def _choose_data_for_fit(self):
        if self.calibration_points:
            # the last two data points are omitted from the fit as these are cal points
            y_data = self.dataset_processed.pop_exc.values[:-2]
            x_data = self.dataset_processed.x0.values[:-2]
        else:
            # if no calibration points are present, fit the magnitude of the signal
            y_data = np.abs(self.dataset_processed.S21.values)
            x_data = self.dataset_processed.x0.values

        return x_data, y_data


# pylint: disable=too-few-public-methods
class _DecayFigMixin:
    """A mixin for common analysis logic."""

    def _create_decay_figure(self, fig_id: str):
        """
        Creates a figure ready for plotting a fit.
        """

        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        plot_textbox(ax, self.quantities_of_interest["fit_msg"])

        if self.calibration_points:
            ax.plot(
                self.dataset_processed.x0,
                self.dataset_processed.pop_exc,
                marker=".",
                ls="",
            )
            set_ylabel(
                self.dataset_processed.pop_exc.long_name,
                self.dataset_processed.pop_exc.units,
                ax,
            )
        else:
            ax.plot(
                self.dataset_processed.x0,
                abs(self.dataset_processed.S21),
                marker=".",
                ls="",
            )
            set_ylabel(
                r"Magnitude |$S_{21}|$",
                self.dataset_processed.S21.units,
                ax,
            )

        set_xlabel(
            self.dataset_processed.x0.long_name, self.dataset_processed.x0.units, ax
        )

        set_suptitle_from_dataset(fig, self.dataset)

        return ax


class T1Analysis(SingleQubitTimedomainAnalysis, _DecayFigMixin):
    """
    Analysis class for a qubit T1 experiment,
    which fits an exponential decay and extracts the T1 time.
    """

    def run_fitting(self):
        """
        Fit the data to :class:`~quantify_core.analysis.fitting_models.ExpDecayModel`.
        """

        model = fm.ExpDecayModel()
        delay, data = self._choose_data_for_fit()
        guess_pars = model.guess(data, delay=delay)

        if self.calibration_points:
            # if the data is on corrected axes certain parameters can be fixed
            model.set_param_hint("offset", value=0, vary=False)
            model.set_param_hint("amplitude", value=1, vary=False)
            # this call provides updated guess_pars, model.guess is still needed.
            guess_pars = model.make_params()

        fit_result = model.fit(data, params=guess_pars, t=delay)

        self.fit_results.update({"exp_decay_func": fit_result})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """

        fit_result = self.fit_results["exp_decay_func"]
        fit_warning = ba.wrap_text(ba.check_lmfit(fit_result))

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True
            text_msg = "Summary\n"
            text_msg += format_value_string(r"$T1$", fit_result.params["tau"], unit="s")

            if not self.calibration_points:
                unit = self.dataset_processed.S21.units
                text_msg += format_value_string(
                    "\namplitude",
                    fit_result.params["amplitude"],
                    end_char="\n",
                    unit=unit,
                )
                text_msg += format_value_string(
                    "offset", fit_result.params["offset"], unit=unit
                )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["T1"] = ba.lmfit_par_to_ufloat(
            fit_result.params["tau"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """
        Create a figure showing the exponential decay and fit.
        """

        ax = self._create_decay_figure(fig_id="T1_decay")
        plot_fit(
            ax=ax,
            fit_res=self.fit_results["exp_decay_func"],
            plot_init=False,
        )


class EchoAnalysis(SingleQubitTimedomainAnalysis, _DecayFigMixin):
    """
    Analysis class for a qubit spin-echo experiment,
    which fits an exponential decay and extracts the T2_echo time.
    """

    def run_fitting(self):
        """
        Fit the data to :class:`~quantify_core.analysis.fitting_models.ExpDecayModel`.
        """

        model = fm.ExpDecayModel()
        delay, data = self._choose_data_for_fit()
        guess_pars = model.guess(data, delay=delay)

        if self.calibration_points:
            # if the data is on corrected axes certain parameters can be fixed
            model.set_param_hint("offset", value=0.5, vary=False)
            model.set_param_hint("amplitude", value=-0.5, vary=False)
            # this call provides updated guess_pars, model.guess is still needed.
            guess_pars = model.make_params()

        fit_result = model.fit(data, params=guess_pars, t=delay)

        self.fit_results.update({"exp_decay_func": fit_result})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """
        fit_result = self.fit_results["exp_decay_func"]
        fit_warning = ba.wrap_text(ba.check_lmfit(fit_result))

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True
            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$T_{2,\mathrm{Echo}}$", fit_result.params["tau"], unit="s"
            )

            if not self.calibration_points:
                unit = self.dataset_processed.S21.units
                text_msg += format_value_string(
                    "\namplitude",
                    fit_result.params["amplitude"],
                    end_char="\n",
                    unit=unit,
                )
                text_msg += format_value_string(
                    "offset", fit_result.params["offset"], unit=unit
                )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["t2_echo"] = ba.lmfit_par_to_ufloat(
            fit_result.params["tau"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """
        Create a figure showing the exponential decay and fit.
        """

        ax = self._create_decay_figure(fig_id="Echo_decay")
        plot_fit(
            ax=ax,
            fit_res=self.fit_results["exp_decay_func"],
            plot_init=False,
        )


class RamseyAnalysis(SingleQubitTimedomainAnalysis, _DecayFigMixin):
    """
    Fits a decaying cosine curve to Ramsey data (possibly with artificial detuning)
    and finds the true detuning, qubit frequency and T2* time.
    """

    # Override the run method so that we can add the new optional arguments
    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def run(
        self,
        artificial_detuning: float = 0,
        qubit_frequency: float = None,
        calibration_points: Union[bool, Literal["auto"]] = "auto",
    ):
        r"""
        Parameters
        ----------
        artificial_detuning
            The detuning in Hz that will be emulated by adding an extra phase in
            software.
        qubit_frequency
            The initial recorded value of the qubit frequency (before
            accurate fitting is done) in Hz.
        calibration_points
            Indicates if the data analyzed includes calibration points. If set to
            :code:`True`, will interpret the last two data points in the dataset as
            :math:`|0\rangle` and :math:`|1\rangle` respectively. If ``"auto"``, will
            use :func:`~.has_calibration_points` to determine if the data contains
            calibration points.

        Returns
        -------
        :class:`~.RamseyAnalysis`:
            The instance of this analysis.
        """  # NB the return type need to be specified manually to avoid circular import
        self.artificial_detuning = artificial_detuning
        self.qubit_frequency = qubit_frequency
        return super().run(calibration_points=calibration_points)

    def run_fitting(self):
        """
        Fits a :class:`~quantify_core.analysis.fitting_models.DecayOscillationModel`
        to the data.
        """
        model = fm.DecayOscillationModel()
        time, data = self._choose_data_for_fit()
        guess_pars = model.guess(data, t=time)

        if self.calibration_points:
            # if the data is on corrected axes certain parameters can be fixed
            model.set_param_hint("offset", value=0.5, vary=False)
            model.set_param_hint("amplitude", value=0.5, vary=False)
            model.set_param_hint("phase", value=0, vary=False)
            # this call provides updated guess_pars, model.guess is still needed.
            guess_pars = model.make_params()

        fit_result = model.fit(data, params=guess_pars, t=time)

        self.fit_results.update({"Ramsey_decay": fit_result})

    def analyze_fit_results(self):
        """
        Extract the real detuning and qubit frequency based on the artificial detuning
        and fitted detuning.
        """
        fit_warning = ba.check_lmfit(self.fit_results["Ramsey_decay"])

        fit_parameters = self.fit_results["Ramsey_decay"].params

        self.quantities_of_interest["T2*"] = ba.lmfit_par_to_ufloat(
            fit_parameters["tau"]
        )
        self.quantities_of_interest["fitted_detuning"] = ba.lmfit_par_to_ufloat(
            fit_parameters["frequency"]
        )
        self.quantities_of_interest["detuning"] = (
            self.quantities_of_interest["fitted_detuning"] - self.artificial_detuning
        )

        if self.qubit_frequency is not None:
            self.quantities_of_interest["qubit_frequency"] = (
                self.qubit_frequency - self.quantities_of_interest["detuning"]
            )

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True

            text_msg = "Summary\n"
            text_msg += format_value_string(
                r"$T_2^*$",
                self.quantities_of_interest["T2*"],
                unit="s",
                end_char="\n\n",
            )
            text_msg += format_value_string(
                "artificial detuning",
                self.artificial_detuning,
                unit="Hz",
                end_char="\n",
            )
            text_msg += format_value_string(
                "fitted detuning",
                self.quantities_of_interest["fitted_detuning"],
                unit="Hz",
                end_char="\n",
            )
            text_msg += format_value_string(
                "actual detuning",
                self.quantities_of_interest["detuning"],
                unit="Hz",
                end_char="\n",
            )

            if self.qubit_frequency is not None:
                text_msg += "\n"
                text_msg += format_value_string(
                    "initial qubit frequency",
                    self.qubit_frequency,
                    unit="Hz",
                    end_char="\n",
                )
                text_msg += format_value_string(
                    "fitted qubit frequency",
                    self.quantities_of_interest["qubit_frequency"],
                    unit="Hz",
                )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """Plot Ramsey decay figure."""

        ax = self._create_decay_figure(fig_id="Ramsey_decay")
        plot_fit(
            ax=ax,
            fit_res=self.fit_results["Ramsey_decay"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )


class AllXYAnalysis(SingleQubitTimedomainAnalysis):
    """
    Normalizes the data from an AllXY experiment and plots against an ideal curve.

    See section 2.3.2 of :cite:t:`reed_entanglement_2013` for an explanation of
    the AllXY experiment and it's applications in diagnosing errors in single-qubit
    control pulses.
    """

    # pylint: disable=arguments-differ
    def run(self):
        """
        Executes the analysis using specific datapoints as calibration points.

        Returns
        -------
        :class:`~.AllXYAnalysis`:
            The instance of this analysis.
        """  # NB the return type need to be specified manually to avoid circular import
        # The standard analysis of the AllXY analysis always uses datapoints measured
        # within this experiment as calibration points.
        return super().run(calibration_points=True)

    # pylint: disable=arguments-differ
    def _rotate_to_calibrated_axis(self):

        if len(self.dataset.x0) == 21:
            ref_idx_1 = 17
        elif len(self.dataset.x0) == 21 * 2:
            ref_idx_1 = 17 + 21
        # use the values measured for II and XI as reference values.
        super()._rotate_to_calibrated_axis(ref_idx_0=0, ref_idx_1=ref_idx_1)

    def process_data(self):

        # Raise an exception if we do not have the correct number of points for a
        # complete ALLXY experiment
        number_points = len(self.dataset.x0)
        if number_points % 21 != 0:
            raise ValueError(
                "Invalid dataset. The number of calibration points in an "
                "AllXY experiment must be a multiple of 21."
            )

        super().process_data()

        # add the ideal data as a reference curve
        repeats = int(round(number_points / 21))
        ### Creating the ideal data ###
        ideal_data = xr.DataArray(
            data=np.concatenate(
                (
                    0 * np.ones(5 * repeats),
                    0.5 * np.ones(12 * repeats),
                    np.ones(4 * repeats),
                )
            ),
            name="ideal_data",
            coords={"x0": self.dataset_processed.coords["x0"]},
        )
        self.dataset_processed["ideal_data"] = ideal_data

        ### Analyzing Data ###
        deviation = np.mean(np.abs(self.dataset_processed.pop_exc - ideal_data)).item()
        self.quantities_of_interest["deviation"] = deviation

    def create_figures(self):

        fig, ax = plt.subplots()
        fig_id = "AllXY"
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        labels = [
            "II",
            "XX",
            "YY",
            "XY",
            "YX",
            "xI",
            "yI",
            "xy",
            "yx",
            "xY",
            "yX",
            "Xy",
            "Yx",
            "xX",
            "Xx",
            "yY",
            "Yy",
            "XI",
            "YI",
            "xx",
            "yy",
        ]

        ax.plot(
            self.dataset_processed.x0,
            self.dataset_processed.pop_exc,
            marker="o",
            ls="-",
            label="Measured",
        )

        deviation = self.quantities_of_interest["deviation"]
        ax.plot(
            self.dataset_processed.x0,
            self.dataset_processed.ideal_data,
            label=f"Target, \nMean deviation {deviation:#.3g}",
        )

        ax.xaxis.set_ticks(np.arange(21))
        ax.set_xticklabels(labels, rotation=60)
        set_ylabel(
            self.dataset_processed.pop_exc.long_name,
            self.dataset_processed.pop_exc.units,
            ax,
        )
        ax.legend(loc=4)

        set_suptitle_from_dataset(fig, self.dataset)


class RabiAnalysis(SingleQubitTimedomainAnalysis):
    """
    Fits a cosine curve to Rabi oscillation data and finds the qubit drive
    amplitude required to implement a pi-pulse.

    The analysis will automatically rotate the data so that the data lies along the
    axis with the best SNR.
    """

    def run(self, calibration_points: bool = True):
        """
        Parameters
        ----------
        calibration_points
            Specifies if the data should be rotated so that it lies along the axis with
            the best SNR.

        Returns
        -------
        :class:`~.RabiAnalysis`:
            The instance of this analysis.
        """  # NB the return type need to be specified manually to avoid circular import
        # Override the `calibration_points="auto"`
        if not isinstance(calibration_points, bool):
            raise TypeError(
                "Incorrect input. "
                f"calibration_points={calibration_points} must be a bool."
            )
        return super().run(calibration_points=calibration_points)

    # pylint: disable=arguments-differ
    def _rotate_to_calibrated_axis(self):
        """
        If calibration points are True, automatically determine the point farthest
        from the 0 point to use as a reference to rotate the data.

        This will ensure the data lies along the axis with the best SNR.
        """

        # index closest to rabi-pulse amplitude = 0
        min_idx = abs(self.dataset_processed.x0.values).argmin()
        # transmission measured for Rabi-pulse amplitude closest to 0
        min_val = self.dataset_processed.S21[min_idx]

        # find index with max absolute difference (distance in IQ space) to the S21_0
        max_idx = abs(self.dataset_processed.S21 - min_val).argmax()
        max_val = self.dataset_processed.S21[max_idx]

        rotation_angle = np.angle(max_val - min_val)
        rot_data = self.dataset_processed.S21 * np.exp(-1j * rotation_angle)

        self.dataset_processed["S21_rot"] = rot_data
        self.dataset_processed.S21_rot.attrs["name"] = "S21_rot"
        self.dataset_processed.S21_rot.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.S21_rot.attrs[
            "long_name"
        ] = "Rotated transmission $S_{21}^R$"

    def _choose_data_for_fit(self):
        if self.calibration_points:
            y_data = self.dataset_processed.S21_rot.real.values
            x_data = self.dataset_processed.x0.values
        else:
            # if the data is not rotated,fit the magnitude of the signal
            y_data = np.abs(self.dataset_processed.S21.values)
            x_data = self.dataset_processed.x0.values

        return x_data, y_data

    def run_fitting(self):
        """
        Fits a :class:`~quantify_core.analysis.fitting_models.RabiModel` to the data.
        """
        model = fm.RabiModel()
        drive_amplitude, data = self._choose_data_for_fit()

        guess = model.guess(data, drive_amp=drive_amplitude)
        fit_result = model.fit(data, params=guess, x=drive_amplitude)

        self.fit_results.update({"Rabi_oscillation": fit_result})

    def analyze_fit_results(self):
        """
        Checks fit success and populates :code:`.quantities_of_interest`.
        """
        fit_result = self.fit_results["Rabi_oscillation"]
        fit_warning = ba.wrap_text(ba.check_lmfit(fit_result))

        # If there is a problem with the fit, display an error message in the text box.
        # Otherwise, display the parameters as normal.
        if fit_warning is None:
            self.quantities_of_interest["fit_success"] = True

            text_msg = "Summary\n"
            text_msg += format_value_string(
                "Pi-pulse amplitude",
                fit_result.params["amp180"],
                unit="V",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Oscillation amplitude",
                fit_result.params["amplitude"],
                unit="V",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Offset", fit_result.params["offset"], unit="V", end_char="\n"
            )
        else:
            text_msg = ba.wrap_text(fit_warning)
            self.quantities_of_interest["fit_success"] = False

        self.quantities_of_interest["Pi-pulse amplitude"] = ba.lmfit_par_to_ufloat(
            fit_result.params["amp180"]
        )
        self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self):
        """Creates Rabi oscillation figure"""

        fig_id = "Rabi_oscillation"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        plot_textbox(ax, self.quantities_of_interest["fit_msg"])

        if self.calibration_points:
            ax.plot(
                self.dataset_processed.x0,
                self.dataset_processed.S21_rot.real,
                marker=".",
                linestyle="",
            )

            set_ylabel(
                self.dataset_processed.S21_rot.long_name,
                self.dataset_processed.S21_rot.units,
                ax,
            )
        else:
            ax.plot(
                self.dataset_processed.x0,
                self.dataset_processed.S21.real,
                marker=".",
                linestyle="",
            )
            set_ylabel(
                self.dataset_processed.S21.long_name,
                self.dataset_processed.S21.units,
                ax,
            )

        plot_fit(
            ax=ax,
            fit_res=self.fit_results["Rabi_oscillation"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        set_xlabel(
            self.dataset_processed.x0.long_name, self.dataset_processed.x0.units, ax
        )

        set_suptitle_from_dataset(fig, self.dataset)
