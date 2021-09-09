# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
import numpy as np
import xarray as xr
from typing import Union

import matplotlib.pyplot as plt
from quantify_core.analysis import base_analysis as ba
from quantify_core.analysis import fitting_models as fm
from quantify_core.visualization.mpl_plotting import (
    set_xlabel,
    set_ylabel,
    set_suptitle_from_dataset,
    plot_textbox,
    plot_fit,
)
from quantify_core.data.types import TUID
from quantify_core.visualization.SI_utilities import format_value_string


def rotate_to_calibrated_axis(data: np.ndarray, ref_val_0: complex, ref_val_1: complex):
    """
    Rotates, normalizes and offsets complex valued data based on calibration points.

    Parameters
    ----------
    data
        An array of complex valued data points.
    ref_val_0
        The reference value corresponding to the 0 state.
    ref_val_1
        The reference value corresponding to the 1 state.
    """
    rotation_anle = np.angle(ref_val_1 - ref_val_0)
    norm = np.abs(ref_val_1 - ref_val_0)
    offset = ref_val_0 * np.exp(-1j * rotation_anle) / norm

    corrected_data = np.real(data * np.exp(-1j * rotation_anle) / norm - offset)

    return corrected_data


class SingleQubitTimedomainAnalysis(ba.BaseAnalysis):
    """
    Base Analysis class for single-qubit timedomain experiments.
    """

    def __init__(
        self,
        dataset: xr.Dataset = None,
        tuid: Union[TUID, str] = None,
        label: str = "",
        settings_overwrite: dict = None,
    ):
        self.calibration_points: bool = True
        """indicates if the data analyzed includes calibration points."""

        super().__init__(
            dataset=dataset,
            tuid=tuid,
            label=label,
            settings_overwrite=settings_overwrite,
        )

    def run(self, calibration_points: bool = True):
        """
        Parameters
        ----------
        calibration_points:

        Returns
        -------
        :class:`~quantify_core.analysis.single_qubit_timedomain.SingleQubitTimedomainAnalysis`:
            The instance of this analysis.

        """  # NB the return type need to be specified manually to avoid circular import
        self.calibration_points = calibration_points
        return super().run()

    def process_data(self):
        """
        Processes the data so that the analysis can make assumptions on the format.

        Populates self.dataset_processed.S21 with the complex (I,Q) valued transmission,
        and if calibration points are present for the 0 and 1 state, populates
        self.dataset_processed.pop_exc with the excited state population.
        """
        if self.dataset.y1.units == "deg":
            self.dataset_processed["S21"] = self.dataset.y0 * np.exp(
                1j * np.deg2rad(self.dataset.y1)
            )
        else:
            self.dataset_processed["S21"] = self.dataset.y0 + 1j * self.dataset.y1

        self.dataset_processed.S21.attrs["name"] = "S21"
        self.dataset_processed.S21.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.S21.attrs["long_name"] = "Transmission $S_{21}$"

        if self.calibration_points:
            ref_val_0 = self.dataset_processed.S21[-2]
            ref_val_1 = self.dataset_processed.S21[-1]
            pop_exc = rotate_to_calibrated_axis(
                data=self.dataset_processed.S21,
                ref_val_0=ref_val_0,
                ref_val_1=ref_val_1,
            )
            self.dataset_processed["pop_exc"] = pop_exc
            self.dataset_processed.pop_exc.attrs["name"] = "pop_exc"
            self.dataset_processed.pop_exc.attrs["units"] = ""
            self.dataset_processed.pop_exc.attrs[
                "long_name"
            ] = r"$|1\rangle$ population"
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})


class T1Analysis(SingleQubitTimedomainAnalysis):
    """
    Analysis class for a qubit T1 experiment,
    which fits an exponential decay and extracts the T1 time.
    """

    def run_fitting(self):
        """
        Fit the data to :class:`~quantify_core.analysis.fitting_models.ExpDecayModel`.
        """

        model = fm.ExpDecayModel()

        if self.calibration_points:
            # the last two data points are omitted from the fit as these are cal points
            data = self.dataset_processed.pop_exc.values[:-2]
            delay = self.dataset_processed.x0.values[:-2]
        else:
            # if no calibration points are present, fit the magnitude of the signal
            data = np.abs(self.dataset_processed.S21.values)
            delay = self.dataset_processed.x0.values

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

        fig_id = "T1_decay"
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
                ax,
                self.dataset_processed.pop_exc.long_name,
                self.dataset_processed.pop_exc.units,
            )
        else:
            ax.plot(
                self.dataset_processed.x0,
                abs(self.dataset_processed.S21),
                marker=".",
                ls="",
            )
            set_ylabel(
                ax,
                r"Magnitude |$S_{21}|$",
                self.dataset_processed.S21.units,
            )
        plot_fit(
            ax=ax,
            fit_res=self.fit_results["exp_decay_func"],
            plot_init=False,
        )

        set_xlabel(
            ax, self.dataset_processed.x0.long_name, self.dataset_processed.x0.units
        )

        set_suptitle_from_dataset(fig, self.dataset, "S21")


class EchoAnalysis(SingleQubitTimedomainAnalysis):
    """
    Analysis class for a qubit spin-echo experiment,
    which fits an exponential decay and extracts the T2_echo time.
    """

    def run_fitting(self):
        """
        Fit the data to :class:`~quantify_core.analysis.fitting_models.ExpDecayModel`.
        """

        model = fm.ExpDecayModel()

        if self.calibration_points:
            # the last two data points are omitted from the fit as these are cal points
            data = self.dataset_processed.pop_exc.values[:-2]
            delay = self.dataset_processed.x0.values[:-2]
        else:
            # if no calibration points are present, fit the magnitude of the signal
            data = np.abs(self.dataset_processed.S21.values)
            delay = self.dataset_processed.x0.values

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

        fig_id = "Echo_decay"
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
                ax,
                self.dataset_processed.pop_exc.long_name,
                self.dataset_processed.pop_exc.units,
            )
        else:
            ax.plot(
                self.dataset_processed.x0,
                abs(self.dataset_processed.S21),
                marker=".",
                ls="",
            )
            set_ylabel(
                ax,
                r"Magnitude |$S_{21}|$",
                self.dataset_processed.S21.units,
            )
        plot_fit(
            ax=ax,
            fit_res=self.fit_results["exp_decay_func"],
            plot_init=False,
        )

        set_xlabel(
            ax, self.dataset_processed.x0.long_name, self.dataset_processed.x0.units
        )

        set_suptitle_from_dataset(fig, self.dataset, "S21")


class RamseyAnalysis(ba.BaseAnalysis):
    """
    Fits a decaying cosine curve to Ramsey data (possibly with artificial detuning)
    and finds the true detuning, qubit frequency and T2* time.
    """

    # Override the run method so that we can add the new optional arguments
    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def run(self, artificial_detuning: float = 0, qubit_frequency: float = None):
        """
        Parameters
        ----------
        artificial_detuning:
            The detuning in Hz that will be emulated by adding an extra phase in
            software.
        qubit_frequency:
            The initial recorded value of the qubit frequency in software (before
            accurate fitting is done) in Hz.

        Returns
        -------
        :class:`~quantify_core.analysis.ramsey_analysis.RamseyAnalysis`:
            The instance of this analysis.

        """  # NB the return type need to be specified manually to avoid circular import
        self.artificial_detuning = artificial_detuning
        self.qubit_frequency = qubit_frequency
        return super().run()

    def process_data(self):
        """
        Populates the :code:`.dataset_processed`.
        """

        magnitude = self.dataset.y0
        # TODO solve NaNs properly when #176 has a solution, pylint: disable=fixme
        valid_measurement = np.logical_not(np.isnan(magnitude))
        self.dataset_processed["Magnitude"] = magnitude[valid_measurement]
        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = self.dataset.y0.units
        self.dataset_processed.Magnitude.attrs["long_name"] = r"Magnitude, $|S_{21}|$"

        self.dataset_processed["x0"] = self.dataset.x0[valid_measurement]
        self.dataset_processed = self.dataset_processed.set_coords("x0")
        # replace the default dim_0 with x0
        self.dataset_processed = self.dataset_processed.swap_dims({"dim_0": "x0"})

    def run_fitting(self):
        """
        Fits a :class:`~quantify_core.analysis.fitting_models.DecayOscillationModel` to the
        data.
        """
        model = fm.DecayOscillationModel()

        magnitude = self.dataset_processed["Magnitude"].values
        time = self.dataset_processed.x0.values
        guess = model.guess(magnitude, time=time)
        fit_result = model.fit(magnitude, params=guess, t=time)

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
        """Plot Ramsey decay figure"""

        fig_id = "Ramsey_decay"
        fig, ax = plt.subplots()
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax

        # Add a textbox with the fit_message
        plot_textbox(ax, self.quantities_of_interest["fit_msg"])

        self.dataset_processed.Magnitude.plot(ax=ax, marker=".", linestyle="")

        plot_fit(
            ax=ax,
            fit_res=self.fit_results["Ramsey_decay"],
            plot_init=not self.quantities_of_interest["fit_success"],
            range_casting="real",
        )

        set_ylabel(ax, r"Output voltage", self.dataset_processed.Magnitude.units)
        set_xlabel(
            ax,
            self.dataset_processed.x0.long_name,
            self.dataset_processed.x0.units,
        )

        set_suptitle_from_dataset(fig, self.dataset, "S21")
