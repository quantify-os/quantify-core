# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing analysis class for time of flight measurement."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from quantify_core.analysis import base_analysis as ba
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import format_value_string


class TimeOfFlightAnalysis(ba.BaseAnalysis):
    """Analysis for time of flight measurement."""

    # pylint: disable=no-member
    # pylint: disable=attribute-defined-outside-init

    def run(
        self, acquisition_delay: float = 4e-9, playback_delay: float = 146e-9
    ) -> ba.BaseAnalysis:
        """
        Execute analysis steps and let user specify `acquisition_delay`.

        Assumes that the sample time is always 1 ns.

        Parameters
        ----------
        acquisition_delay
            Time from the start of the pulse to the start of the measurement in seconds.
            By default 4 ns.
        playback_delay
            Time from the start of playback to appearance of pulse at the output
            of the instrument in seconds. By default 146 ns, which is the playback
            delay for Qblox instruments.

        Returns
        -------
        :
            The instance of the analysis object.
        """
        self.acquisition_delay_ns = round(acquisition_delay * 1e9)
        self.playback_delay_ns = round(playback_delay * 1e9)
        return super().run()

    def process_data(self) -> None:
        """Populate the :code:`.dataset_processed`."""
        self.dataset_processed = xr.Dataset(
            {
                "Magnitude": (("Time",), self.dataset.y0.data),
            },
            coords={
                "Time": np.arange(self.dataset.y0.data.shape[0])
                + self.acquisition_delay_ns
            },
        )
        self.dataset_processed.Magnitude.attrs["name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["long_name"] = "Magnitude"
        self.dataset_processed.Magnitude.attrs["units"] = self.dataset.y0.units

        self.dataset_processed.Time.attrs["name"] = "Time"
        self.dataset_processed.Time.attrs["long_name"] = "Trace acquisition sample"
        self.dataset_processed.Time.attrs["units"] = "ns"

    def analyze_fit_results(self) -> None:
        """Check fit success and populates :code:`.quantities_of_interest`."""
        mean_val = np.mean(
            self.dataset_processed.Magnitude[0 : self.playback_delay_ns // 3]
        )
        std_val = np.std(
            self.dataset_processed.Magnitude[0 : self.playback_delay_ns // 3]
        )
        threshold = mean_val + 4 * float(std_val)
        idx_above_threshold = np.where(
            np.abs(self.dataset_processed.Magnitude) > threshold
        )[0]
        two_subseq_indices_above_threshold = np.where(
            np.diff(idx_above_threshold) == 1
        )[0]

        if (len(idx_above_threshold) == 0) or (
            len(two_subseq_indices_above_threshold) == 0
        ):
            self.quantities_of_interest["fit_success"] = False
            self.quantities_of_interest["fit_msg"] = (
                "Can not find the Time of flight,\n"
                "try to reduce the acquisition_delay "
                f"(current value: {self.acquisition_delay_ns} ns)."
            )
        else:
            self.quantities_of_interest["fit_success"] = True
            tof_idx = two_subseq_indices_above_threshold[0]
            self.quantities_of_interest["tof"] = (
                int(idx_above_threshold[tof_idx]) + self.acquisition_delay_ns
            )
            self.quantities_of_interest["nco_prop_delay"] = (
                self.quantities_of_interest["tof"] - self.playback_delay_ns
            )

            for var in ("tof", "nco_prop_delay"):
                self.quantities_of_interest[var] = round(
                    self.quantities_of_interest[var] * 1e-9, ndigits=9
                )

            text_msg = "Summary:\n"
            text_msg += format_value_string(
                "Time of flight",
                self.quantities_of_interest["tof"],
                unit="s",
                end_char="\n",
            )
            text_msg += format_value_string(
                "Playback delay",
                self.playback_delay_ns,
                unit="ns",
                end_char="\n",
            )
            text_msg += format_value_string(
                "NCO propagation delay",
                self.quantities_of_interest["nco_prop_delay"],
                unit="s",
                end_char="",
            )
            self.quantities_of_interest["fit_msg"] = text_msg

    def create_figures(self) -> None:
        """Display the Data and the measured time of flight."""
        fig_id = "Time_of_flight"
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        self.dataset_processed.Magnitude.plot(
            ax=ax, color="mediumturquoise", zorder=100
        )
        ax.grid(color="black", zorder=1, alpha=1 / 20)

        if self.quantities_of_interest["fit_success"]:
            tof = self.quantities_of_interest["tof"]
            ax.axvline(tof * 1e9, ls="--", color="red", zorder=101)

        # Put a text box summarizing the QOI
        qpl.plot_textbox(
            ax=ax, transform=ax.transAxes, text=self.quantities_of_interest["fit_msg"]
        )

        qpl.set_suptitle_from_dataset(fig, self.dataset)  # type: ignore
        ax.set_title("Trace acquisition")

        fig.tight_layout()

        self.figs_mpl[fig_id] = fig  # type: ignore
        self.axs_mpl[fig_id] = (ax,)  # type: ignore
