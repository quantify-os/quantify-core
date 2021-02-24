"""
This module should contain different analyses corresponding to discrete experiments
"""
import sys
from abc import ABC
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh

from quantify.visualization import mpl_plotting as qpl
from quantify.data.handling import (
    load_dataset,
    get_latest_tuid,
    _locate_experiment_file,
    get_datadir,
)

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# global configurations at the level of the analysis module
this.settings = {
    "DPI": 600,  # define resolution of some matplotlib output formats
    "fig_formats": ("png", "svg"),
    "presentation_mode": False,
    "transparent_background": False,
}


class BaseAnalysis(ABC):
    """
    Abstract base class for data analysis. Provides a template from which to
    inherit when doing any analysis.
    """

    def __init__(
        self,
        label: str = "",
        tuid: str = None,
        interrupt_after: str = "save_figures",
    ):
        """
        Initializes the variables that are used in the analysis and to which data is stored.

        Parameters
        ------------------
        label: str
            Will look for a dataset that contains "label" in the name.
        tuid: str
            If specified, will look for the dataset with the matching tuid.
        """

        self.label = label
        self.tuid = tuid
        self.interrupt_after = interrupt_after

        # This will be overwritten
        self.dset = None
        # To be populated by a subclass
        self.figs_mpl = OrderedDict()
        self.axs_mpl = OrderedDict()
        self.fit_res = OrderedDict()

        self.run_analysis()

    def run_analysis(self):
        """
        This function is at the core of all analysis and defines the flow.

        This function is typically called at the end of __init__.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            method_stop=self.interrupt_after,
            method_stop_inclusive=True,
        )

        for method in flow_methods:
            method()

    def continue_analysis_from(self, method_name: str):
        """
        Runs the analysis starting from specified method.

        The methods are called in the same order as in :meth:`~run_analysis`.
        Useful when the analysis interrupted at some stage with `interrupt_after`.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            method_start=method_name,
            method_start_inclusive=True,  # self.method_name will be executed
        )

        for method in flow_methods:
            method()

    def continue_analysis_after(self, method_name: str):
        """
        Runs the analysis starting from specified method.

        The methods are called in the same order as in :meth:`~run_analysis`.
        Useful when the analysis interrupted at some stage with `interrupt_after`.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            method_start=method_name,
            method_start_inclusive=False,  # self.method_name will not be executed
        )

        for method in flow_methods:
            method()

    def get_flow(self):
        """
        Returns a tuple with the ordered methods to be called by run analysis
        """
        return (
            self.extract_data,  # extract data from the dataset
            self.process_data,  # binning, filtering etc
            self.prepare_fitting,  # set up fit_dicts
            self.run_fitting,  # fitting to models
            self.save_fit_results,
            self.analyze_fit_results,  # analyzing the results of the fits
            self.save_quantities_of_interest,
            self.create_figures,
            self.adjust_figures,
            self.save_figures,
        )

    def extract_data(self):
        """
        Populates `self.dset` with data from the experiment matching the tuid/label.

        This method should be overwritten if an analysis does not relate to a single datafile.
        """

        # if no TUID is specified use the label to search for the latest file with a match.
        if self.tuid is None:
            self.tuid = get_latest_tuid(contains=self.label)

        self.dset = load_dataset(tuid=self.tuid)

        # maybe also load in the metadata here?

    def process_data(self):
        """
        This method can be used to process, e.g., reshape, filter etc. the data
        before starting the analysis. By default this method is empty (pass).
        """

    def prepare_fitting(self):
        pass

    def run_fitting(self):
        pass

    def save_fit_results(self):
        pass

    def analyze_fit_results(self):
        pass

    def save_quantities_of_interest(self):
        pass

    def create_figures(self):
        pass

    def adjust_figures(self):
        """
        Perform global adjustments after creating the figures but
        before saving them
        """
        for fig in self.figs_mpl.values():
            if this.settings["presentation_mode"]:
                # Remove the experiment name and tuid from figures
                fig.suptitle(r"")
            if this.settings["transparent_background"]:
                # Set transparent background on figures
                fig.patch.set_alpha(0)

    def save_figures(self, close_figs: bool = True):
        """
        Saves all the figures in the :code:`figs_mpl` dict

        Parameters
        ----------

        close_figs
            If True, closes `matplotlib` figures after saving
        """
        dpi = this.settings["DPI"]
        formats = this.settings["fig_formats"]

        for figname, fig in self.figs_mpl.items():
            filename = _locate_experiment_file(self.tuid, get_datadir(), f"{figname}")
            for form in formats:
                fig.savefig(f"{filename}.{form}", bbox_inches="tight", dpi=dpi)
            if close_figs:
                plt.close(fig)


class Basic1DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):

        ys = set(self.dset.keys())
        ys.discard("x0")
        for yi in ys:
            fig, ax = plt.subplots()
            fig_id = f"Line plot x0-{yi}"

            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax

            qpl.plot_basic_1d(
                ax=ax,
                x=self.dset["x0"].values,
                xlabel=self.dset["x0"].attrs["long_name"],
                xunit=self.dset["x0"].attrs["units"],
                y=self.dset[f"{yi}"].values,
                ylabel=self.dset[f"{yi}"].attrs["long_name"],
                yunit=self.dset[f"{yi}"].attrs["units"],
            )

            fig.suptitle(
                f"x0-{yi} {self.dset.attrs['name']}\ntuid: {self.dset.attrs['tuid']}"
            )


class Basic2DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):
        ys = set(self.dset.keys())
        ys.discard("x0")
        ys.discard("x1")

        for yi in ys:
            fig, ax = plt.subplots()
            fig_id = f"Heatmap x0x1-{yi}"

            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax

            qpl.plot_2d_grid(
                x=self.dset["x0"],
                y=self.dset["x1"],
                z=self.dset[f"{yi}"],
                xlabel=self.dset["x0"].attrs["long_name"],
                xunit=self.dset["x0"].attrs["units"],
                ylabel=self.dset["x1"].attrs["long_name"],
                yunit=self.dset["x1"].attrs["units"],
                zlabel=self.dset[f"{yi}"].attrs["long_name"],
                zunit=self.dset[f"{yi}"].attrs["units"],
                ax=ax,
            )

            fig.suptitle(
                f"x0x1-{yi} {self.dset.attrs['name']}\ntuid: {self.dset.attrs['tuid']}"
            )


def adjust_ylim(
    analysis_obj: BaseAnalysis,
    ymin: float = None,
    ymax: float = None,
    contains: str = "",
) -> None:
    axs = analysis_obj.axs_mpl
    for ax_id, ax in axs.items():
        if contains in ax_id:
            ax.set_ylim(ymin, ymax)


def adjust_xlim(
    analysis_obj: BaseAnalysis,
    xmin: float = None,
    xmax: float = None,
    contains: str = "",
) -> None:
    axs = analysis_obj.axs_mpl
    for ax_id, ax in axs.items():
        if contains in ax_id:
            ax.set_xlim(xmin, xmax)


def adjust_clim(
    analysis_obj: BaseAnalysis, vmin: float, vmax: float, contains: str = ""
) -> None:
    axs = analysis_obj.axs_mpl
    for ax in axs.values():
        # For plots created with `imshow` or `pcolormesh`
        for im_or_col in (
            *ax.get_images(),
            *(c for c in ax.collections if isinstance(c, QuadMesh)),
        ):
            c_ax = im_or_col.colorbar.ax
            # print(im_or_col, c_ax.get_xlabel(), c_ax.get_ylabel())
            if contains in c_ax.get_xlabel() or contains in c_ax.get_ylabel():
                im_or_col.set_clim(vmin, vmax)


def _get_modified_flow(
    flow_functions: tuple,
    method_start: str = "",
    method_start_inclusive: bool = True,
    method_stop: str = "",
    method_stop_inclusive: bool = True,
):
    method_names = [meth.__name__ for meth in flow_functions]

    if method_start:
        start_idx = method_names.index(method_start)
        if not method_start_inclusive:
            start_idx += 1
    else:
        start_idx = 0

    if method_stop:
        stop_idx = method_names.index(method_stop)
        if method_stop_inclusive:
            stop_idx += 1
    else:
        stop_idx = None

    flow_functions = flow_functions[start_idx:stop_idx]

    return flow_functions
