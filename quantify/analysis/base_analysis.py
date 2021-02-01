"""
This module should contain different analyses corresponding to discrete experiments
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from abc import ABC
from collections import OrderedDict
from typing import Callable
from quantify.visualization import mpl_plotting as qpl
from quantify.data.handling import (
    load_dataset,
    get_latest_tuid,
    _locate_experiment_file,
    get_datadir,
)
from quantify.visualization.SI_utilities import set_xlabel, set_ylabel

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
        flow: tuple = "auto",
        flow_skip_steps: tuple = tuple(),
        flow_override_step: tuple = tuple(),
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

        if flow == "auto":
            flow = (
                "extract_data",  # extract data specified in params dict
                "process_data",  # binning, filtering etc
                "prepare_fitting",  # set up fit_dicts
                "run_fitting",  # fitting to models
                "save_fit_results",
                "analyze_fit_results",  # analyzing the results of the fits
                "save_quantities_of_interest",
                "create_figures",
                "adjust_figures",
                "save_figures",
            )
        self.flow = flow
        self.flow_skip_steps = flow_skip_steps
        self.flow_override_step = flow_override_step

        # This will be overwritten
        self.dset = None
        # To be populated by a subclass
        self.figs_mpl = OrderedDict()
        self.axs_mpl = OrderedDict()
        self.fit_res = OrderedDict()

        self.run_analysis()

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

    def run_analysis(self):
        """
        This function is at the core of all analysis and defines the flow.

        This function is typically called at the end of __init__.
        """

        for step in self.flow:
            if type(step) is tuple and len(step) == 2:
                method_name = step[0]
                method_kwargs = step[1]
            elif type(step) is str:
                method_name = step
                method_kwargs = OrderedDict({})
            else:
                raise ValueError(
                    f"Analysis flow step `{step}` is not a len-2 `tuple` nor a `str`."
                )

            if self.flow_override_step and method_name == self.flow_override_step[0]:
                method_kwargs = self.flow_override_step[1]

            if method_name not in self.flow_skip_steps:
                # method_name must be a method of the class/subclass instance
                method = getattr(self, method_name)

                # execute step and pass any (optional) arguments to it
                try:
                    method(**method_kwargs)
                except Exception as e:
                    raise RuntimeError(
                        "An exception occurred while executing "
                        f"`{method_name}(**{method_kwargs})`."  # point to the culprit
                    ) from e  # and raise the original exception

    def process_data(self):
        """
        This method can be used to process, e.g., reshape, filter etc. the data
        before starting the analysis. By default this method is empty (pass).
        """
        pass

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

    def adjust_figures(self, user_func: Callable = lambda analysis_obj: None):
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

        # Execute a user-defined function that adjusts aspect of the figures
        # Main use-case: change plotting ranges
        user_func(analysis_obj=self)

    def save_figures(self, close_figs: bool = True):
        """
        Saves all the figures in the :code:`figs_mpl` dict

        Parameters
        ----------

        close_figs
            If True, closes `matplotlib` figures after saving
        """
        DPI = this.settings["DPI"]
        formats = this.settings["fig_formats"]

        for figname, fig in self.figs_mpl.items():
            filename = _locate_experiment_file(self.tuid, get_datadir(), f"{figname}")
            for form in formats:
                fig.savefig(f"{filename}.{form}", bbox_inches="tight", dpi=DPI)
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
            f, ax = plt.subplots()
            fig_id = f"Line plot x0-{yi}"
            self.figs_mpl[fig_id] = f
            self.axs_mpl[fig_id] = ax

            plot_basic1D(
                ax=ax,
                x=self.dset["x0"].values,
                xlabel=self.dset["x0"].attrs["long_name"],
                xunit=self.dset["x0"].attrs["unit"],
                y=self.dset[f"{yi}"].values,
                ylabel=self.dset[f"{yi}"].attrs["long_name"],
                yunit=self.dset[f"{yi}"].attrs["unit"],
            )

            f.suptitle(
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
            f, ax = plt.subplots()
            fig_id = f"Heatmap x0x1-{yi}"

            self.figs_mpl[fig_id] = f
            self.axs_mpl[fig_id] = ax

            qpl.plot_2D_grid(
                x=self.dset["x0"],
                y=self.dset["x1"],
                z=self.dset[f"{yi}"],
                xlabel=self.dset["x0"].attrs["long_name"],
                xunit=self.dset["x0"].attrs["unit"],
                ylabel=self.dset["x1"].attrs["long_name"],
                yunit=self.dset["x1"].attrs["unit"],
                zlabel=self.dset[f"{yi}"].attrs["long_name"],
                zunit=self.dset[f"{yi}"].attrs["unit"],
                ax=ax,
            )

            f.suptitle(
                f"x0x1-{yi} {self.dset.attrs['name']}\ntuid: {self.dset.attrs['tuid']}"
            )


def plot_basic1D(
    x,
    y,
    xlabel: str,
    xunit: str,
    ylabel: str,
    yunit: str,
    ax,
    title: str = None,
    plot_kw: dict = {},
    **kw,
):
    ax.plot(x, y, **plot_kw)
    if title is not None:
        ax.set_title(title)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)


def plot_fit(ax, fit_res, plot_init: bool = True, plot_numpoints: int = 1000, **kw):
    model = fit_res.model

    if len(model.independent_vars) == 1:
        independent_var = model.independent_vars[0]
    else:
        raise ValueError(
            "Fit can only be plotted if the model function"
            " has one independent variable."
        )

    x_arr = fit_res.userkws[independent_var]
    x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
    y = model.eval(fit_res.params, **{independent_var: x})
    ax.plot(x, y, label="Fit", c="C3")

    if plot_init:
        x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
        y = model.eval(fit_res.init_params, **{independent_var: x})
        ax.plot(x, y, ls="--", c="grey", label="Guess")


def mk_adjust_xylim(
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
):
    def adjust_xylim(analysis_obj: BaseAnalysis):
        axs = analysis_obj.axs_mpl
        for ax_id, ax in axs.items():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    return adjust_xylim


def mk_adjust_clim(min_val: float, max_val: float, contains: str = ""):
    def adjust_clim(analysis_obj: BaseAnalysis):
        axs = analysis_obj.axs_mpl
        for ax_id, ax in axs.items():
            # For plots created with `imshow` or `pcolormesh`
            for im_or_col in (
                *ax.get_images(),
                *(c for c in ax.collections if isinstance(c, QuadMesh)),
            ):
                c_ax = im_or_col.colorbar.ax
                # print(im_or_col, c_ax.get_xlabel(), c_ax.get_ylabel())
                if contains in c_ax.get_xlabel() or contains in c_ax.get_ylabel():
                    im_or_col.set_clim(min_val, max_val)

    return adjust_clim
