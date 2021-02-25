"""
This module should contain different analyses corresponding to discrete experiments
"""
from __future__ import annotations
import sys
import os
import json
from abc import ABC
from collections import OrderedDict
from typing import Union, List
from pathlib import Path

import xarray as xr
import lmfit
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from qcodes.utils.helpers import NumpyJSONEncoder

from quantify.visualization import mpl_plotting as qpl
from quantify.data.handling import (
    load_dataset,
    get_latest_tuid,
    get_datadir,
    write_dataset,
    locate_experiment_container,
)

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# global configurations at the level of the analysis module
this.settings = {
    "DPI": 450,  # define resolution of some matplotlib output formats
    "fig_formats": ("png", "svg"),
    "presentation_mode": False,
    "transparent_background": False,
}


class BaseAnalysis(ABC):
    """
    Abstract base class for data analysis. Provides a template from which to
    inherit when doing any analysis.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        label: str = "",
        tuid: str = None,
        interrupt_after: Union[
            "extract_data",
            "process_data",
            "prepare_fitting",
            "run_fitting",
            "analyze_fit_results",
            "create_figures",
            "adjust_figures",
            "save_figures_mpl",
            "save_quantities_of_interest",
            "save_processed_dataset",
        ] = "",
    ):
        """
        Initializes the variables used in the analysis and to which data is stored.

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
        # Used to save a reference of the raw dataset
        self.dset_raw = None
        # To be populated by a subclass
        self.figs_mpl = OrderedDict()
        self.axs_mpl = OrderedDict()
        self.quantities_of_interest = OrderedDict()
        self.fit_res = OrderedDict()

        self.run_analysis()

    @property
    def name(self):
        """The name of the analysis, used in data saving."""
        # used to store data and figures resulting from the analysis. Can be overwritten
        return self.__class__.__name__

    @property
    def analysis_dir(self):
        """
        Analysis dir based on the tuid. Will create a directory if it does not exist yet.
        """
        if self.tuid is None:
            raise ValueError("Unknown TUID, cannot determine the analysis directory.")
        # This is a property as it depends
        exp_folder = Path(locate_experiment_container(self.tuid, get_datadir()))
        analysis_dir = exp_folder / f"analysis_{self.name}"
        if not os.path.isdir(analysis_dir):
            os.makedirs(analysis_dir)

        return analysis_dir

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

    def continue_analysis_from(
        self,
        method_name: Union[
            "extract_data",
            "process_data",
            "prepare_fitting",
            "run_fitting",
            "analyze_fit_results",
            "create_figures",
            "adjust_figures",
            "save_figures_mpl",
            "save_quantities_of_interest",
            "save_processed_dataset",
        ],
    ):
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

    def continue_analysis_after(
        self,
        method_name: Union[
            "extract_data",
            "process_data",
            "prepare_fitting",
            "run_fitting",
            "analyze_fit_results",
            "create_figures",
            "adjust_figures",
            "save_figures_mpl",
            "save_quantities_of_interest",
        ],
    ):
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
            self.analyze_fit_results,  # analyzing the results of the fits
            self.save_quantities_of_interest,
            self.create_figures,
            self.adjust_figures,
            self.save_figures_mpl,
            self.save_quantities_of_interest,
            self.save_processed_dataset,
        )

    def extract_data(self):
        """
        Populates `self.dset_raw` with data from the experiment matching the tuid/label.

        This method should be overwritten if an analysis does not relate to a single
        datafile.
        """

        # if no TUID is specified use the label to search for the latest file with a match.
        if self.tuid is None:
            self.tuid = get_latest_tuid(contains=self.label)

        # Keep a reference to the original dataset.
        self.dset_raw = load_dataset(tuid=self.tuid)
        # Initialize an empty dataset for the processed data.
        self.dset = xr.Dataset()

    def process_data(self):
        """
        This method can be used to process, e.g., reshape, filter etc. the data
        before starting the analysis. By default this method is empty (pass).
        """

    def prepare_fitting(self):
        pass

    def run_fitting(self):
        pass

    def _add_fit_res_to_qoi(self):
        if len(self.fit_res) > 0:
            self.quantities_of_interest["fit_res"] = OrderedDict()
            for fr_name, fit_result in self.fit_res.items():
                res = flatten_lmfit_modelresult(fit_result)
                self.quantities_of_interest["fit_res"][fr_name] = res

    def analyze_fit_results(self):
        pass

    def save_quantities_of_interest(self):
        self._add_fit_res_to_qoi()

        with open(
            os.path.join(self.analysis_dir, "quantities_of_interest.json"), "w"
        ) as file:
            json.dump(self.quantities_of_interest, file, cls=NumpyJSONEncoder, indent=4)

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

    def save_processed_dataset(self, exclude_raw: bool = None):
        """
        Saves a copy of the (processed) self.dset in the analysis folder of the experiment.
        """

        # if statement exist to be compatible with child classes that do not load data
        # onto the self.dset object.
        if self.dset is not None:
            dataset = self.dset
            write_dataset(Path(self.analysis_dir) / "processed_dataset.hdf5", dataset)

    def save_figures_mpl(self, close_figs: bool = True):
        """
        Saves all the matplotlib figures in the :code:`figs_mpl` dict

        Parameters
        ----------

        close_figs
            If True, closes `matplotlib` figures after saving
        """
        dpi = this.settings["DPI"]
        formats = this.settings["fig_formats"]

        if len(self.figs_mpl) != 0:
            mpl_figdir = Path(self.analysis_dir) / "figs_mpl"
            if not os.path.isdir(mpl_figdir):
                os.makedirs(mpl_figdir)

            for figname, fig in self.figs_mpl.items():
                filename = os.path.join(mpl_figdir, f"{figname}")
                for form in formats:
                    fig.savefig(f"{filename}.{form}", bbox_inches="tight", dpi=dpi)
                if close_figs:
                    plt.close(fig)

    def adjust_ylim(
        self,
        ymin: float = None,
        ymax: float = None,
        ax_ids: List[str] = None,
    ) -> None:
        """
        Adjust the ylim of matplotlib figures generated by analysis object.

        Parameters
        ----------
        ymin
            The bottom ylim in data coordinates. Passing None leaves the limit unchanged.
        ymax
            The top ylim in data coordinates. Passing None leaves the limit unchanged.
        ax_ids
            A list of ax_ids specifying what axes to adjust. Passing None results in
            all axes of an analysis object being adjusted.
        """
        axs = self.axs_mpl
        if ax_ids is None:
            ax_ids = axs.keys()

        for ax_id, ax in axs.items():
            if ax_id in ax_ids:
                ax.set_ylim(ymin, ymax)

    def adjust_xlim(
        self,
        xmin: float = None,
        xmax: float = None,
        ax_ids: List[str] = None,
    ) -> None:
        """
        Adjust the xlim of matplotlib figures generated by analysis object.

        Parameters
        ----------
        xmin
            The bottom xlim in data coordinates. Passing None leaves the limit unchanged.
        xmax
            The top xlim in data coordinates. Passing None leaves the limit unchanged.
        ax_ids
            A list of ax_ids specifying what axes to adjust. Passing None results in
            all axes of an analysis object being adjusted.
        """
        axs = self.axs_mpl
        if ax_ids is None:
            ax_ids = axs.keys()

        for ax_id, ax in axs.items():
            if ax_id in ax_ids:
                ax.set_xlim(xmin, xmax)

    def adjust_clim(
        self,
        vmin: float,
        vmax: float,
        ax_ids: List[str] = None,
    ) -> None:
        """
        Adjust the clim of matplotlib figures generated by analysis object.

        Parameters
        ----------
        vmin
            The bottom vlim in data coordinates. Passing None leaves the limit unchanged.
        vmax
            The top vlim in data coordinates. Passing None leaves the limit unchanged.
        ax_ids
            A list of ax_ids specifying what axes to adjust. Passing None results in
            all axes of an analysis object being adjusted.
        """
        axs = self.axs_mpl
        if ax_ids is None:
            ax_ids = axs.keys()

        for ax_id, ax in axs.items():
            if ax_id in ax_ids:
                # For plots created with `imshow` or `pcolormesh`
                for im_or_col in (
                    *ax.get_images(),
                    *(c for c in ax.collections if isinstance(c, QuadMesh)),
                ):
                    c_ax = im_or_col.colorbar.ax

                im_or_col.set_clim(vmin, vmax)


class Basic1DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):

        ys = set(self.dset_raw.keys())
        ys.discard("x0")
        for yi in ys:
            fig, ax = plt.subplots()
            fig_id = f"Line plot x0-{yi}"

            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax

            qpl.plot_basic_1d(
                ax=ax,
                x=self.dset_raw["x0"].values,
                xlabel=self.dset_raw["x0"].attrs["long_name"],
                xunit=self.dset_raw["x0"].attrs["units"],
                y=self.dset_raw[f"{yi}"].values,
                ylabel=self.dset_raw[f"{yi}"].attrs["long_name"],
                yunit=self.dset_raw[f"{yi}"].attrs["units"],
            )

            fig.suptitle(
                f"x0-{yi} {self.dset_raw.attrs['name']}\ntuid: {self.dset_raw.attrs['tuid']}"
            )


class Basic2DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):
        ys = set(self.dset_raw.keys())
        ys.discard("x0")
        ys.discard("x1")

        for yi in ys:
            fig, ax = plt.subplots()
            fig_id = f"Heatmap x0x1-{yi}"

            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax

            qpl.plot_2d_grid(
                x=self.dset_raw["x0"],
                y=self.dset_raw["x1"],
                z=self.dset_raw[f"{yi}"],
                xlabel=self.dset_raw["x0"].attrs["long_name"],
                xunit=self.dset_raw["x0"].attrs["units"],
                ylabel=self.dset_raw["x1"].attrs["long_name"],
                yunit=self.dset_raw["x1"].attrs["units"],
                zlabel=self.dset_raw[f"{yi}"].attrs["long_name"],
                zunit=self.dset_raw[f"{yi}"].attrs["units"],
                ax=ax,
            )

            fig.suptitle(
                f"x0x1-{yi} {self.dset_raw.attrs['name']}\ntuid: {self.dset_raw.attrs['tuid']}"
            )


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


def flatten_lmfit_modelresult(model):
    """
    Flatten an lmfit model result to a dictionary in order to be able to save it to disk.

    Notes
    -----
    We use this method as opposed to :func:`lmfit.model.save_modelresult` as the
    corresponding :func:`lmfit.model.load_modelresult` cannot handle loading data with
    a custom fit function.
    """
    assert isinstance(model, (lmfit.model.ModelResult, lmfit.minimizer.MinimizerResult))
    dic = OrderedDict()
    dic["success"] = model.success
    dic["message"] = model.message
    dic["params"] = {}
    for param_name in model.params:
        dic["params"][param_name] = {}
        param = model.params[param_name]
        for k in param.__dict__:
            if not k.startswith("_") and k not in ["from_internal"]:
                dic["params"][param_name][k] = getattr(param, k)
        dic["params"][param_name]["value"] = getattr(param, "value")
    return dic
