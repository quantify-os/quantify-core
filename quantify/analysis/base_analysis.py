# -----------------------------------------------------------------------------
# Description:    Module containing base analysis.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
"""Analysis abstract base class and several basic analyses."""
from __future__ import annotations
import os
import json
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import List
from enum import Enum
from pathlib import Path
import logging

from IPython.display import display
import numpy as np
import xarray as xr
import lmfit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from qcodes.utils.helpers import NumpyJSONEncoder

from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import adjust_axeslabels_SI, set_cbarlabel
from quantify.data.handling import (
    load_dataset,
    get_latest_tuid,
    get_datadir,
    write_dataset,
    locate_experiment_container,
    to_gridded_dataset,
)
from .types import AnalysisSettings

# global configurations at the level of the analysis module
settings = AnalysisSettings(
    {
        "mpl_dpi": 450,  # define resolution of some matplotlib output formats
        "mpl_fig_formats": [
            "png",
            "svg",
        ],  # svg is superior but at least OneNote does not support it
        "mpl_exclude_fig_titles": False,
        "mpl_transparent_background": True,
    }
)
"""
For convenience the analysis framework provides a set of global settings.

For available settings see :class:`~BaseAnalysis`.
These can be overwritten for each instance of an analysis.

.. admonition:: Example

    .. jupyter-execute::

        from quantify.analysis import base_analysis as ba
        ba.settings["mpl_dpi"] = 300  # set resolution of matplotlib figures
"""


class AnalysisSteps(Enum):
    """
    An enumerate of the steps executed by the :class:`~BaseAnalysis` (and its subclasses).

    The involved steps are specified below.

    .. jupyter-execute::
        :hide-code:

        from quantify.analysis import base_analysis as ba
        print(ba.analysis_steps_to_str(ba.AnalysisSteps))

    .. include:: ./docstring_examples/quantify.analysis.base_analysis.AnalysisSteps.rst.txt

    .. tip::

        A custom analysis flow (e.g. inserting new steps) can be created be implementing
        an object similar to this one and overloading the :obj:`~BaseAnalysis.analysis_steps`.
    """

    # Variables must start with a letter but we want them have sorted names
    # for auto-complete
    S00_EXTRACT_DATA = "extract_data"
    S01_PROCESS_DATA = "process_data"
    S02_PREPARE_FITTING = "prepare_fitting"
    S03_RUN_FITTING = "run_fitting"
    S04_ANALYZE_FIT_RESULTS = "analyze_fit_results"
    S05_CREATE_FIGURES = "create_figures"
    S06_ADJUST_FIGURES = "adjust_figures"
    S07_SAVE_FIGURES = "save_figures"
    S08_SAVE_QUANTITIES_OF_INTEREST = "save_quantities_of_interest"
    # blocked by #161
    # S09_SAVE_PROCESSED_DATASET = "save_processed_dataset"


class BaseAnalysis(ABC):
    """A template for analysis classes."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        label: str = "",
        tuid: str = None,
        interrupt_before: AnalysisSteps = None,
        settings_overwrite: dict = None,
    ):
        """
        Initializes the variables used in the analysis and to which data is stored.


        Parameters
        ----------
        label:
            Will look for a dataset that contains "label" in the name.
        tuid:
            If specified, will look for the dataset with the matching tuid.
        interrupt_before:
            Stops `run_analysis` before executing the specified step.
        settings_overwrite:
            A dictionary containing overrides for the global
            `base_analysis.settings` for this specific instance.
            See table below for available settings.


        .. jsonschema:: schemas/AnalysisSettings.json#/configurations
        """
        self.logger = logging.getLogger(self.name)

        self.label = label
        self.tuid = tuid
        self.interrupt_before = interrupt_before

        # Allows individual setting per analysis instance
        # with defaults from global settings
        self.settings_overwrite = deepcopy(settings)
        self.settings_overwrite.update(settings_overwrite or dict())

        # This will be overwritten
        self.dataset = None
        # Used to save a reference of the raw dataset
        self.dataset_raw = None
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
        This function is at the core of all analysis and defines the flow of methods to call.

        This function is typically called at the end of __init__.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            step_stop=self.interrupt_before,
            step_stop_inclusive=False,
        )

        self.logger.info(f"Executing run_analysis of {self.name}")
        for i, method in enumerate(flow_methods):
            self.logger.info(f"execution step {i}: {method}")
            try:
                method()
            except Exception as e:
                raise RuntimeError(
                    "An exception occurred while executing "
                    f"{method}.\n\n"  # point to the culprit
                    "Use `interrupt_before='<analysis step>'` to run a partial analysis. "
                    "Method names:\n"
                    f"{analysis_steps_to_str(analysis_steps=self.analysis_steps, class_name=self.__class__.__name__)}"
                ) from e  # and raise the original exception

    def continue_analysis_from(self, step: AnalysisSteps):
        """
        Runs the analysis starting from specified method.

        The methods are called in the same order as in :meth:`~run_analysis`.
        Useful when the analysis interrupted at some stage with `interrupt_before`.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            step_start=step,
            step_start_inclusive=True,  # self.step will be executed
        )

        for method in flow_methods:
            method()

    def continue_analysis_after(self, step: AnalysisSteps):
        """
        Runs the analysis starting from specified method.

        The methods are called in the same order as in :meth:`~run_analysis`.
        Useful when the analysis interrupted at some stage with `interrupt_before`.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            step_start=step,
            step_start_inclusive=False,  # self.step will not be executed
        )

        for method in flow_methods:
            method()

    @property
    def analysis_steps(self) -> AnalysisSteps:
        """
        Returns an Enum subclass that defines the steps of the analysis.

        Can be overloaded in a subclass in order to define a custom analysis flow.
        """
        return AnalysisSteps

    def get_flow(self) -> tuple:
        """
        Returns a tuple with the ordered methods to be called by run analysis.
        """
        return tuple(getattr(self, elm.value) for elm in self.analysis_steps)

    def extract_data(self):
        """
        Populates `self.dataset_raw` with data from the experiment matching the tuid/label.

        This method should be overwritten if an analysis does not relate to a single
        datafile.
        """

        # if no TUID is specified use the label to search for the latest file with a match.
        if self.tuid is None:
            self.tuid = get_latest_tuid(contains=self.label)

        # Keep a reference to the original dataset.
        self.dataset_raw = load_dataset(tuid=self.tuid)
        # Initialize an empty dataset for the processed data.
        self.dataset = xr.Dataset()

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
            if self.settings_overwrite["mpl_exclude_fig_titles"]:
                # Remove the experiment name and tuid from figures
                fig.suptitle(r"")
            if self.settings_overwrite["mpl_transparent_background"]:
                # Set transparent background on figures
                fig.patch.set_alpha(0)

    def save_processed_dataset(self):
        """
        Saves a copy of the (processed) self.dataset in the analysis folder of the experiment.
        """

        # if statement exist to be compatible with child classes that do not load data
        # onto the self.dataset object.
        if self.dataset is not None:
            dataset = self.dataset
            write_dataset(Path(self.analysis_dir) / "processed_dataset.hdf5", dataset)

    def save_figures(self):
        """
        Saves figures to disk. By default saves matplotlib figures.

        Can be overloaded to make use of other plotting packages.
        """
        self.save_figures_mpl()

    def save_figures_mpl(self, close_figs: bool = True):
        """
        Saves all the matplotlib figures in the :code:`figs_mpl` dict

        Parameters
        ----------
        close_figs
            If True, closes `matplotlib` figures after saving
        """
        dpi = self.settings_overwrite["mpl_dpi"]
        formats = self.settings_overwrite["mpl_fig_formats"]

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

    def display_figs_mpl(self):
        """
        Displays figures in self.figs_mpl in all frontends.
        """
        for fig in self.figs_mpl.values():
            display(fig)

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
                for image_or_collection in (
                    *ax.get_images(),
                    *(c for c in ax.collections if isinstance(c, QuadMesh)),
                ):
                    image_or_collection.set_clim(vmin, vmax)


class Basic1DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):

        gridded_dataset = to_gridded_dataset(self.dataset_raw)

        for yi, yvals in gridded_dataset.data_vars.items():
            fig, ax = plt.subplots()
            fig_id = f"Line plot x0-{yi}"
            # plotting works because it is an xarray with associated dimensions.
            yvals.plot(ax=ax, marker=".")
            adjust_axeslabels_SI(ax)

            fig.suptitle(
                f"x0-{yi} {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
            )

            # add the figure and axis to the dicts for saving
            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax


class Basic2DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):

        gridded_dataset = to_gridded_dataset(self.dataset_raw)

        # plot heatmaps of the data
        for yi, yvals in gridded_dataset.data_vars.items():
            fig, ax = plt.subplots()
            fig_id = f"Heatmap x0x1-{yi}"

            # transpose is required to have x0 on the xaxis and x1 on the y-axis
            quadmesh = yvals.transpose().plot(ax=ax)
            # adjust the labels to be SI aware
            adjust_axeslabels_SI(ax)
            set_cbarlabel(quadmesh.colorbar, yvals.long_name, yvals.units)
            # autodect degrees and radians to use circular colormap.
            qpl.set_cyclic_colormap(quadmesh, shifted=yvals.min() < 0, unit=yvals.units)

            fig.suptitle(
                f"x0x1-{yi} {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
            )

            # add the figure and axis to the dicts for saving
            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax

        # plot linecuts of the data
        for yi, yvals in gridded_dataset.data_vars.items():
            fig, ax = plt.subplots()
            fig_id = f"Linecuts x0x1-{yi}"

            lines = yvals.plot.line(x="x0", hue="x1", ax=ax)
            # Change the color and labels of the line as we want to tweak this with respect to xarray default.
            for line, z_value in zip(lines, np.array(gridded_dataset["x1"])):
                # use the default colormap specified
                cmap = matplotlib.cm.get_cmap()
                norm = matplotlib.colors.Normalize(
                    vmin=np.min(gridded_dataset["x1"]),
                    vmax=np.max(gridded_dataset["x1"]),
                )
                line.set_color(cmap(norm(z_value)))
                line.set_label(f"{z_value:.3g}")

            ax.legend(
                loc=(1.05, 0.0),
                title="{} ({})".format(
                    gridded_dataset["x1"].attrs["long_name"],
                    gridded_dataset["x1"].attrs["units"],
                ),
                ncol=max(len(gridded_dataset["x1"]) // 8, 1),
            )
            # adjust the labels to be SI aware
            adjust_axeslabels_SI(ax)

            fig.suptitle(
                f"x0x1-{yi} {self.dataset_raw.attrs['name']}\ntuid: {self.dataset_raw.attrs['tuid']}"
            )

            # add the figure and axis to the dicts for saving
            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax


def flatten_lmfit_modelresult(model):
    """
    Flatten an lmfit model result to a dictionary in order to be able to save it to disk.

    Notes
    -----
    We use this method as opposed to :func:`~lmfit.model.save_modelresult` as the
    corresponding :func:`~lmfit.model.load_modelresult` cannot handle loading data with
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


def analysis_steps_to_str(
    analysis_steps: Enum, class_name: str = BaseAnalysis.__name__
):
    """A utility for generating the docstring for the analysis steps"""
    col0 = tuple(element.name for element in analysis_steps)
    col1 = tuple(element.value for element in analysis_steps)

    header_r = "# <STEP>"
    header_l = "<corresponding class method>"
    sep = "  # "

    col0_len = max(map(len, col0 + (header_r,)))
    col0_len += len(analysis_steps.__name__) + 1

    string = f"{header_r:<{col0_len}}{sep}{header_l}\n\n"
    string += "\n".join(
        f"{analysis_steps.__name__+ '.' + name:<{col0_len}}{sep}{class_name + '.' + value}"
        for name, value in zip(col0, col1)
    )

    return string


def _get_modified_flow(
    flow_functions: tuple,
    step_start: AnalysisSteps = None,
    step_start_inclusive: bool = True,
    step_stop: AnalysisSteps = None,
    step_stop_inclusive: bool = True,
):
    step_names = [meth.__name__ for meth in flow_functions]

    if step_start:
        start_idx = step_names.index(step_start.value)
        if not step_start_inclusive:
            start_idx += 1
    else:
        start_idx = 0

    if step_stop:
        stop_idx = step_names.index(step_stop.value)
        if step_stop_inclusive:
            stop_idx += 1
    else:
        stop_idx = None

    flow_functions = flow_functions[start_idx:stop_idx]

    return flow_functions
