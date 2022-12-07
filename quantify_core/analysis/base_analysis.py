# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing the analysis abstract base class and several basic analyses."""
from __future__ import annotations
from dataclasses import dataclass

import json
import logging
import os
import warnings
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from functools import wraps
from pathlib import Path
from textwrap import wrap
from typing import List, Union, Dict

import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display
from matplotlib.collections import QuadMesh
from methodtools import lru_cache
from qcodes.utils.helpers import NumpyJSONEncoder
from uncertainties import ufloat

from quantify_core.data.handling import (
    DATASET_NAME,
    PROCESSED_DATASET_NAME,
    QUANTITIES_OF_INTEREST_NAME,
    create_exp_folder,
    get_datadir,
    get_latest_tuid,
    load_dataset,
    locate_experiment_container,
    to_gridded_dataset,
    write_dataset,
)
from quantify_core.data.types import TUID
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import adjust_axeslabels_SI, set_cbarlabel

from .types import AnalysisSettings

FIGURES_LRU_CACHE_SIZE = 8


@dataclass
class _FiguresMplCache:
    __slots__ = ["figs", "axes", "initialized"]
    figs: Dict[str, matplotlib.figure.Figure]
    axes: Dict[str, matplotlib.axes.Axes]
    initialized: bool


# global configurations at the level of the analysis module
settings = AnalysisSettings(
    {
        "mpl_dpi": 450,  # define resolution of some matplotlib output formats
        # svg is superior but at least OneNote does not support it
        "mpl_fig_formats": ["png", "svg"],
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

        from quantify_core.analysis import base_analysis as ba
        ba.settings["mpl_dpi"] = 300  # set resolution of matplotlib figures
"""


class AnalysisSteps(Enum):
    """
    An enumerate of the steps executed by the :class:`~BaseAnalysis` (and the default
    for subclasses).

    The involved steps are specified below.

    .. jupyter-execute::
        :hide-code:

        from quantify_core.analysis import base_analysis as ba
        print(ba.analysis_steps_to_str(ba.AnalysisSteps))

    .. include:: examples/analysis.base_analysis.AnalysisSteps.rst.txt

    .. tip::

        A custom analysis flow (e.g. inserting new steps) can be created by implementing
        an object similar to this one and overriding the
        :obj:`~BaseAnalysis.analysis_steps`.
    """

    # Variables must start with a letter but we want them to have sorted names
    # for auto-complete to indicate the execution order
    STEP_1_PROCESS_DATA = "process_data"
    STEP_2_RUN_FITTING = "run_fitting"
    STEP_3_ANALYZE_FIT_RESULTS = "analyze_fit_results"
    STEP_4_CREATE_FIGURES = "create_figures"
    STEP_5_ADJUST_FIGURES = "adjust_figures"
    STEP_6_SAVE_FIGURES = "save_figures"
    STEP_7_SAVE_QUANTITIES_OF_INTEREST = "save_quantities_of_interest"
    STEP_8_SAVE_PROCESSED_DATASET = "save_processed_dataset"
    STEP_9_SAVE_FIT_RESULTS = "save_fit_results"


class AnalysisMeta(ABCMeta):
    """Metaclass, whose purpose is to avoid storing large amount of figure in memory.

    By convention, analysis object stores figures in ``self.figs_mpl`` and
    ``self.axs_mpl`` dictionaries. This causes troubles for long-running operations,
    because figures are all in memory and eventually this uses all available memory of
    the PC. In order to avoid it, :meth:`BaseAnalysis.create_figures` and its
    derivatives are patched so that all the figures are put in LRU cache and
    reconstructed upon request to :attr:`BaseAnalysis.figs_mpl` or
    :attr:`BaseAnalysis.axs_mpl` if they were removed from the cache.

    Provided that analyses subclasses follow convention of figures being created in
    :meth:`BaseAnalysis.create_figures`, this approach should solve the memory issue
    and preserve reverse compatibility with present code.
    """

    def __new__(cls, name, bases, namespace, /, **kwargs):
        if "create_figures" in namespace.keys():
            namespace = dict(namespace)
            create_figures_orig = namespace.pop("create_figures")

            @wraps(create_figures_orig)
            def create_figures_patched(self):
                self._analyses_figures_cache().initialized = True
                self._creating_figures = True
                create_figures_orig(self)
                self._creating_figures = False

            def _figs_axs_mpl(self):
                cache = self._analyses_figures_cache()
                if not cache.initialized and not self._creating_figures:
                    create_figures_patched(self)
                return cache

            def figs_mpl(self):
                return self._figs_axs_mpl.figs

            def axs_mpl(self):
                return self._figs_axs_mpl.axes

            namespace["_creating_figures"] = False
            namespace["_figs_axs_mpl"] = property(_figs_axs_mpl)
            namespace["figs_mpl"] = property(figs_mpl)
            namespace["axs_mpl"] = property(axs_mpl)
            namespace["create_figures"] = create_figures_patched

        return super().__new__(cls, name, bases, namespace, **kwargs)


class BaseAnalysis(metaclass=AnalysisMeta):
    """A template for analysis classes."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        dataset: xr.Dataset = None,
        tuid: Union[TUID, str] = None,
        label: str = "",
        settings_overwrite: dict = None,
        plot_figures: bool = True,
    ):
        """
        Initializes the variables used in the analysis and to which data is stored.

        .. warning::

            We highly discourage overriding the class initialization.
            If the analysis requires the user passing in any arguments, the
            :meth:`~quantify_core.analysis.base_analysis.BaseAnalysis.run()` should be
            overridden and extended (see its docstring for an example).

        .. rubric:: Settings schema:

        .. jsonschema:: schemas/AnalysisSettings.json#/configurations

        Parameters
        ----------
        dataset:
            an unprocessed (raw) quantify dataset to perform the analysis on.
        tuid:
            if no dataset is specified, will look for the dataset with the matching tuid
            in the data directory.
        label:
            if no dataset and no tuid is provided, will look for the most recent dataset
            that contains "label" in the name.
        settings_overwrite:
            A dictionary containing overrides for the global
            `base_analysis.settings` for this specific instance.
            See `Settings schema` above for available settings.
        plot_figures:
            Option to create and save figures for analysis.
        """
        # NB at least logging.basicConfig() needs to be called in the python kernel
        # in order to see the logger messages
        self.logger = logging.getLogger(self.name)

        self.label = label
        self.tuid = tuid

        # Allows individual setting per analysis instance
        # with defaults from global settings
        self.settings_overwrite = deepcopy(settings)
        # NB this also runs validation against the corresponding schema
        self.settings_overwrite.update(settings_overwrite or {})

        # Used to have access to a reference of the raw dataset, see also
        # self.extract_data
        self.dataset = dataset

        # Initialize an empty dataset for the processed data.
        # This dataset will be overwritten during the analysis.
        self.dataset_processed = xr.Dataset()

        # A dictionary to contain the outputs of any custom analysis
        self.analysis_result = {}

        # To be populated by a subclass
        self.quantities_of_interest = {}

        self.fit_results = {}

        self.plot_figures = plot_figures

    analysis_steps = AnalysisSteps
    """
    Defines the steps of the analysis specified as an :class:`~enum.Enum`.
    Can be overridden in a subclass in order to define a custom analysis flow.
    See :class:`~quantify_core.analysis.base_analysis.AnalysisSteps` for a template.
    """

    @classmethod
    def load_fit_result(cls, tuid: TUID, fit_name: str) -> lmfit.model.ModelResult:
        """
        Load a saved :code:`lmfit.model.ModelResult` object from file. For analyses
        that use custom fit functions, the :code:`cls.fit_function_definitions` object
        must be defined in the subclass for that analysis.

        Parameters
        ------------
        tuid:
            The TUID reference of the saved analysis.
        fit_name:
            The name of the fit result to be loaded.

        Returns
        ------------
        :
            The lmfit model result object.
        """
        analysis_dir = cls._get_analysis_dir(
            tuid=tuid, name=cls.__name__, create_missing=False
        )

        if not os.path.isdir(analysis_dir):
            raise FileNotFoundError(
                f"Analysis not found for this experiment ({analysis_dir} not found)."
            )

        results_dir = cls._get_results_dir(
            analysis_dir=analysis_dir, create_missing=False
        )

        if not os.path.isdir(results_dir):
            raise FileNotFoundError(
                f"No fit results found for this analysis ({results_dir} not found)."
            )

        result = lmfit.model.load_modelresult(
            os.path.join(results_dir, f"{fit_name}.txt")
        )
        return result

    @property
    def name(self):
        """The name of the analysis, used in data saving."""
        # used to store data and figures resulting from the analysis. Can be overwritten
        return self.__class__.__name__

    @staticmethod
    def _get_analysis_dir(tuid: TUID, name: str, create_missing: bool = True):
        """
        Generate an analysis dir based on a given tuid and analysis class name.

        Parameters
        ------------
        tuid:
            TUID of the analysis dir.
        name:
            The name of the analysis class.
        create_missing:
            If True, create the analysis dir if it does not already exist.

        """
        exp_folder = Path(locate_experiment_container(tuid, get_datadir()))
        analysis_dir = os.path.join(exp_folder, f"analysis_{name}")
        if create_missing:
            if not os.path.isdir(analysis_dir):
                os.makedirs(analysis_dir)

        return analysis_dir

    @property
    def analysis_dir(self):
        """
        Analysis dir based on the tuid of the analysis class instance.
        Will create a directory if it does not exist.
        """
        if self.tuid is None:
            raise ValueError("Unknown TUID, cannot determine the analysis directory.")

        return self._get_analysis_dir(tuid=self.tuid, name=self.name)

    @staticmethod
    def _get_results_dir(analysis_dir: str, create_missing: bool = True):
        """
        Generate an results dir based on a given analysis dir path.

        Parameters
        ------------
        analysis_dir:
            The path of the analysis directory.
        create_missing:
            If True, create the analysis dir if it does not already exist.

        """
        results_dir = os.path.join(analysis_dir, "fit_results")
        if create_missing:
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

        return results_dir

    @lru_cache(maxsize=FIGURES_LRU_CACHE_SIZE)
    def _analyses_figures_cache(self):
        return _FiguresMplCache({}, {}, False)

    @property
    def results_dir(self):
        """
        Analysis dirrectory for this analysis.
        Will create a directory if it does not exist.
        """
        return self._get_results_dir(analysis_dir=self.analysis_dir)

    def run(self) -> BaseAnalysis:
        """
        This function is at the core of all analysis. It calls
        :meth:`~quantify_core.analysis.base_analysis.BaseAnalysis.execute_analysis_steps`
        which executes all the methods defined in the

        First step of any analysis is always extracting data, that is not configurable.
        Errors in `extract_data()` are considered fatal for analysis.
        Later steps are configurable by overriding
        :attr:`~quantify_core.analysis.base_analysis.BaseAnalysis.analysis_steps`.
        Exceptions in these steps are logged and suppressed and analysis is considered
        partially successful.

        This function is typically called right after instantiating an analysis class.

        .. include:: examples/analysis.base_analysis.BaseAnalysis.run_custom_analysis_args.rst.txt

        Returns
        -------
        :
            The instance of the analysis object so that
            :meth:`~quantify_core.analysis.base_analysis.BaseAnalysis.run()`
            returns an analysis object.
            You can initialize, run and assign it to a variable on a
            single line:, e.g. :code:`a_obj = MyAnalysis().run()`.
        """  # pylint: disable=line-too-long
        # The following two lines must be included when when implementing a custom
        # analysis that requires passing in some (optional) arguments.
        self.execute_analysis_steps()
        return self

    def execute_analysis_steps(self):
        """
        Executes the methods corresponding to the analysis steps as defined by the
        :attr:`~quantify_core.analysis.base_analysis.BaseAnalysis.analysis_steps`.

        Intended to be called by `.run` when creating a custom analysis that requires
        passing analysis configuration arguments to
        :meth:`~quantify_core.analysis.base_analysis.BaseAnalysis.run`.
        """
        self.logger.info(f"Executing `.analysis_steps` of {self.name}")
        self.logger.info(f"extracting data: {self.extract_data}")
        self.extract_data()

        for i, method in enumerate(self.get_flow(), start=1):
            self.logger.info(f"executing step {i}: {method}")
            try:
                method()
            except Exception:  # pylint: disable=broad-except
                self.logger.exception(
                    f"Exception was raised while executing analysis step {i} "
                    f'("{method}"). Terminating analysis and returning partial result.'
                )
                return

    # pylint: disable=no-member
    def get_flow(self) -> tuple:
        """
        Returns a tuple with the ordered methods to be called by run analysis.
        Only return the figures methods if :code:`self.plot_figures` is :code:`True`.
        """
        if self.plot_figures:
            return tuple(getattr(self, elm.value) for elm in self.analysis_steps)

        return tuple(
            getattr(self, elm.value)
            for elm in self.analysis_steps
            if "figures" not in elm.value
        )

    def extract_data(self):
        """
        If no `dataset` is provided, populates :code:`.dataset` with data from
        the experiment matching the tuid/label.

        This method should be overwritten if an analysis does not relate to a single
        datafile.
        """
        if self.dataset is not None:
            # pylint: disable=fixme
            # FIXME: to be replaced by a validate_dateset see #187
            if "tuid" not in self.dataset.attrs.keys():
                raise AttributeError('Invalid dataset, missing the "tuid" attribute')

            self.tuid = TUID(self.dataset.attrs["tuid"])
            # an experiment container is required to store output of the analysis.
            # it is possible for this not to exist for a custom dataset as it can
            # come from a source outside of the data directory.
            try:
                locate_experiment_container(self.tuid)
            except FileNotFoundError:
                # if the file did not exist, an experiment folder is created
                # and a copy of the dataset is stored there.
                exp_folder = create_exp_folder(tuid=self.tuid, name=self.dataset.name)
                write_dataset(
                    path=os.path.join(exp_folder, DATASET_NAME),
                    dataset=self.dataset,
                )

        if self.dataset is None:
            # if no TUID is specified use the label to search for the latest file with
            # a match.
            if self.tuid is None:
                self.tuid = get_latest_tuid(contains=self.label)
            # Keep a reference to the original dataset.
            self.dataset = load_dataset(tuid=self.tuid)

    def process_data(self):
        """
        To be implemented by subclasses.

        Should process, e.g., reshape, filter etc. the data
        before starting the analysis.
        """

    def run_fitting(self):
        """
        To be implemented by subclasses.

        Should create fitting model(s) and fit data to the model(s) adding the result
        to the :code:`.fit_results` dictionary.
        """

    def _add_fit_res_to_qoi(self):
        if len(self.fit_results) > 0:
            self.quantities_of_interest["fit_result"] = {}
            for fr_name, fit_result in self.fit_results.items():
                res = flatten_lmfit_modelresult(fit_result)
                self.quantities_of_interest["fit_result"][fr_name] = res

    def analyze_fit_results(self):
        """
        To be implemented by subclasses.

        Should analyze and process the :code:`.fit_results` and add the quantities of
        interest to the :code:`.quantities_of_interest` dictionary.
        """

    def create_figures(self):
        """
        To be implemented by subclasses.

        Should generate figures of interest. matplolib figures and axes objects should
        be added to the :code:`.figs_mpl` and :code:`axs_mpl` dictionaries.,
        respectively.
        """

    def adjust_figures(self):
        """
        Perform global adjustments after creating the figures but
        before saving them.

        By default applies `mpl_exclude_fig_titles` and `mpl_transparent_background`
        from :code:`.settings_overwrite` to any matplotlib figures in
        :code:`.figs_mpl`.

        Can be extended in a subclass for additional adjustments.
        """
        for fig in self.figs_mpl.values():
            if self.settings_overwrite["mpl_exclude_fig_titles"]:
                # Remove the experiment name and tuid from figures
                fig.suptitle(r"")
            if self.settings_overwrite["mpl_transparent_background"] is True:
                # Set transparent background on figures
                fig.patch.set_alpha(0)
            else:
                fig.patch.set_alpha(1)

    def save_processed_dataset(self):
        """
        Saves a copy of the processed :code:`.dataset_processed` in the analysis folder
        of the experiment.
        """
        if self.dataset_processed is not None:
            write_dataset(
                Path(self.analysis_dir) / PROCESSED_DATASET_NAME, self.dataset_processed
            )

    def save_quantities_of_interest(self):
        """
        Saves the :code:`.quantities_of_interest` as a JSON file in the analysis
        directory.

        The file is written using :func:`json.dump` with the
        :class:`qcodes.utils.helpers.NumpyJSONEncoder` custom encoder.
        """
        self._add_fit_res_to_qoi()

        with open(
            os.path.join(self.analysis_dir, QUANTITIES_OF_INTEREST_NAME),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.quantities_of_interest, file, cls=NumpyJSONEncoder, indent=4)

    def save_fit_results(self):
        """
        Saves the :code:`lmfit.model.model_result` objects for each fit in a
        sub-directory within the analysis directory
        """

        for fr_name, fit_result in self.fit_results.items():
            path = os.path.join(self.results_dir, f"{fr_name}.txt")
            lmfit.model.save_modelresult(fit_result, path)

    def save_figures(self):
        """
        Saves figures to disk. By default saves matplotlib figures.

        Can be overridden or extended to make use of other plotting packages.
        """
        self.save_figures_mpl()

    def save_figures_mpl(self, close_figs: bool = True):
        """
        Saves all the matplotlib figures in the :code:`.figs_mpl` dict.

        Parameters
        ----------
        close_figs
            If True, closes matplotlib figures after saving.
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
        Displays figures in :code:`.figs_mpl` in all frontends.
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
            The bottom ylim in data coordinates. Passing :code:`None` leaves the
            limit unchanged.
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
            The bottom xlim in data coordinates. Passing :code:`None` leaves the limit
            unchanged.
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
            The bottom vlim in data coordinates. Passing :code:`None` leaves the limit
            unchanged.
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


class BasicAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):
        """
        Creates a line plot x vs y for every data variable yi and coordinate xi in the
        dataset.
        """

        # NB we do not use `to_gridded_dataset` because that can potentially drop
        # repeated measurement of the same x0_i setpoint (e.g., AllXY experiment)
        dataset = self.dataset
        # for compatibility with older datasets
        # in case "x0" is not a coordinate we use "dim_0"
        coords = list(dataset.coords)
        dims = list(dataset.dims)
        plot_against = coords if coords else (dims if dims else [None])
        for idx, xi in enumerate(plot_against):
            for yi, yvals in dataset.data_vars.items():
                # for compatibility with older datasets, do not plot "x0" vs "x0"
                if yi.startswith("y"):
                    fig, ax = plt.subplots()

                    fig_id = f"Line plot x{idx}-{yi}"

                    yvals.plot.line(ax=ax, x=xi, marker=".")

                    adjust_axeslabels_SI(ax)

                    qpl.set_suptitle_from_dataset(fig, self.dataset, f"x{idx}-{yi}")

                    # add the figure and axis to the dicts for saving
                    self.figs_mpl[fig_id] = fig
                    self.axs_mpl[fig_id] = ax


class Basic1DAnalysis(BasicAnalysis):
    """
    Deprecated. Alias of :class:`~quantify_core.analysis.base_analysis.BasicAnalysis`
    for backwards compatibility.
    """

    def run(self) -> BaseAnalysis:
        warnings.warn("Use `BasicAnalysis`", category=FutureWarning)
        return super().run()


class Basic2DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):

        gridded_dataset = to_gridded_dataset(self.dataset)

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

            qpl.set_suptitle_from_dataset(fig, self.dataset, f"x0x1-{yi}")

            # add the figure and axis to the dicts for saving
            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax

        # plot linecuts of the data
        for yi, yvals in gridded_dataset.data_vars.items():
            fig, ax = plt.subplots()
            fig_id = f"Linecuts x0x1-{yi}"

            lines = yvals.plot.line(x="x0", hue="x1", ax=ax)
            # Change the color and labels of the line as we want to tweak this with
            # respect to xarray default.
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
                    gridded_dataset["x1"].long_name,
                    gridded_dataset["x1"].units,
                ),
                ncol=max(len(gridded_dataset["x1"]) // 8, 1),
            )
            # adjust the labels to be SI aware
            adjust_axeslabels_SI(ax)

            qpl.set_suptitle_from_dataset(fig, self.dataset, f"x0x1-{yi}")

            # add the figure and axis to the dicts for saving
            self.figs_mpl[fig_id] = fig
            self.axs_mpl[fig_id] = ax


def flatten_lmfit_modelresult(model):
    """
    Flatten an lmfit model result to a dictionary in order to be able to save
    it to disk.

    Notes
    -----
    We use this method as opposed to :func:`~lmfit.model.save_modelresult` as the
    corresponding :func:`~lmfit.model.load_modelresult` cannot handle loading data with
    a custom fit function.
    """
    assert isinstance(model, (lmfit.model.ModelResult, lmfit.minimizer.MinimizerResult))
    dic = {}
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


def lmfit_par_to_ufloat(param: lmfit.parameter.Parameter):
    """
    Safe conversion of an :class:`lmfit.parameter.Parameter` to
    :code:`uncertainties.ufloat(value, std_dev)`.

    This function is intended to be used in custom analyses to avoid errors when an
    `lmfit` fails and the `stderr` is :code:`None`.

    Parameters
    ----------
    param:
        The :class:`~lmfit.parameter.Parameter` to be converted

    Returns
    -------
    :class:`!uncertainties.UFloat` :
        An object representing the value and the uncertainty of the parameter.
    """

    value = param.value
    stderr = np.nan if param.stderr is None else param.stderr

    return ufloat(value, stderr)


def check_lmfit(fit_res: lmfit.model.ModelResult) -> str:
    """
    Check that `lmfit` was able to successfully return a valid fit, and give
    a warning if not.

    The function looks at `lmfit`'s success parameter, and also checks whether
    the fit was able to obtain valid error bars on the fitted parameters.

    Parameters
    ----------
    fit_res:
        The :class:`~lmfit.model.ModelResult` object output by `lmfit`

    Returns
    -------
    :
        A warning message if there is a problem with the fit.
    """
    if not fit_res.success:
        fit_warning = "fit failed. lmfit was not able to fit the data."
        warnings.warn(fit_warning)
        return "Warning: " + fit_warning

    errorbars_failed = not fit_res.errorbars
    if errorbars_failed:
        fit_warning = (
            "lmfit could not find a good fit. Fitted parameters may not be accurate."
        )
        warnings.warn(fit_warning)
        return "Warning: " + fit_warning

    return None


def wrap_text(text, width=35, replace_whitespace=True, **kwargs):
    """
    A text wrapping (braking over multiple lines) utility.

    Intended to be used with :func:`~quantify_core.visualization.mpl_plotting.plot_textbox`
    in order to avoid too wide figure when, e.g.,
    :func:`~quantify_core.analysis.base_analysis.check_lmfit` fails and a warning message is
    generated.

    For usage see, for example, source code of
    :meth:`~quantify_core.analysis.single_qubit_timedomain.T1Analysis.create_figures`.

    Parameters
    ----------
    text:
        The text string to be wrapped over several lines.
    width:
        Maximum line width in characters.
    kwargs:
        Any other keyword arguments to be passed to :func:`textwrap.wrap`.

    Returns
    -------
    :
        The wrapped text (or :code:`None` if text is :code:`None`).
    """
    if text is not None:
        # make sure existing line breaks are preserved
        text_lines = text.split("\n")
        wrapped_text = "\n".join(
            "\n".join(
                wrap(line, width=width, replace_whitespace=replace_whitespace, **kwargs)
            )
            for line in text_lines
        )

        return wrapped_text


def analysis_steps_to_str(
    analysis_steps: Enum, class_name: str = BaseAnalysis.__name__
) -> str:
    """
    A utility for generating the docstring for the analysis steps

    Parameters
    ----------
    analysis_steps:
        An :class:`~enum.Enum` similar to
        :class:`quantify_core.analysis.base_analysis.AnalysisSteps`.
    class_name:
        The class name that has the `analysis_steps` methods and for which the
        `analysis_steps` are intended.

    Returns
    -------
    :
        A formatted string version of the `analysis_steps` and corresponding methods.
    """
    col0 = tuple(element.name for element in analysis_steps)
    col1 = tuple(element.value for element in analysis_steps)

    header_r = "# <STEP>"
    header_l = "<corresponding class method>"
    sep = "  # "

    col0_len = max(map(len, col0 + (header_r,)))
    col0_len += len(analysis_steps.__name__) + 1

    string = f"{header_r:<{col0_len}}{sep}{header_l}\n\n"
    string += "\n".join(  # NB the `+ '.' +` is not redundant
        f"{analysis_steps.__name__ + '.' + name:<{col0_len}}{sep}{class_name}.{value}"
        for name, value in zip(col0, col1)
    )

    return string
