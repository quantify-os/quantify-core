# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Module containing the analysis abstract base class and several basic analyses."""
from __future__ import annotations
import os
import json
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import List, Union
from enum import Enum
from pathlib import Path
import logging
import inspect
import warnings

from IPython.display import display
import numpy as np
import xarray as xr
import lmfit
from uncertainties import ufloat
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from qcodes.utils.helpers import NumpyJSONEncoder

from quantify.visualization import mpl_plotting as qpl
from quantify.visualization.SI_utilities import adjust_axeslabels_SI, set_cbarlabel
from quantify.data.types import TUID
from quantify.data.handling import (
    load_dataset,
    get_latest_tuid,
    get_datadir,
    DATASET_NAME,
    write_dataset,
    create_exp_folder,
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
    An enumerate of the steps executed by the :class:`~BaseAnalysis` (and the default
    for subclasses).

    The involved steps are specified below.

    .. jupyter-execute::
        :hide-code:

        from quantify.analysis import base_analysis as ba
        print(ba.analysis_steps_to_str(ba.AnalysisSteps))

    .. include:: ./docstring_examples/quantify.analysis.base_analysis.AnalysisSteps.rst.txt

    .. tip::

        A custom analysis flow (e.g. inserting new steps) can be created by implementing
        an object similar to this one and overloading the
        :obj:`~BaseAnalysis.analysis_steps`.
    """  # pylint: disable=line-too-long

    # Variables must start with a letter but we want them have sorted names
    # for auto-complete
    STEP_0_EXTRACT_DATA = "extract_data"
    STEP_1_PROCESS_DATA = "process_data"
    STEP_2_RUN_FITTING = "run_fitting"
    STEP_3_ANALYZE_FIT_RESULTS = "analyze_fit_results"
    STEP_4_CREATE_FIGURES = "create_figures"
    STEP_5_ADJUST_FIGURES = "adjust_figures"
    STEP_6_SAVE_FIGURES = "save_figures"
    STEP_7_SAVE_QUANTITIES_OF_INTEREST = "save_quantities_of_interest"
    STEP_8_SAVE_PROCESSED_DATASET = "save_processed_dataset"


class BaseAnalysis(ABC):
    """A template for analysis classes."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        dataset_raw: xr.Dataset = None,
        tuid: str = None,
        label: str = "",
        settings_overwrite: dict = None,
    ):
        """
        Initializes the variables used in the analysis and to which data is stored.

        .. tip::

            For scripting/development/debugging purposes the
            :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run_until` can be used
            for a partial execution of the analysis. E.g.,

            .. jupyter-execute::

                from quantify.analysis.base_analysis import Basic1DAnalysis

                a_obj = Basic1DAnalysis(label="my experiment").run_until(
                    interrupt_before="extract_data"
                )

            **OR** use the corresponding members of the
            :attr:`~quantify.analysis.base_analysis.BaseAnalysis.analysis_steps`:

            .. jupyter-execute::

                a_obj = Basic1DAnalysis(label="my experiment").run_until(
                    interrupt_before=Basic1DAnalysis.analysis_steps.STEP_0_EXTRACT_DATA
                )

        Parameters
        ----------
        dataset_raw:
            an unprocessed (raw) quantify dataset to perform the analysis on.
        tuid:
            if no dataset is specified, will look for the dataset with the matching tuid
            in the datadirectory.
        label:
            if no dataset and no tuid is provided, will look for the most recent dataset
            that contains "label" in the name.
        settings_overwrite:
            A dictionary containing overrides for the global
            `base_analysis.settings` for this specific instance.
            See table below for available settings.


        .. jsonschema:: schemas/AnalysisSettings.json#/configurations
        """
        self.logger = logging.getLogger(self.name)

        self.label = label
        self.tuid = tuid

        # Allows individual setting per analysis instance
        # with defaults from global settings
        self.settings_overwrite = deepcopy(settings)
        self.settings_overwrite.update(settings_overwrite or dict())

        # Used to have access to a reference of the raw dataset, see also
        # self.extract_data
        self.dataset_raw = dataset_raw

        # Initialize an empty dataset for the processed data.
        # This dataset will be overwritten during the analysis.
        self.dataset = xr.Dataset()

        # To be populated by a subclass
        self.figs_mpl = OrderedDict()
        self.axs_mpl = OrderedDict()
        self.quantities_of_interest = OrderedDict()
        self.fit_res = OrderedDict()

        self._interrupt_before = None

    # Defines the steps of the analysis specified as an Enum.
    # Can be overloaded in a subclass in order to define a custom analysis flow.
    # See `AnalysisSteps` for a template.
    analysis_steps = AnalysisSteps

    @property
    def name(self):
        """The name of the analysis, used in data saving."""
        # used to store data and figures resulting from the analysis. Can be overwritten
        return self.__class__.__name__

    @property
    def analysis_dir(self):
        """
        Analysis dir based on the tuid. Will create a directory if it does not exist
        yet.
        """
        if self.tuid is None:
            raise ValueError("Unknown TUID, cannot determine the analysis directory.")
        # This is a property as it depends
        exp_folder = Path(locate_experiment_container(self.tuid, get_datadir()))
        analysis_dir = exp_folder / f"analysis_{self.name}"
        if not os.path.isdir(analysis_dir):
            os.makedirs(analysis_dir)

        return analysis_dir

    def run(self) -> BaseAnalysis:
        """
        This function is at the core of all analysis. It calls
        :meth:`~quantify.analysis.base_analysis.BaseAnalysis.execute_analysis_steps`
        which executes all the methods defined in the
        :attr:`~quantify.analysis.base_analysis.BaseAnalysis.analysis_steps`.

        This function is typically called right after instantiating an analysis class.

        .. include:: ./docstring_examples/quantify.analysis.base_analysis.BaseAnalysis.run_custom_analysis_args.rst.txt

        Returns
        -------
        :
            The instance of the analysis object so that
            :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run()`
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
        :attr:`~quantify.analysis.base_analysis.BaseAnalysis.analysis_steps`.

        Intended to be called by `.run` when creating a custom analysis that requires
        passing analysis configuration arguments to
        :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run`.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            step_stop=self._interrupt_before,  # can be set by .run_until
            step_stop_inclusive=False,
        )

        # Always reset so that it only has an effect when set by .run_until
        self._interrupt_before = None

        self.logger.info(f"Executing `.run()` of {self.name}")
        for i, method in enumerate(flow_methods):
            self.logger.info(f"execution step {i}: {method}")
            method()

    def run_from(self, step: Union[str, AnalysisSteps]):
        """
        Runs the analysis starting from the specified method.

        The methods are called in the same order as in
        :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run`.
        Useful when first running a partial analysis and continuing again.
        """
        flow_methods = _get_modified_flow(
            flow_functions=self.get_flow(),
            step_start=step,
            step_start_inclusive=True,  # self.step will be executed
        )

        for method in flow_methods:
            method()

    def run_until(self, interrupt_before: Union[str, AnalysisSteps], **kwargs):
        """
        Executes the analysis partially by calling
        :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run` and
        stopping before the specified step.

        .. note::

            Any code inside :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run`
            is still executed. Only the
            :meth:`~quantify.analysis.base_analysis.BaseAnalysis.execute_analysis_steps`
            [which is called by
            :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run` ] is affected.

        Parameters
        ----------
        interrupt_before:
            Stops the analysis before executing the specified step. For convenience
            the analysis step can be specified either as a string or as the member of
            the :attr:`~quantify.analysis.base_analysis.BaseAnalysis.analysis_steps`
            enumerate member.
        **kwargs:
            Any other keyword arguments to be passed to
            :meth:`~quantify.analysis.base_analysis.BaseAnalysis.run`
        """

        # Used by `execute_analysis_steps` to stop
        self._interrupt_before = interrupt_before

        run_params = dict(inspect.signature(self.run).parameters)
        run_params.update(kwargs)

        return self.run(**run_params)

    def get_flow(self) -> tuple:
        """
        Returns a tuple with the ordered methods to be called by run analysis.
        """
        return tuple(getattr(self, elm.value) for elm in self.analysis_steps)

    def extract_data(self):
        """
        If no `dataset_raw` is provided, populates `self.dataset_raw` with data from
        the experiment matching the tuid/label.

        This method should be overwritten if an analysis does not relate to a single
        datafile.
        """
        if self.dataset_raw is not None:
            # pylint: disable=fixme
            # FIXME: to be replaced by a validate_dateset see #187
            if "tuid" not in self.dataset_raw.attrs.keys():
                raise AttributeError('Invalid dataset, missing the "tuid" attribute')

            self.tuid = TUID(self.dataset_raw.attrs["tuid"])
            # an experiment container is required to store output of the analysis.
            # it is possible for this not to exist for a custom dataset as it can
            # come from a source outside of the data directory.
            try:
                locate_experiment_container(self.tuid)
            except FileNotFoundError:
                # if the file did not exist, an experiment folder is created
                # and a copy of the dataset_raw is stored there.
                exp_folder = create_exp_folder(
                    tuid=self.tuid, name=self.dataset_raw.name
                )
                write_dataset(
                    path=os.path.join(exp_folder, DATASET_NAME),
                    dataset=self.dataset_raw,
                )

        if self.dataset_raw is None:
            # if no TUID is specified use the label to search for the latest file with
            # a match.
            if self.tuid is None:
                self.tuid = get_latest_tuid(contains=self.label)
            # Keep a reference to the original dataset.
            self.dataset_raw = load_dataset(tuid=self.tuid)

    def process_data(self):
        """
        This method can be used to process, e.g., reshape, filter etc. the data
        before starting the analysis. By default this method is empty (pass).
        """

    def run_fitting(self):
        """
        Used to fit data to a model. Overwrite this method in a child class if this
        step is required for you analysis.
        """

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
        Saves a copy of the (processed) `.dataset` in the analysis folder of the
        experiment.
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


class Basic1DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def create_figures(self):
        """
        Creates a line plot x vs y for every data variable yi in the dataset.
        """

        # NB we do not use `to_gridded_dataset` because that can potentially drop
        # repeated measurement of the same x0_i setpoint (e.g., AllXY experiment)
        dataset = self.dataset_raw
        # for compatibility with older datasets
        # in case "x0" is not a coordinate we use "dim_0"
        coords = tuple(dataset.coords)
        dims = tuple(dataset.dims)
        plot_against = coords[0] if coords else (dims[0] if dims else None)
        for yi, yvals in dataset.data_vars.items():
            # for compatibility with older datasets, do not plot "x0" vx "x0"
            if yi.startswith("y"):
                fig, ax = plt.subplots()
                fig_id = f"Line plot x0-{yi}"
                # plot this variable against x0
                yvals.plot.line(ax=ax, x=plot_against, marker=".")
                adjust_axeslabels_SI(ax)

                fig.suptitle(
                    f"x0-{yi} {self.dataset_raw.attrs['name']}\n"
                    f"tuid: {self.dataset_raw.attrs['tuid']}"
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
                f"x0x1-{yi} {self.dataset_raw.attrs['name']}\n"
                f"tuid: {self.dataset_raw.attrs['tuid']}"
            )

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
                    gridded_dataset["x1"].attrs["long_name"],
                    gridded_dataset["x1"].attrs["units"],
                ),
                ncol=max(len(gridded_dataset["x1"]) // 8, 1),
            )
            # adjust the labels to be SI aware
            adjust_axeslabels_SI(ax)

            fig.suptitle(
                f"x0x1-{yi} {self.dataset_raw.attrs['name']}\n"
                f"tuid: {self.dataset_raw.attrs['tuid']}"
            )

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


def check_lmfit(fit_res: lmfit.ModelResult):
    """
    Check that `lmfit` was able to successfully return a valid fit, and give
    a warning if not.

    The function looks at `lmfit`'s success parameter, and also checks whether
    the fit was able to obtain valid error bars on the fitted parameters.

    Parameters
    -----------
        fit_res: The :class:`~lmfit.model.ModelResult` object output by `lmfit`

    Returns
    -----------
    str:
        a warning message if there is a problem with the fit
    """
    if fit_res.success is False:
        fit_warning = "fit failed. lmfit was not able to fit the data."
        warnings.warn(fit_warning)
        return "Warning: " + fit_warning

    if fit_res.errorbars is False:
        fit_warning = (
            "lmfit could not find a good fit. Fitted parameters may not be accurate."
        )
        warnings.warn(fit_warning)
        return "Warning: " + fit_warning

    return None


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
    string += "\n".join(  # NB the `+ '.' +` is not redundant
        f"{analysis_steps.__name__ + '.' + name:<{col0_len}}{sep}{class_name}.{value}"
        for name, value in zip(col0, col1)
    )

    return string


def _get_modified_flow(
    flow_functions: tuple,
    step_start: Union[str, AnalysisSteps] = None,
    step_start_inclusive: bool = True,
    step_stop: Union[str, AnalysisSteps] = None,
    step_stop_inclusive: bool = True,
):
    step_names = [meth.__name__ for meth in flow_functions]

    if step_start:
        if not issubclass(type(step_start), str):
            step_start = step_start.value
        start_idx = step_names.index(step_start)
        if not step_start_inclusive:
            start_idx += 1
    else:
        start_idx = 0

    if step_stop:
        if not issubclass(type(step_stop), str):
            step_stop = step_stop.value
        stop_idx = step_names.index(step_stop)
        if step_stop_inclusive:
            stop_idx += 1
    else:
        stop_idx = None

    flow_functions = flow_functions[start_idx:stop_idx]

    return flow_functions
