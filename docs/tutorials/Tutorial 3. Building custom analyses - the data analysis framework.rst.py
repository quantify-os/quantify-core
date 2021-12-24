# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
rst_conf = {"jupyter_execute_options": [":hide-code:"]}
# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=pointless-statement
# pylint: disable=invalid-name
# pylint: disable=duplicate-code
# pylint: disable=abstract-method
# pylint: disable=too-few-public-methods

# %% [raw]
"""
.. _analysis_framework_tutorial:
"""

# %% [raw]
"""
Tutorial 3. Building custom analyses - the data analysis framework
==================================================================
"""

# %% [raw]
"""
.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 3. Building custom analyses - the data analysis framework.py`

    :jupyter-download:script:`Tutorial 3. Building custom analyses - the data analysis framework.py`
"""


# %% [raw]
"""
Quantify provides an analysis framework in the form of a :class:`~quantify_core.analysis.base_analysis.BaseAnalysis` class and several subclasses for simple cases (e.g., :class:`~quantify_core.analysis.base_analysis.BasicAnalysis`, :class:`~quantify_core.analysis.base_analysis.Basic2DAnalysis`, :class:`~quantify_core.analysis.spectroscopy_analysis.ResonatorSpectroscopyAnalysis`). The framework provides a structured, yet flexible, flow of the analysis steps. We encourage all users to adopt the framework by sub-classing the :class:`~quantify_core.analysis.base_analysis.BaseAnalysis`.
"""

# %% [raw]
"""
To give insight into the concepts and ideas behind the analysis framework, we first write analysis scripts to *"manually"* analyze the data as if we had a new type of experiment in our hands.
Next, we encapsulate these steps into reusable functions packing everything together into a simple python class.
"""

# %% [raw]
"""
We conclude by showing how the same class is implemented much more easily by extending the :class:`~quantify_core.analysis.base_analysis.BaseAnalysis` and making use of the quantify framework.
"""

# %%
# %matplotlib inline

import json
import logging
from pathlib import Path
from typing import Tuple

import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from directory_tree import display_tree

import quantify_core.visualization.pyqt_plotmon as pqm
from quantify_core.analysis.cosine_analysis import CosineAnalysis
from quantify_core.analysis.fitting_models import CosineModel, cos_func
from quantify_core.data.handling import (
    get_latest_tuid,
    load_dataset,
    locate_experiment_container,
    set_datadir,
)
from quantify_core.measurement import MeasurementControl
from quantify_core.utilities.examples_support import (
    default_datadir,
    mk_cosine_instrument,
)
from quantify_core.utilities.inspect_utils import display_source_code
from quantify_core.visualization.SI_utilities import set_xlabel, set_ylabel

# %% [raw]
"""
.. include:: /tutorials/set_data_dir_notes.rst.txt
"""

# %%
set_datadir(default_datadir())  # change me!

# %% [raw]
"""
Run an experiment
-----------------
"""

# %% [raw]
"""
We mock an experiment in order to generate a toy dataset to use in this tutorial.

.. admonition:: Create dataset with mock experiment
    :class: dropdown
"""
# %%
rst_conf = {"indent": "    "}
meas_ctrl = MeasurementControl("meas_ctrl")
plotmon = pqm.PlotMonitor_pyqt("plotmon")
meas_ctrl.instr_plotmon(plotmon.name)


# %%
rst_conf = {"indent": "    "}
# We create an instrument to contain all the parameters of our model to ensure
# we have proper data logging.
display_source_code(mk_cosine_instrument)

# %%
rst_conf = {"indent": "    "}
pars = mk_cosine_instrument()

meas_ctrl.settables(pars.t)
meas_ctrl.setpoints(np.linspace(0, 2, 30))
meas_ctrl.gettables(pars.sig)
dataset = meas_ctrl.run("Cosine experiment")
plotmon.main_QtPlot

# %% [raw]
"""
Manual analysis steps
---------------------
"""

# %% [raw]
"""
1. Loading the data

    The :class:`~xarray.Dataset` contains all the information required to perform basic analysis of the experiment.
    We can alternatively load the dataset from disk based on it's :class:`~quantify_core.data.types.TUID`, a timestamp-based unique identifier. If you do not know the tuid of the experiment you can find the latest tuid containing a certain string in the experiment name using :meth:`~quantify_core.data.handling.get_latest_tuid`.
    See the :ref:`data_storage` documentation for more details on the folder structure and files contained in the data directory.
"""

# %%
rst_conf = {"indent": "    "}

tuid = get_latest_tuid(contains="Cosine experiment")
dataset = load_dataset(tuid)
dataset

# %% [raw]
"""
#. Performing a fit
"""

# %% [raw]
"""
    We have a sinusoidal signal in the experiment dataset, the goal is to find the underlying parameters.
    We extract these parameters by performing a fit to a model, a cosine function in this case.
    For fitting we recommend using the lmfit library. See `the lmfit documentation <https://lmfit.github.io/lmfit-py/model.html>`_ on how to fit data to a custom model.
"""

# %%
rst_conf = {"indent": "    "}

# create a fitting model based on a cosine function
fitting_model = lmfit.Model(cos_func)

# specify initial guesses for each parameter
fitting_model.set_param_hint("amplitude", value=0.5, min=0.1, max=2, vary=True)
fitting_model.set_param_hint("frequency", value=0.8, vary=True)
fitting_model.set_param_hint("phase", value=0)
fitting_model.set_param_hint("offset", value=0)
params = fitting_model.make_params()

# here we run the fit
fit_result = fitting_model.fit(dataset.y0.values, x=dataset.x0.values, params=params)

# It is possible to get a quick visualization of our fit using a build-in method of lmfit
_ = fit_result.plot_fit(show_init=True)

# %% [raw]
"""
    The summary of the fit result can be nicely printed in a Jupyter-like notebook:
"""

# %%
rst_conf = {"indent": "    "}

fit_result

# %% [raw]
"""
#. Analyzing the fit result and saving key quantities
"""

# %%
rst_conf = {"indent": "    "}

quantities_of_interest = {
    "amplitude": fit_result.params["amplitude"].value,
    "frequency": fit_result.params["frequency"].value,
}
quantities_of_interest

# %% [raw]
"""
    Now that we have the relevant quantities, we want to store them in the same
    `experiment directory` where the raw dataset is stored.

    First, we determine the experiment directory on the file system.
"""

# %%
rst_conf = {"indent": "    "}

# the experiment folder is retrieved with a convenience function
exp_folder = Path(locate_experiment_container(dataset.tuid))
exp_folder

# %% [raw]
"""
    Then, we save the the quantities of interest to disk in the human-readable JSON format.
"""

# %%
rst_conf = {"indent": "    "}

with open(exp_folder / "quantities_of_interest.json", "w", encoding="utf-8") as file:
    json.dump(quantities_of_interest, file)

# %% [raw]
"""
#. Plotting and saving figures
"""

# %% [raw]
"""
    We would like to save a plot of our data and fit in our lab logbook but the figure above is not fully satisfactory: there are no units and no reference to the original dataset.

    Below we create our own plot for full control over the appearance and we store it on disk in the same `experiment directory`.
    For plotting we use the ubiquitous matplolib and some visualization utilities.
"""

# %%
rst_conf = {"indent": "    "}

# create matplotlib figure
fig, ax = plt.subplots()

# plot data
dataset.y0.plot.line(ax=ax, x="x0", marker="o", label="Data")

# plot fit
x_fit = np.linspace(dataset["x0"][0], dataset["x0"][-1], 1000)
y_fit = cos_func(x=x_fit, **fit_result.best_values)
ax.plot(x_fit, y_fit, label="Fit")
ax.legend()

# set units-aware tick labels
set_xlabel(ax, dataset.x0.long_name, dataset.x0.units)
set_ylabel(ax, dataset.y0.long_name, dataset.y0.units)

# add a reference to the origal dataset in the figure title
fig.suptitle(f"{dataset.attrs['name']}\ntuid: {dataset.attrs['tuid']}")

# Save figure
fig.savefig(exp_folder / "Cosine fit.png", dpi=300, bbox_inches="tight")


# %% [raw]
"""
Reusable fitting model and analysis steps
-----------------------------------------
"""

# %% [raw]
"""
The previous steps achieve our goal, however, the code above is not easily reusable and hard to maintain or debug.
We can do better then this! We can package our code in functions that perform specific tasks.
In addition, we will use the objected-oriented interface of `lmfit` to further structure our code.
We explore the details of the object-oriented approach later in this tutorial.
"""


# %%
class MyCosineModel(lmfit.model.Model):
    """
    `lmfit` model with a guess for a cosine fit.
    """

    def __init__(self, *args, **kwargs):
        """Configures the constraints of the model."""
        # pass in the model's equation
        super().__init__(cos_func, *args, **kwargs)

        # configure constraints that are independent from the data to be fitted

        self.set_param_hint("frequency", min=0, vary=True)  # enforce positive frequency
        self.set_param_hint("amplitude", min=0, vary=True)  # enforce positive amplitude
        self.set_param_hint("offset", vary=True)
        self.set_param_hint(
            "phase", vary=True, min=-np.pi, max=np.pi
        )  # enforce phase range

    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        """Guess parameters based on the data."""

        self.set_param_hint("offset", value=np.average(data))
        self.set_param_hint("amplitude", value=(np.max(data) - np.min(data)) / 2)
        # a simple educated guess based on experiment type
        # a more elaborate but general approach is to use a Fourier transform
        self.set_param_hint("frequency", value=1.2)

        params_ = self.make_params()
        return lmfit.models.update_param_vals(params_, self.prefix, **kws)


# %% [raw]
"""
Most of the code related to the fitting model is now packed in a single object, while the analysis steps are split into functions that take care of specific tasks.
"""


# %%
def extract_data(label: str) -> xr.Dataset:
    """Loads a dataset from its label."""
    tuid_ = get_latest_tuid(contains=label)
    dataset_ = load_dataset(tuid_)
    return dataset_


def run_fitting(dataset_: xr.Dataset) -> lmfit.model.ModelResult:
    """Executes fitting."""
    model = CosineModel()  # create the fitting model
    params_guess = model.guess(data=dataset_.y0.values)
    result = model.fit(
        data=dataset_.y0.values, x=dataset_.x0.values, params=params_guess
    )
    return result


def analyze_fit_results(fit_result_: lmfit.model.ModelResult) -> dict:
    """Analyzes the fit results and saves quantities of interest."""
    quantities = {
        "amplitude": fit_result_.params["amplitude"].value,
        "frequency": fit_result_.params["frequency"].value,
    }
    return quantities


def plot_fit(
    fig_: matplotlib.figure.Figure,
    ax_: matplotlib.axes.Axes,
    dataset_: xr.Dataset,
    fit_result_: lmfit.model.ModelResult,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plots a fit result."""
    dataset_.y0.plot.line(ax=ax_, x="x0", marker="o", label="Data")  # plot data

    x_fit_ = np.linspace(dataset_["x0"][0], dataset_["x0"][-1], 1000)
    y_fit_ = cos_func(x=x_fit_, **fit_result_.best_values)
    ax_.plot(x_fit, y_fit_, label="Fit")  # plot fit
    ax_.legend()

    # set units-aware tick labels
    set_xlabel(ax_, dataset_.x0.long_name, dataset_.x0.units)
    set_ylabel(ax_, dataset_.y0.long_name, dataset_.y0.units)

    # add a reference to the original dataset_ in the figure title
    fig_.suptitle(f"{dataset_.attrs['name']}\ntuid: {dataset_.attrs['tuid']}")


def save_quantities_of_interest(tuid_: str, quantities_of_interest_: dict) -> None:
    """Saves the quantities of interest to disk in JSON format."""
    exp_folder_ = Path(locate_experiment_container(tuid_))
    # Save fit results
    with open(exp_folder_ / "quantities_of_interest.json", "w", encoding="utf-8") as f_:
        json.dump(quantities_of_interest_, f_)


def save_mpl_figure(tuid_: str, fig_: matplotlib.figure.Figure) -> None:
    """Saves a matplotlib figure as PNG."""
    exp_folder_ = Path(locate_experiment_container(tuid_))
    fig_.savefig(exp_folder_ / "Cosine fit.png", dpi=300, bbox_inches="tight")
    plt.close(fig_)


# %% [raw]
"""
Now the execution of the entire analysis becomes much more readable and clean:
"""

# %%
dataset = extract_data(label="Cosine experiment")
fit_result = run_fitting(dataset)
quantities_of_interest = analyze_fit_results(fit_result)
save_quantities_of_interest(dataset.tuid, quantities_of_interest)
fig, ax = plt.subplots()
plot_fit(fig_=fig, ax_=ax, dataset_=dataset, fit_result_=fit_result)
save_mpl_figure(dataset.tuid, fig)

# %% [raw]
"""
We can inspect the `experiment directory` which now contains the analysis results as expected:
"""

# %%
print(display_tree(locate_experiment_container(dataset.tuid), string_rep=True))


# %% [raw]
"""
Creating a simple analysis class
--------------------------------
"""

# %% [raw]
"""
Even though we have improved code structure greatly, in order to execute the same analysis against some other dataset we would have to copy-paste a significant portion of code (the analysis steps).
"""

# %% [raw]
"""
We tackle this by taking advantage of the Object Oriented Programming (OOP) in python.
We will create a python class that serves as a structured container for data (attributes) and the methods (functions) that act on the information.
"""

# %% [raw]
"""
Some of the advantages of OOP are:
"""

# %% [raw]
"""
- the same class can be instantiated multiple times to act on different data while reusing the same methods;
- all the methods have access to all the data (attributes) associated with a particular instance of the class;
- subclasses can inherit from other classes and extend their functionalities.
"""

# %% [raw]
"""
Let's now observe how such a class could look like.
"""

# %% [raw]
"""
.. warning::

    This analysis class is intended for educational purposes only.
    It is not intended to be used as a template!
    See the end of the tutorial for the recommended usage of the analysis framework.
"""


# %%
class MyCosineAnalysis:
    """Analysis as a class."""

    def __init__(self, label: str):
        """This is a special method that python calls when an instance of this class is
        created."""

        self.label = label

        # objects to be filled up later when running the analysis
        self.tuid = None
        self.dataset = None
        self.fit_results = {}
        self.quantities_of_interest = {}
        self.figs_mpl = {}
        self.axs_mpl = {}

    # with just slight modification our functions become methods
    # with the advantage that we have access to all the necessary information from self
    def run(self):
        """Execute the analysis steps."""
        self.extract_data()
        self.run_fitting()
        self.analyze_fit_results()
        self.create_figures()
        self.save_quantities_of_interest()
        self.save_figures()

    def extract_data(self):
        """Load data from disk."""
        self.tuid = get_latest_tuid(contains=self.label)
        self.dataset = load_dataset(tuid)

    def run_fitting(self):
        """Fits the model to the data."""
        model = MyCosineModel()
        guess = model.guess(self.dataset.y0.values)
        result = model.fit(
            self.dataset.y0.values, x=self.dataset.x0.values, params=guess
        )
        self.fit_results.update({"cosine": result})

    def analyze_fit_results(self):
        """Analyzes the fit results and saves quantities of interest."""
        self.quantities_of_interest.update(
            {
                "amplitude": self.fit_results["cosine"].params["amplitude"].value,
                "frequency": self.fit_results["cosine"].params["frequency"].value,
            }
        )

    def save_quantities_of_interest(self):
        """Save quantities of interest to disk."""
        exp_folder_ = Path(locate_experiment_container(self.tuid))
        with open(
            exp_folder_ / "quantities_of_interest.json", "w", encoding="utf-8"
        ) as file_:
            json.dump(self.quantities_of_interest, file_)

    def plot_fit(self, fig_: matplotlib.figure.Figure, ax_: matplotlib.axes.Axes):
        """Plot the fit result."""

        self.dataset.y0.plot.line(ax=ax_, x="x0", marker="o", label="Data")  # plot data

        x_fit_ = np.linspace(self.dataset["x0"][0], self.dataset["x0"][-1], 1000)
        y_fit_ = cos_func(x=x_fit_, **self.fit_results["cosine"].best_values)
        ax_.plot(x_fit_, y_fit_, label="Fit")  # plot fit
        ax_.legend()

        # set units-aware tick labels
        set_xlabel(ax_, self.dataset.x0.long_name, self.dataset.x0.attrs["units"])
        set_ylabel(ax_, self.dataset.y0.long_name, self.dataset.y0.attrs["units"])

        # add a reference to the original dataset in the figure title
        fig_.suptitle(f"{dataset.attrs['name']}\ntuid: {dataset.attrs['tuid']}")

    def create_figures(self):
        """Create figures."""
        fig_, ax_ = plt.subplots()
        self.plot_fit(fig_, ax_)

        fig_id = "cos-data-and-fit"
        self.figs_mpl.update({fig_id: fig_})
        # keep a reference to `ax` as well
        # it can be accessed later to apply modifications (e.g., in a notebook)
        self.axs_mpl.update({fig_id: ax_})

    def save_figures(self):
        """Save figures to disk."""
        exp_folder_ = Path(locate_experiment_container(self.tuid))
        for fig_name, fig_ in self.figs_mpl.items():
            fig_.savefig(exp_folder_ / f"{fig_name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig_)


# %% [raw]
"""
Running the analysis is now as simple as:
"""

# %%
a_obj = MyCosineAnalysis(label="Cosine experiment")
a_obj.run()
a_obj.figs_mpl["cos-data-and-fit"]

# %% [raw]
"""
The first line will instantiate the class by calling the :code:`.__init__()` method.
"""

# %% [raw]
"""
As expected this will save similar files into the `experiment directory`:
"""

# %%
print(display_tree(locate_experiment_container(a_obj.dataset.tuid), string_rep=True))

# %% [raw]
"""
Extending the BaseAnalysis
--------------------------
"""

# %% [raw]
"""
While the above stand-alone class provides the gist of an analysis, we can do even better by defining a structured framework that all analysis need to adhere to and factoring out the pieces of code that are common to most analyses.
Beside that, the overall functionality can be improved.
"""

# %% [raw]
"""
Here is where the :class:`~quantify_core.analysis.base_analysis.BaseAnalysis` enters the scene.
It allows us to focus only on the particular aspect of our custom analysis by implementing only the relevant methods. Take a look at how the above class is implemented where we are making use of the analysis framework. For completeness, a fully documented :class:`~quantify_core.analysis.fitting_models.CosineModel` that can serve as a template is shown as well.
"""

# %%
display_source_code(CosineModel)
display_source_code(CosineAnalysis)

# %% [raw]
"""
Now we can simply execute it against our latest experiment as follows:
"""


# %%
a_obj = CosineAnalysis(label="Cosine experiment").run()
a_obj.display_figs_mpl()


# %% [raw]
"""
Inspecting the `experiment directory` yields:
"""

# %%
print(display_tree(locate_experiment_container(a_obj.dataset.tuid), string_rep=True))


# %% [raw]
"""
As you can conclude from the :class:`!CosineAnalysis` code, we did not implement quite a few methods in there.
These are provided by the :class:`~quantify_core.analysis.base_analysis.BaseAnalysis`.
To gain some insight on what exactly is being executed we can enable the logging module and use the internal logger of the analysis instance:
"""

# %%
rst_conf = {"jupyter_execute_options": [":stderr:"]}

# activate logging and set global level to show warnings only
logging.basicConfig(level=logging.WARNING)

# set analysis logger level to info (the logger is inherited from BaseAnalysis)
a_obj.logger.setLevel(level=logging.INFO)
_ = a_obj.run()
