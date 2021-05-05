#!/usr/bin/env python
# coding: utf-8
# %%

import lmfit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %matplotlib inline


# %%


from quantify.measurement import MeasurementControl
from quantify.measurement.control import Settable, Gettable
import quantify.visualization.pyqt_plotmon as pqm
from quantify.visualization.instrument_monitor import InstrumentMonitor
from qcodes import ManualParameter, Parameter, validators, Instrument


# %%


# We recommend to always set the directory at the start of the python kernel
# and stick to a single common data directory for all
# notebooks/experiments within your measurement setup/PC


# %%


# This sets a default data directory for tutorial purposes. Change it to your desired data directory.
from pathlib import Path
from os.path import join
from quantify.data.handling import get_datadir, set_datadir

set_datadir(join(Path.home(), "quantify-data"))  # change me!
print(f"Data will be saved in:\n{get_datadir()}")


# %% [markdown]
# ## Instantiate instruments

# %%


MC = MeasurementControl("MC")
# Create the live plotting intrument which handles the graphical interface
# Two windows will be created, the main will feature 1D plots and any 2D plots will go to the secondary
plotmon = pqm.PlotMonitor_pyqt("plotmon")
# Connect the live plotting monitor to the measurement control
MC.instr_plotmon(plotmon.name)

# %% [markdown]
# ### Mock experiment

# %%
from time import sleep
from quantify.analysis.fitting_models import cos_func

# We create an instrument to contain all the parameters of our model to ensure we have proper data logging.
from qcodes.instrument import Instrument

pars = Instrument("ParameterHolder")

# ManualParameter's is a handy class that preserves the QCoDeS' Parameter
# structure without necessarily having a connection to the physical world
pars.add_parameter(
    "amp", initial_value=1, unit="V", label="Amplitude", parameter_class=ManualParameter
)
pars.add_parameter(
    "freq",
    initial_value=0.5,
    unit="Hz",
    label="Frequency",
    parameter_class=ManualParameter,
)
pars.add_parameter(
    "t", initial_value=1, unit="s", label="Time", parameter_class=ManualParameter
)
pars.add_parameter(
    "phi", initial_value=0, unit="Rad", label="Phase", parameter_class=ManualParameter
)
pars.add_parameter(
    "noise_level",
    initial_value=0.05,
    unit="V",
    label="Noise level",
    parameter_class=ManualParameter,
)
pars.add_parameter(
    "acq_delay", initial_value=0.05, unit="s", parameter_class=ManualParameter
)


def cosine_model():
    sleep(pars.acq_delay())  # simulates the acquisition delay of an instrument
    return (
        cos_func(pars.t(), pars.amp(), pars.freq(), phase=pars.phi(), offset=0)
        + np.random.randn() * pars.noise_level()
    )


# We wrap our function in a Parameter to be able to associate metadata to it, e.g. units
pars.add_parameter(name="sig", label="Signal level", unit="V", get_cmd=cosine_model)

# %%
MC.settables(pars.t)
MC.setpoints(np.linspace(0, 2, 30))
MC.gettables(pars.sig)
dataset = MC.run("Cosine experiment")
plotmon.main_QtPlot

# %% [markdown]
# # Analyse

# %%


from quantify.data.handling import load_dataset, get_latest_tuid

# here we look for the latest datafile in the datadirectory containing "Cosine experiment"
# note that this is not he last dataset but one dataset earlier
tuid = get_latest_tuid(contains="Cosine experiment")
print("tuid: {}".format(tuid))
dataset = load_dataset(tuid)
dataset


# %%

fitting_model = lmfit.Model(
    cos_func
)  # create a fitting model based on a cosine function

# and specify initial guesses for each parameter
fitting_model.set_param_hint("amplitude", value=0.5, min=0.1, max=2, vary=True)
fitting_model.set_param_hint("frequency", value=0.8, vary=True)
fitting_model.set_param_hint("phase", value=0)
fitting_model.set_param_hint("offset", value=0)
params = fitting_model.make_params()
# and here we perform the fit.
fit_result = fitting_model.fit(dataset.y0.values, x=dataset.x0.values, params=params)

# It is possible to get a quick visualization of our fit using a build-in method of lmfit
_ = fit_result.plot_fit(show_init=True)


# %%
fit_result


# %%
quantities_of_interst = {
    "amplitude": fit_result.params["amplitude"].value,
    "frequency": fit_result.params["frequency"].value,
}
quantities_of_interst


# %% [markdown]
# ### do a bit better, modulairty and reusbale

# %% [markdown]
# Object oriented interface of lmfit

# %%
class CosineModel(lmfit.model.Model):
    """
    lmfit model with a guess for a cosine fit.
    """

    def __init__(self, *args, **kwargs):
        # pass in the model's equation equation
        super().__init__(cos_func, *args, **kwargs)

        # configure constraints that are independent from the data to be fitted

        self.set_param_hint("frequency", min=0, vary=True)  # enforce positive frequency
        self.set_param_hint("amplitude", min=0, vary=True)  # enforce positive amplitude
        self.set_param_hint("offset", vary=True)
        self.set_param_hint(
            "phase", vary=True, min=-np.pi, max=np.pi
        )  # enforce phase range

    def guess(self, data, **kws) -> lmfit.parameter.Parameters:

        # guess parameters based on the data

        self.set_param_hint("offset", value=np.average(data))
        self.set_param_hint("amplitude", value=(np.max(data) - np.min(data)) / 2)
        # a simple educated guess based on experiment type
        # a more elaborate but general approach is to use a Fourier transform
        self.set_param_hint("frequency", value=1.2)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)


# %%
def extract_data(label: str) -> xr.Dataset:
    tuid = get_latest_tuid(contains=label)
    dataset = load_dataset(tuid)
    return dataset


def run_fitting(dataset: xr.Dataset) -> lmfit.model.ModelResult:
    model = CosineModel()  # create the fitting model
    params_guess = model.guess(data=dataset.y0.values)
    result = model.fit(data=dataset.y0.values, x=dataset.x0.values, params=params_guess)
    return result


def analyze_fit_results(fit_result: lmfit.model.ModelResult) -> dict:
    # analyze results
    # ...

    # extract relevant quantities
    quantities = {
        "amplitude": fit_result.params["amplitude"].value,
        "frequency": fit_result.params["frequency"].value,
    }
    return quantities


# %%
dataset = extract_data(label="Cosine experiment")
fit_result = run_fitting(dataset=dataset)
quantities_of_interst = analyze_fit_results(fit_result=fit_result)
# as expected we obtain the same results
_ = fit_result.plot_fit(show_init=True)
print(f"quantities_of_interst: {quantities_of_interst}")

# %% [markdown]
# We would like to save this plot for in our lab logbook but the figure is not fully satisfactory. There are no units and no refermce to the original data...

# %%
# We include some visualization utilities in quantify
from quantify.visualization.SI_utilities import set_xlabel, set_ylabel

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
_ = fig.suptitle(f"{dataset.attrs['name']}\ntuid: {dataset.attrs['tuid']}")


# %% [markdown]
# Again, this can be packed into a reusable function

# %%
import matplotlib
from typing import Tuple


def plot_fit(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    dataset: xr.Dataset,
    fit_result: lmfit.model.ModelResult,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

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


fig, ax = plt.subplots()
# same output as expected:
plot_fit(fig=fig, ax=ax, dataset=dataset, fit_result=fit_result)

# %%
from quantify.data.handling import locate_experiment_container
from pathlib import Path

# Here we are using this function as a convenient way of retrieving the experiment
# folder without using an absolute path
exp_folder = Path(locate_experiment_container(dataset.tuid))
exp_folder


# %%
import json

# Save fit results
with open(exp_folder / "quantities_of_interst.json", "w") as file:
    json.dump(quantities_of_interst, file)

# Save figure
fig.savefig(exp_folder / "Cosine fit.png", dpi=300, bbox_inches="tight")


# %% [markdown]
# Again... tedious...

# %%
def save_quantities_of_interst(tuid: str, quantities_of_interst: dict) -> None:
    exp_folder = Path(locate_experiment_container(tuid))
    # Save fit results
    with open(exp_folder / "quantities_of_interst.json", "w") as file:
        json.dump(quantities_of_interst, file)


def save_mpl_figure(tuid: str, fig: matplotlib.figure.Figure) -> None:
    exp_folder = Path(locate_experiment_container(tuid))
    fig.savefig(exp_folder / "Cosine fit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# %%
save_quantities_of_interst(dataset.tuid, quantities_of_interst)
save_mpl_figure(dataset.tuid, fig)

# %% [markdown]
# ### Runnning the full analysis using the functions
#
# We can already a recognize a certain common flow

# %%
# load data
dataset = extract_data(label="Cosine experiment")

# typically the function would be imported form a module
# from my_module import fit, plot_fit, save_quantities_of_interst, ...

# run fitting
fit_result = run_fitting(dataset)

# extract quantities of interest
quantities_of_interst = analyze_fit_results(fit_result)

# plot results and fit
fig, ax = plt.subplots()
plot_fit(fig, ax, dataset, fit_result)

# save analysis results and figures to disk
save_quantities_of_interst(dataset.tuid, quantities_of_interst)
save_mpl_figure(dataset.tuid, fig)

# %%
from directory_tree import display_tree

print(display_tree(Path(locate_experiment_container(dataset.tuid)), string_rep=True))

# %% [markdown]
# ### Class approach
#
# Best practices:
# - **do not add** any attributes to the analysis object after its initial creation and configuration (e.g., `self.my_new_attr = [1, 2, 3]` in the `fit` method is a bad practice, instead create the attribute before running the analysis as `self.my_new_attr = None`). This avoids common pitfalls of the objected-oriented-programming that can be hard to debbug later
# - do no modify the raw dataset
# - split your code in multiple methods (functions) that execute specific tasks

# %%
from collections import OrderedDict


class MyCosineAnalysis:
    def __init__(self, label: str):
        """This is a special method that python calls when an instance of this class is created."""

        self.label = label
        self.tuid = None
        self.dataset_raw = None

        # objects to be filled up later when running the analysis
        self.fit_results = OrderedDict()
        self.quantities_of_interst = OrderedDict()
        self.figs_mpl = OrderedDict()
        self.axs_mpl = OrderedDict()

    # with just slight modification our functions become methods
    # with the advantage that we have access to all the necessary information from `self.`
    def run(self):
        """Execute the analysis steps"""
        self.extract_data()
        self.run_fitting()
        self.analyze_fit_results()
        self.create_figures()
        self.save_quantities_of_interst()
        self.save_figures()

        return self

    def extract_data(self):
        self.tuid = get_latest_tuid(contains=self.label)
        self.dataset_raw = load_dataset(tuid)

    def run_fitting(self):
        """Fits a CosineModel to the data."""
        model = CosineModel()
        guess = model.guess(self.dataset_raw.y0.values)
        result = model.fit(
            self.dataset_raw.y0.values, x=self.dataset_raw.x0.values, params=guess
        )
        self.fit_results.update({"cosine": result})

    def analyze_fit_results(self):
        self.quantities_of_interst.update(
            {
                "amplitude": self.fit_results["cosine"].params["amplitude"].value,
                "frequency": self.fit_results["cosine"].params["frequency"].value,
            }
        )

    def save_quantities_of_interst(self):
        exp_folder = Path(locate_experiment_container(self.tuid))
        with open(exp_folder / "quantities_of_interst.json", "w") as file:
            json.dump(self.quantities_of_interst, file)

    def plot_fit(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes):
        # plot data
        self.dataset_raw.y0.plot.line(ax=ax, x="x0", marker="o", label="Data")

        # plot fit
        x_fit = np.linspace(self.dataset_raw["x0"][0], self.dataset_raw["x0"][-1], 1000)
        y_fit = cos_func(x=x_fit, **fit_result.best_values)
        ax.plot(x_fit, y_fit, label="Fit")
        ax.legend()

        # set units-aware tick labels
        set_xlabel(
            ax, self.dataset_raw.x0.long_name, self.dataset_raw.x0.attrs["units"]
        )
        set_ylabel(
            ax, self.dataset_raw.y0.long_name, self.dataset_raw.y0.attrs["units"]
        )

        # add a reference to the origal dataset in the figure title
        fig.suptitle(f"{dataset.attrs['name']}\ntuid: {dataset.attrs['tuid']}")

    def create_figures(self):
        fig, ax = plt.subplots()
        self.plot_fit(fig, ax)

        fig_id = "cos-data-and-fit"
        self.figs_mpl.update({fig_id: fig})
        # keep a reference to `ax` as well
        # it can be accessed later to apply modifications (e.g., in a notebook)
        self.axs_mpl.update({fig_id: ax})

    def save_figures(self):
        exp_folder = Path(locate_experiment_container(self.tuid))
        for fig_name, fig in self.figs_mpl.items():
            fig.savefig(exp_folder / f"{fig_name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


# %%
a_obj = MyCosineAnalysis(
    label="Cosine experiment"
)  # calls MyCosineAnalysis.__init__(...)
a_obj.run()

# %%
print(
    display_tree(
        Path(locate_experiment_container(a_obj.dataset_raw.tuid)), string_rep=True
    )
)

# %% [markdown]
# ## Using the BaseAnalysis class

# %% [raw]
# import quantify.analysis.base_analysis as ba
# from quantify.visualization import mpl_plotting as qpl
# from quantify.visualization.SI_utilities import format_value_string, adjust_axeslabels_SI
#
# # extend the BaseAnalysis
# class CosineAnalysis(ba.BaseAnalysis):
#
#     def process_data(self):
#         """
#         In some cases, you might need to process the data, e.g., reshape, filter etc.,
#         before starting the analysis. This is the method where it should be done.
#
#         For examples see :meth:`~quantify.analysis.spectroscopy_analysis.ResonatorSpectroscopyAnalysis.process_dataset`
#         """
#
#     def run_fitting(self):
#         # create a fitting model based on a cosine function
#         model = CosineModel()
#         guess = model.guess(self.dataset_raw.y0.values)
#         result = model.fit(
#             self.dataset_raw.y0.values, x=self.dataset_raw.x0.values, params=guess
#         )
#         self.fit_results.update({"cosine": result})
#
#     def create_figures(self):
#         fig, ax = plt.subplots()
#         fig_id = "cos_fit"
#         self.figs_mpl.update({fig_id: fig})
#         self.axs_mpl.update({fig_id: ax})
#
#         self.dataset_raw.y0.plot(ax=ax, x="x0", marker="o", linestyle="")
#         qpl.plot_fit(ax, self.fit_results["cosine"])
#         qpl.plot_textbox(ax, ba.wrap_text(self.quantities_of_interest["fit_msg"]))
#
#         adjust_axeslabels_SI(ax)
#         qpl.set_suptitle_from_dataset(fig, self.dataset_raw, f"x0-y0")
#         ax.legend()
#
#     def analyze_fit_results(self):
#         fit_res = self.fit_results["cosine"]
#         fit_warning = ba.check_lmfit(fit_res)
#
#         # If there is a problem with the fit, display an error message in the text box.
#         # Otherwise, display the parameters as normal.
#         if fit_warning is None:
#             self.quantities_of_interest["fit_success"] = True
#             unit = self.dataset_raw.y0.units
#             text_msg = "Summary\n"
#             text_msg += format_value_string(
#                 r"$f$", fit_res.params["frequency"], end_char="\n", unit="Hz"
#             )
#             text_msg += format_value_string(
#                 r"$A$", fit_res.params["amplitude"], unit=unit
#             )
#         else:
#             text_msg = fit_warning
#             self.quantities_of_interest["fit_success"] = False
#
#         # save values and fit uncertainty
#         for parname in ["frequency", "amplitude"]:
#             self.quantities_of_interest[parname] = ba.lmfit_par_to_ufloat(
#                 fit_res.params[parname]
#             )
#         self.quantities_of_interest["fit_msg"] = text_msg

# %%
MC.settables(pars.t)
MC.setpoints(np.linspace(0, 2, 50))
MC.gettables(pars.sig)
dataset = MC.run("Cosine experiment")

# %%
from quantify.analysis.cosine_analysis import CosineAnalysis

a_obj = CosineAnalysis(label="Cosine experiment").run()

a_obj.display_figs_mpl()

# %%
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%
logger.info("bla")

# %%
a_obj.logger.setLevel("INFO")
a_obj.run()

# %%
a_obj.logger.info("bla")

# %%
print(
    display_tree(
        Path(locate_experiment_container(a_obj.dataset_raw.tuid)), string_rep=True
    )
)

# %%
import logging

logging.info("")  # call to info too early

# %%
