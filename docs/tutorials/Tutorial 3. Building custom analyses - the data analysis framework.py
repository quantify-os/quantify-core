#!/usr/bin/env python
# coding: utf-8
# %%

import lmfit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %matplotlib inline # automatically display matplolib figure created in each cell


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
dataset

# %% [markdown]
# # Analyse

# %%


from quantify.data.handling import load_dataset, get_latest_tuid

# here we look for the latest datafile in the datadirectory containing "Cosine experiment"
# note that this is not he last dataset but one dataset earlier
tuid = get_latest_tuid(contains="Cosine experiment")
print("tuid: {}".format(tuid))
dataset = load_dataset(tuid)
dataset.y0.plot()  # quick preview
dataset


# %% [markdown]
# ## TO DO
#
# - [ ] use the object orinted interface for lmfit from the beggining

# %%

fitting_model = lmfit.Model(
    cos_func
)  # create a fitting model based on a cosine function

# and specify initial guesses for each parameter
fitting_model.set_param_hint("amplitude", value=0.5, min=0.1, max=2, vary=True)
fitting_model.set_param_hint("frequency", value=0.8, vary=True)
fitting_model.set_param_hint("phase", value=0, vary=False)
fitting_model.set_param_hint("offset", value=0, vary=False)
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

# %%
def guess_cosine_paramaters(model: lmfit.Model) -> lmfit.parameter.Parameters:
    # specify initial guesses for each parameter
    model.set_param_hint("amplitude", value=0.5, min=0.1, max=2, vary=True)
    model.set_param_hint("frequency", value=0.8, vary=True)
    model.set_param_hint("phase", value=0, vary=False)
    model.set_param_hint("offset", value=0, vary=False)
    params = model.make_params()
    return params


def fit_cosine(dataset: xr.Dataset) -> lmfit.model.ModelResult:
    model = lmfit.Model(cos_func)  # create a fitting model based on a cosine function
    params = guess_cosine_paramaters(model)
    result = model.fit(dataset.y0.values, x=dataset.x0.values, params=params)
    return result


def extract_quantities_of_interst(cosine_fit_result: lmfit.model.ModelResult) -> dict:
    quantities = {
        "amplitude": cosine_fit_result.params["amplitude"].value,
        "frequency": cosine_fit_result.params["frequency"].value,
    }
    return quantities


fit_result = fit_cosine(dataset)
quantities_of_interst = extract_quantities_of_interst(fit_result)
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
y_fit = cos_func(x=x_fit, **fit_res.best_values)
ax.plot(x_fit, y_fit, label="Fit")
ax.legend()

# set units-aware tick labels
set_xlabel(ax, dataset.x0.attrs["long_name"], dataset.x0.attrs["units"])
set_ylabel(ax, dataset.y0.attrs["long_name"], dataset.y0.attrs["units"])

# add a reference to the origal dataset in the figure title
_ = fig.suptitle(f"{dataset.attrs['name']}\n" f"{dataset.attrs['tuid']}")


# %% [markdown]
# Again this can be packed into a reusable function

# %%
import matplotlib
from typing import Tuple


def plot_cosine_fit(
    dataset: xr.Dataset,
    fit_result: lmfit.model.ModelResult,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axes.Axes = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

    if fig is None or ax is None:
        # create matplotlib figure
        fig, ax = plt.subplots()

    # plot data
    dataset.y0.plot.line(ax=ax, x="x0", marker="o", label="Data")

    # plot fit
    x_fit = np.linspace(dataset["x0"][0], dataset["x0"][-1], 1000)
    y_fit = cos_func(x=x_fit, **fit_res.best_values)
    ax.plot(x_fit, y_fit, label="Fit")
    ax.legend()

    # set units-aware tick labels
    set_xlabel(ax, dataset.x0.attrs["long_name"], dataset.x0.attrs["units"])
    set_ylabel(ax, dataset.y0.attrs["long_name"], dataset.y0.attrs["units"])

    # add a reference to the origal dataset in the figure title
    fig.suptitle(f"{dataset.attrs['name']}\n" f"{dataset.attrs['tuid']}")

    return fig, ax


fig, ax = plot_cosine_fit(dataset, fit_result)

# %%

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
with open(exp_folder / "quantities_of_interst_cosine.json", "w") as file:
    json.dump(quantities_of_interst, file)

# Save figure
fig.savefig(exp_folder / "Cosine fit.png", dpi=300, bbox_inches="tight")


# %% [markdown]
# Again... tedious...

# %%
def save_quantities_of_interst_cosine(tuid: str, quantities_of_interst: dict) -> None:
    exp_folder = Path(locate_experiment_container(tuid))
    # Save fit results
    with open(exp_folder / "quantities_of_interst_cosine.json", "w") as file:
        json.dump(quantities_of_interst, file)


save_quantities_of_interst_cosine(dataset.tuid, quantities_of_interst)


# %%
def save_mpl_figure_cosine(tuid: str, fig: matplotlib.figure.Figure) -> None:
    exp_folder = Path(locate_experiment_container(tuid))
    fig.savefig(exp_folder / "Cosine fit.png", dpi=300, bbox_inches="tight")


save_mpl_figure_cosine(dataset.tuid, fig)

# %% [markdown]
# ### Runnning the full analysis using the functions

# %%
# load data
dataset = load_dataset(get_latest_tuid(contains="Cosine experiment"))

# typically the function would be imported form a module
# from my_module import fit_cosine, plot_cosine_fit, save_quantities_of_interst_cosine, ...

# run fitting
fit_result = fit_cosine(dataset)

# extract quantities of interest
quantities_of_interst = extract_quantities_of_interst(fit_result)

# plot results and fit
fig, ax = plot_cosine_fit(dataset, fit_result)

# save analysis results and figures to disk
save_quantities_of_interst_cosine(dataset.tuid, quantities_of_interst)
save_mpl_figure_cosine(dataset.tuid, fig)

# %%
from directory_tree import display_tree

print(
    display_tree(Path(locate_experiment_container(dataset.tuid)), string_rep=True),
    end="",
)


# %% [markdown]
# ### Class approach
#
# Best practices:
# - **do not add** any attributes to the analysis object after its initial creation and configuration (e.g., `self.my_new_attr = [1, 2, 3]` in the `fit_cosine` method is a bad practice, instead create the attribute before running the analysis as `self.my_new_attr = None`). This avoids common pitfalls of the objected-oriented-programming that can be hard to debbug later
# - do no modify the raw dataset
# - split your code in multiple methods (functions) that execute specific tasks

# %%
class MyCosineAnalysis:
    def __init__(self, label: str):
        """This is a special method that python calls when a an instance of this class is created."""
        self.raw_dataset = load_dataset(get_latest_tuid(contains=label))
        self.tuid = self.raw_dataset.tuid

        # objects to be filled up later when running the analysis
        self.fit_result = None
        self.fit_model = None
        self.fit_params = None
        self.quantities_of_interst = None
        self.fig = None
        self.ax = None

    # with just slight modification our functions become methods
    # with the advantage that we have access to all the necessary information from `self.`

    def fit_cosine(self):
        self.fitting_model = lmfit.Model(
            cos_func
        )  # create a fitting model based on a cosine function
        self.params = guess_cosine_paramaters(self.fitting_model)
        self.result = model.fit(
            dataset.y0.values, x=dataset.x0.values, params=self.params
        )

    def extract_quantities_of_interst(
        cosine_fit_result: lmfit.model.ModelResult,
    ) -> dict:
        self.quantities_of_interst = {
            "amplitude": cosine_fit_result.params["amplitude"].value,
            "frequency": cosine_fit_result.params["frequency"].value,
        }


# %%
