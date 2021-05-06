.. _analysis_framework_tutorial:

Tutorial 3. Building custom analyses - the data analysis framework
==================================================================

.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 3. Building custom analyses - the data analysis framework`

    :jupyter-download:script:`Tutorial 3. Building custom analyses - the data analysis framework`


Quantify provides an analysis framework in the form of a :class:`~quantify.analysis.base_analysis.BaseAnalysis` class and several subclasses for simple cases (e.g., :class:`~quantify.analysis.base_analysis.Basic1DAnalysis`, :class:`~quantify.analysis.base_analysis.Basic2DAnalysis`, :class:`~quantify.analysis.spectroscopy_analysis.ResonatorSpectroscopyAnalysis`). The framework provides a structured, yet flexible, flow of the analysis steps. We encourage all users to adopt the framework by sub-classing the :class:`~quantify.analysis.base_analysis.BaseAnalysis`.

To give insight into the concepts and ideas behind the analysis framework, we first write analysis scripts to *"manually"* analyze the data as if we had a new type of experiment in our hands.
Next, we encapsulate these steps into reusable functions packing everything together into a simple python class.

We conclude by showing how the same class is implemented much more easily by extending the :class:`~quantify.analysis.base_analysis.BaseAnalysis` and making use of the quantify framework.

---

.. jupyter-execute::

    %matplotlib inline
    import lmfit
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    from quantify.measurement import MeasurementControl
    from quantify.measurement.control import Settable, Gettable
    import quantify.visualization.pyqt_plotmon as pqm
    from quantify.visualization.instrument_monitor import InstrumentMonitor
    from qcodes import ManualParameter, Parameter, validators, Instrument

.. include:: set_data_dir.rst.txt

Run an experiment
-----------------

We mock an experiment in order to generate a toy dataset to use in this tutorial.

.. admonition:: Create dataset with mock experiment
    :class: dropdown

    .. jupyter-execute::

        MC = MeasurementControl("MC")
        plotmon = pqm.PlotMonitor_pyqt("plotmon")
        MC.instr_plotmon(plotmon.name)

    .. include:: cosine_instrument.rst.txt

    .. jupyter-execute::

        MC.settables(pars.t)
        MC.setpoints(np.linspace(0, 2, 30))
        MC.gettables(pars.sig)
        dataset = MC.run("Cosine experiment")
        plotmon.main_QtPlot

Manual analysis steps
---------------------

1. Loading the data

    The :class:`~xarray.Dataset` contains all the information required to perform basic analysis of the experiment.
    We can alternatively load the dataset from disk based on it's :class:`~quantify.data.types.TUID`, a timestamp-based unique identifier. If you do not know the tuid of the experiment you can find the latest tuid containing a certain string in the experiment name using :meth:`~quantify.data.handling.get_latest_tuid`.
    See the :ref:`data_storage` documentation for more details on the folder structure and files contained in the data directory.

    .. jupyter-execute::

        from quantify.data.handling import load_dataset, get_latest_tuid

        tuid = get_latest_tuid(contains="Cosine experiment")
        dataset = load_dataset(tuid)
        dataset

#. Performing a fit

    We have a sinusoidal signal in the experiment dataset, the goal is to find the underlying parameters.
    We extract these parameters by performing a fit to a model, a cosine function in this case.
    For fitting we recommend using the lmfit library. See `the lmfit documentation <https://lmfit.github.io/lmfit-py/model.html>`_ on how to fit data to a custom model.

    .. jupyter-execute::

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

    The summary of the fit result can be nicely printed in a Jupyter-like notebook:

    .. jupyter-execute::

        fit_result

#. Analyzing the fit result and saving key quantities

    .. jupyter-execute::

        quantities_of_interest = {
            "amplitude": fit_result.params["amplitude"].value,
            "frequency": fit_result.params["frequency"].value,
        }
        quantities_of_interest

    Now that we have the relevant quantities, we want to store them in the same
    `experiment directory` where the raw dataset is stored.

    First, we determine the experiment directory on the file system.

    .. jupyter-execute::

        import json
        from quantify.data.handling import locate_experiment_container
        from pathlib import Path

        # the experiment folder is retrieved with a convenience function
        exp_folder = Path(locate_experiment_container(dataset.tuid))
        exp_folder

    Then, we save the the quantities of interest to disk in the human-readable JSON format.

    .. jupyter-execute::

        with open(exp_folder / "quantities_of_interest.json", "w") as file:
            json.dump(quantities_of_interest, file)

#. Plotting and saving figures

    We would like to save a plot of our data and fit in our lab logbook but the figure above is not fully satisfactory: there are no units and no reference to the original dataset.

    Below we create our own plot for full control over the appearance and we store it on disk in the same `experiment directory`.
    For plotting we use the ubiquitous matplolib and some visualization utilities.

    .. jupyter-execute::

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
        fig.suptitle(f"{dataset.attrs['name']}\ntuid: {dataset.attrs['tuid']}")

        # Save figure
        fig.savefig(exp_folder / "Cosine fit.png", dpi=300, bbox_inches="tight")

Reusable fitting model and analysis steps
-----------------------------------------

The previous steps achieve our goal, however, the code above is not easily reusable and hard to maintain or debug.
We can do better then this! We can package our code in functions that perform specific tasks.
In addition, we will use the objected-oriented interface of `lmfit` to farther structure our code.
We explore the details of the object-oriented approach later in this tutorial.

.. jupyter-execute::

    class CosineModel(lmfit.model.Model):
        """
        lmfit model with a guess for a cosine fit.
        """

        def __init__(self, *args, **kwargs):
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

            # guess parameters based on the data

            self.set_param_hint("offset", value=np.average(data))
            self.set_param_hint("amplitude", value=(np.max(data) - np.min(data)) / 2)
            # a simple educated guess based on experiment type
            # a more elaborate but general approach is to use a Fourier transform
            self.set_param_hint("frequency", value=1.2)

            params = self.make_params()
            return lmfit.models.update_param_vals(params, self.prefix, **kws)

Most of the code related to the fitting model is now packed in a single object, while the analysis steps are split into functions that take care of specific tasks.

.. jupyter-execute::

    import matplotlib
    from typing import Tuple

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
        quantities = {
            "amplitude": fit_result.params["amplitude"].value,
            "frequency": fit_result.params["frequency"].value,
        }
        return quantities

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

    def save_quantities_of_interest(tuid: str, quantities_of_interest: dict) -> None:
        exp_folder = Path(locate_experiment_container(tuid))
        # Save fit results
        with open(exp_folder / "quantities_of_interest.json", "w") as file:
            json.dump(quantities_of_interest, file)


    def save_mpl_figure(tuid: str, fig: matplotlib.figure.Figure) -> None:
        exp_folder = Path(locate_experiment_container(tuid))
        fig.savefig(exp_folder / "Cosine fit.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

Now the execution of the entire analysis becomes much more readable and clean:

.. jupyter-execute::

    dataset = extract_data(label="Cosine experiment")
    fit_result = run_fitting(dataset=dataset)
    quantities_of_interest = analyze_fit_results(fit_result=fit_result)
    save_quantities_of_interest(dataset.tuid, quantities_of_interest)
    fig, ax = plt.subplots()
    plot_fit(fig=fig, ax=ax, dataset=dataset, fit_result=fit_result)
    save_mpl_figure(dataset.tuid, fig)

We can inspect the `experiment directory` which now contains the analysis results as expected:

.. jupyter-execute::

    from directory_tree import display_tree
    print(display_tree(locate_experiment_container(dataset.tuid), string_rep=True))

Creating a simple analysis class
--------------------------------

Even though we have improved code structure greatly, in order to execute the same analysis against some other dataset we would have to copy-paste a significant portion of code (the analysis steps).

We tackle this by taking advantage of the Object Oriented Programming (OOP) in python.
We will create a python class that serves as a structured container for data (attributes) and the methods (functions) that act on the information.

Some of the advantages of OOP are:

- the same class can be instantiated multiple times to act on different data while reusing the same methods;
- all the methods have access to all the data (attributes) associated with a particular instance of the class;
- subclasses can inherit from other classes and extend their functionalities.

Let's now observe how such a class could look like.

.. warning::

    This analysis class is intended for educational purposes only.
    It is not intended to be used as a template!
    See the end of the tutorial for the recommended usage of the analysis framework.

.. jupyter-execute::

    class MyCosineAnalysis:
        def __init__(self, label: str):
            """This is a special method that python calls when an instance of this class is created."""

            self.label = label

            # objects to be filled up later when running the analysis
            self.tuid = None
            self.dataset_raw = None
            self.fit_results = OrderedDict()
            self.quantities_of_interest = OrderedDict()
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
            self.save_quantities_of_interest()
            self.save_figures()

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
            self.quantities_of_interest.update(
                {
                    "amplitude": self.fit_results["cosine"].params["amplitude"].value,
                    "frequency": self.fit_results["cosine"].params["frequency"].value,
                }
            )

        def save_quantities_of_interest(self):
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

Running the analysis is now as simple as:

.. jupyter-execute::

    a_obj = MyCosineAnalysis(label="Cosine experiment")
    a_obj.run()
    a_obj.figs_mpl["cos-data-and-fit"]

The first line will instantiate the class by calling the :code:`.__init__()` method.

As expected this will save similar files into the `experiment directory`:

.. jupyter-execute::

    print(display_tree(locate_experiment_container(a_obj.dataset_raw.tuid), string_rep=True))

Extending the BaseAnalysis
--------------------------

While the above stand-alone class provides the gist of an analysis, we can do even better by defining a structured framework that all analysis need to adhere to and factoring out the pieces of code that are common to most analyses.
Beside that, the overall functionality can be improved.

Here is where the :class:`~quantify.analysis.base_analysis.BaseAnalysis` enters the scene.
It allows us to focus only on the particular aspect of our custom analysis by implementing only the relevant methods. Take a look at how the above class is implemented where we are making use of the analysis framework. For completeness, a fully documented :class:`~quantify.analysis.fitting_models.CosineModel` that can serve as a template is shown as well.

.. jupyter-execute::
    :hide-code:

    from quantify.analysis.cosine_analysis import CosineAnalysis

.. literalinclude:: ../../quantify/analysis/fitting_models.py
    :pyobject: CosineModel

.. literalinclude:: ../../quantify/analysis/cosine_analysis.py
    :pyobject: CosineAnalysis


Now we can simply execute it against our latest experiment as follows:


.. jupyter-execute::

    a_obj = CosineAnalysis(label="Cosine experiment").run()
    a_obj.display_figs_mpl()


Inspecting the `experiment directory` yields:

.. jupyter-execute::

    print(display_tree(locate_experiment_container(a_obj.dataset_raw.tuid), string_rep=True))


As you can conclude from the :class:`!CosineAnalysis` code, we did not implement quite a few methods in there.
These are provided by the :class:`~quantify.analysis.base_analysis.BaseAnalysis`.
To gain some insight on what exactly is being executed we can enable the logging module and use the internal logger of the analysis instance:

.. jupyter-execute::
    :stderr:

    import logging
    # activate logging and set global level to show warnings only
    logging.basicConfig(level=logging.WARNING)

    # set analysis logger level to info (the logger is inherited from BaseAnalysis)
    a_obj.logger.setLevel(level=logging.INFO)
    a_obj.run()
