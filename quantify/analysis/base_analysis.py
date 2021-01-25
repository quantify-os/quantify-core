"""
This module should contain different analyses corresponding to discrete experiments
"""
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from quantify.data.handling import load_dataset, get_latest_tuid, _locate_experiment_file, get_datadir
from quantify.visualization.SI_utilities import set_xlabel, set_ylabel

DPI = 600  # define a constant for data saving


class BaseAnalysis(ABC):
    """
    Abstract base class for data analysis. Provides a template from which to
    inherit when doing any analysis.
    """

    def __init__(self, label: str = '', tuid: str = None,
                 close_figs: bool = True):
        """
        Initializes the variables that are used in the analysis and to which data is stored.

        Parameters
        ------------------
        label: str
            Will look for a dataset that contains "label" in the name.
        tuid: str
            If specified, will look for the dataset with the matching tuid.
        close_figs: bool
            If True, closes matplotlib figures after saving
        """

        self.label = label
        self.tuid = tuid
        self.close_figs = close_figs

        # This will be overwritten
        self.dset = None

        self.fit_res = None
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

        This function is typically called after the __init__.
        """
        self.extract_data()  # extract data specified in params dict
        self.process_data()  # binning, filtering etc

        self.prepare_fitting()  # set up fit_dicts
        self.run_fitting()  # fitting to models
        self.save_fit_results()
        self.analyze_fit_results()  # analyzing the results of the fits

        self.save_quantities_of_interest()

        self.prepare_figures()
        self.create_figures()
        # save stuff
        self.save_figures()

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

    def analyze_fit_results(self):  # analyzing the results of the fits
        pass

    def save_quantities_of_interest(self):
        pass

    def save_figures(self):

        for figname, fig in self.figs.items():
            filename = _locate_experiment_file(
                self.tuid, get_datadir(), '{}'.format(figname))
            fig.savefig(filename+'.png', bbox_inches='tight', dpi=DPI)
            fig.savefig(filename+'.svg', bbox_inches='tight')
            if self.close_figs:
                plt.close(fig)

    def create_figures(self):
        # FIXME: in the simpler world, this will be overwritten.

        # Set up figures and axes
        if not hasattr(self, 'figs'):
            self.figs = {}

        # if no custom axs_dict is provided, create them based on the keys
        # in the axs_dict

        # Auto generate the figures and axes.
        if not hasattr(self, 'axs_dict'):
            self.axs_dict = {}

            for key, pdict in self.plot_dicts.items():
                # If no ax_id is specified, a new figure needs to be set up.
                if 'ax_id' not in pdict.keys():
                    f, ax = plt.subplots(
                        figsize=pdict.get('figsize', None))
                    # transparent background around axes for presenting data
                    self.figs[key] = f
                    self.axs_dict[key] = ax
                    f.patch.set_alpha(0)

        for key, pdict in self.plot_dicts.items():
            ax_id = pdict.get('ax_id', key)
            ax = self.axs_dict[ax_id]
            pdict['plot_fn'](ax=ax, **pdict)


class Basic1DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def prepare_figures(self):

        self.plot_dicts = {}

        # iterate over
        for i in range(len(self.dset.keys())-1):
            self.plot_dicts['x0-y{}'.format(i)] = {
                'plot_fn': plot_basic1D,
                'x': self.dset['x0'].values,
                'xlabel': self.dset['x0'].attrs['long_name'],
                'xunit': self.dset['x0'].attrs['unit'],
                'y': self.dset['y{}'.format(i)].values,
                'ylabel': self.dset['y{}'.format(i)].attrs['long_name'],
                'yunit': self.dset['y{}'.format(i)].attrs['unit'],
                'title': 'x0-y{} {}\ntuid: {}'.format(
                    i, self.dset.attrs['name'], self.dset.attrs['tuid'])
            }


class Basic2DAnalysis(BaseAnalysis):
    """
    A basic analysis that extracts the data from the latest file matching the label
    and plots and stores the data in the experiment container.
    """

    def prepare_figures(self):

        self.plot_dicts = {}

        # iterate over
        for i in range(len(self.dset.keys())-1):
            self.plot_dicts['x0-y{}'.format(i)] = {
                'plot_fn': plot_basic1D,
                'x': self.dset['x0'].values,
                'xlabel': self.dset['x0'].attrs['long_name'],
                'xunit': self.dset['x0'].attrs['unit'],
                'y': self.dset['y{}'.format(i)].values,
                'ylabel': self.dset['y{}'.format(i)].attrs['long_name'],
                'yunit': self.dset['y{}'.format(i)].attrs['unit'],
                'title': 'x0-y{} {}\ntuid: {}'.format(
                    i, self.dset.attrs['name'], self.dset.attrs['tuid'])
            }


def plot_basic1D(x, y, xlabel, xunit, ylabel, yunit, ax, title=None, plot_kw=None, **kw):
    if plot_kw is None:
        plot_kw = {}  # to prevent introducing memory bug

    if ax is None:
        f, ax = plt.subplots()
    ax.plot(x, y, **plot_kw)
    if title is not None:
        ax.set_title(title)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)


def plot_fit(ax, fit_res, plot_init=True, plot_numpoints=1000, **kw):
    model = fit_res.model

    if len(model.independent_vars) == 1:
        independent_var = model.independent_vars[0]
    else:
        raise ValueError('Fit can only be plotted if the model function'
                         ' has one independent variable.')

    x_arr = fit_res.userkws[independent_var]
    x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
    y = model.eval(fit_res.params, **{independent_var: x})
    ax.plot(x, y, label='Fit', c='C3')

    x = np.linspace(np.min(x_arr), np.max(x_arr), plot_numpoints)
    y = model.eval(fit_res.init_params, **{independent_var: x})
    ax.plot(x, y, ls='--', c='grey', label='Guess')
