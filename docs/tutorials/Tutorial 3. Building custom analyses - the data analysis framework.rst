.. _analysis_framework_tutorial:

Tutorial 3. Building custom analyses - the data analysis framework
==================================================================

.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 3. Building custom analyses - the data analysis framework`

    :jupyter-download:script:`Tutorial 3. Building custom analyses - the data analysis framework`

TODO: OVEVIEW STRUCTURE OF THE TUTORIAL

---

.. jupyter-execute::

    import lmfit
    import numpy as np
    import matplotlib.pyplot as plt
    from quantify.analysis.fitting_models import cos_func

.. include:: set_data_dir.rst.txt

Analyzing the experiment
------------------------

Quantify provides an analysis framework in the form of a :class:`~quantify.analysis.base_analysis.BaseAnalysis` class and several subclasses for simple cases (e.g., :class:`~quantify.analysis.base_analysis.Basic1DAnalysis`, :class:`~quantify.analysis.base_analysis.Basic2DAnalysis`, :class:`~quantify.analysis.spectroscopy_analysis.ResonatorSpectroscopyAnalysis`). The framework provides a structured, yet flexible, flow of the analysis steps. We encourage all users to adopt the framework by sub-classing the :class:`~quantify.analysis.base_analysis.BaseAnalysis`.

To give insight into the analysis framework, we first execute the analysis steps *manually*, and afterwards showcase how to encapsulate these steps into a reusable analysis class. Feel free to skip directly to `The analysis framework`_.

Manual analysis steps
~~~~~~~~~~~~~~~~~~~~~

:class: dropdown, toggle-shown

1. Loading the data

    The :class:`~xarray.Dataset` contains all the information required to perform basic analysis of the experiment and information on where the data is stored.
    We can alternatively load the dataset from disk based on it's :class:`~quantify.data.types.TUID`, a timestamp-based unique identifier. If you do not know the tuid of the experiment you can find the latest tuid containing a certain string in the experiment name using :meth:`~quantify.data.handling.get_latest_tuid`.
    See the data storage documentation for more details on the folder structure and files contained in the data directory.

    .. jupyter-execute::

        from quantify.data.handling import load_dataset, get_latest_tuid

        # here we look for the latest datafile in the datadirectory named "Cosine test"
        # note that this is not he last dataset but one dataset earlier
        tuid = get_latest_tuid('Cosine test')
        print('tuid: {}'.format(tuid))
        dset = load_dataset(tuid)

        dset

#. Performing fits and extracting quantities of interest

    We have used a cosine function to "mock" an experiment, the goal of the experiment is to find the underlying parameters.
    We extract these parameters by performing a fit to a model, which coincidentally, is based on the same cosine function.
    For fitting we recommend using the lmfit library.  See https://lmfit.github.io/lmfit-py/model.html on how to fit data to a custom model.

    .. jupyter-execute::

        # we create a model based on our function
        mod = lmfit.Model(cos_func)
        # and specify initial guesses for each parameter
        mod.set_param_hint('amplitude', value=.8, vary=True)
        mod.set_param_hint('frequency', value=.4)
        mod.set_param_hint('phase', value=0, vary=False)
        mod.set_param_hint('offset', value=0, vary=False)
        params = mod.make_params()
        # and here we perform the fit.
        fit_res = mod.fit(dset['y0'].values, x=dset['x0'].values, params=params)

        # It is possible to get a quick visualization of our fit using a build-in method of lmfit
        fit_res.plot_fit(show_init=True)


    .. jupyter-execute::

        fit_res.params


    .. jupyter-execute::

        # And we can print an overview of the fitting results
        print(fit_res.fit_report())


#. Plotting and saving the results of the analysis

    .. jupyter-execute::

        # We include some visualization utilities in quantify
        from quantify.visualization.SI_utilities import set_xlabel, set_ylabel


    .. jupyter-execute::

        fig, ax = plt.subplots()

        ax.plot(dset['x0'], dset['y0'], marker='o', label='Data')
        x_fit = np.linspace(dset['x0'][0], dset['x0'][-1], 1000)
        y_fit = cos_func(x=x_fit, **fit_res.best_values)
        ax.plot(x_fit, y_fit, label='Fit')
        ax.legend()

        set_xlabel(ax, dset['x0'].attrs['long_name'], dset['x0'].attrs['units'])
        set_ylabel(ax, dset['y0'].attrs['long_name'], dset['y0'].attrs['units'])
        ax.set_title('{}\n{}'.format(tuid, 'Cosine test'))

    Now that we have analyzed our data and created a figure, we probably want to store the results of our analysis.
    We will want to store the figure and the results of the fit in the `experiment folder`.


    .. jupyter-execute::

        from quantify.data.handling import locate_experiment_container
        # Here we are using this function as a convenient way of retrieving the experiment
        # folder without using an absolute path
        exp_folder = locate_experiment_container(dset.tuid)


    .. jupyter-execute::

        from os.path import join
        # Save fit results
        lmfit.model.save_modelresult(fit_res, join(exp_folder, 'fit_res.json'))
        # Save figure
        fig.savefig(join(exp_folder, 'Cosine fit.png'), dpi=300, bbox_inches='tight')

The analysis framework
----------------------

