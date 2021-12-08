=========
Changelog
=========


0.5.2 (2021-12-08)
------------------

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Data - Introduced a QuantifyExperiment class within the data.experiment module to separate data handling responsibilities from MeasurementControl. (!273, !274)
* Docs - Added quantify logo to the documentation. (!263)
* Infrastructure - Fixes the latest tests. (Except for Sphinx issues) (!275)
* Infrastructure - Fixes the tests temporarily by pinning matplotlib 3.4.3 (!269)
* Infrastructure - Added prospector config file for mypy in codacy. (copy from quantify-scheduler) (!259)
* Bugfix - Fix a bug in adjust_axeslabels_SI. (!272)


0.5.1 (2021-11-01)
------------------

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Analysis - Automatically rotate Rabi data to the axis with the best SNR (#249, !223)
* Analysis - Added support for calibration points to rotate and scale data to a calibrated axis for single-qubit timedomain experiments (T1, Echo, Ramsey and AllXY) (#227,  !219)
* Analysis - Added extra constraints to fits for T1, Echo and Ramsey when using a calibrated axis (T1, Echo, Ramsey) (#236,  !219)
* Analysis - Removed requirement for data on which to perform timedomain analysis to be acquired in radial coordinates (#227, !213).
* Analysis - Removed positive amplitude constraint from Rabi analysis (!213).
* Analysis - Detect calibration points automatically for single qubit time-domain experiments (!234)
* Docs - Added bibliography with sphinxcontrib-bibtex extension (!207).
* Docs - Added notebook_to_jupyter_sphinx sphinx extension converter for tutorials writing (!220).
* Docs - Add qcodes parameters docs to sphinx build (!255)
* Docs - Adds a notebook to jupyter sphinx converter for tutorials writing. (!220)
* MeasurementControl - Added representation with summary of settables, gettables and setpoints (!222).
* MeasurementControl - Added lazy_set functionality to avoid setting settables to same value (#261, !233).
* InstrumentMonitor - Extended Instrument Monitor to handle submodules and channels (#213, !226).
* Data - Adopted new specification for dataset v2.0 format. (!224)
* Infrastructure - Adds additional pre-commit and pre-push hooks (!254)
* Infrastructure - Ensure line endings are always committed with unix-like style (!227)
* Visualization - Factor out plotmon refresh from MeasurementControl (!248)
* Bugfix - Solved a bug where a fit would fail for a Ramsey experiment with negative values (#246, !219)
* Bugfix - Rabi analysis for negative signal amplitudes can now converge. (!213)
* Bugfix - Fixed divide by 0 warning in resonator spectroscopy analysis (!216).
* Bugfix - Fixed snapshot failing for qcodes instruments with dead weakrefs (!221).
* Bugfix - load_settings_onto_instrument does not try to set parameters to None if they are already None (#232, !225)
* Bugfix - replace OrderedDict with dict (!237)
* Bugfix - Fixes to utilities.general and function rename (!232)
* Bugfix - Fixes temporarily the funcparserlib failing rtd. (!249)
* Bugfix - alpha settings_overwrite["mpl_transparent_background"] = False (!236)
* Bugfix - Ramsey analysis cal points (!235)
* Bugfix - Ensures MeasurementControl representation works even when instruments are closed/freshly instantiated. (follow up from !226) (!229)
* Bugfix - fix snapshot for dead instruments (!221)
* Bugfix - The load_settings_onto_instrument function no longer attempts to set a QCoDeS parameter to None in certain cases. (!225)
* Bugfix - Fix filelock logging (!238)
* Bugfix - Fix divide by 0 which gives warning in resonator analysis (!216)
* Bugfix - Fix a bug in adjust_axeslabels_SI where it would update a label if no unit was provided (!272)

0.5.0 (2021-08-06)
------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Change of namespace from quantify.* to quantify_core.*

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Change namespace. (!195)
* Support xarray >0.18.0. (!198, #223)
* Unpinned pyqt5 version in setup to fix bug on macOS big sur. (!203)
* Added an example stopwatch gettable. (!187)
* Added new utility class quantify_core.utilities.inspect_utils. (!190, !192)
* Delete print(list) statement from locate_experiment_container. (!194)
* Allow for unit-aware printing of floats and other values with no error. (!167, #193)
* Plotmon: support non-linear (e.g., logarithmic space) for x and y coordinates. (!201)
* Consistency of naming conventions in analysis code. (!188)
* Ramsey analysis. (!166)
* Echo analysis. (!176)
* AllXY analysis. (!177)
* Interpolated 2D analysis and ND optimization analysis. (!180)
* Quantities of interest saving to JSON now supports more types, including uncertainties.ufloats. (!164, #152)

0.4.0 (2021-05-10)
------------------

* Release of the analysis framework including basic analyses, example classes and documentation on how to make a custom analysis class.
* Various bug fixes.
* First beta-release of quantify-core.

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Defined how to pass optional arguments for analysis subclasses. (#184, !158)
* Added warning when an analysis class returns a bad fit and improve UX. (!163)
* Renamed analysis variables `.dataset` and `.dataset_raw` for consistency. (#197, !171)
* add support for ufloat in format_value_string (!151)
* Provide methods for loading quantities of interest and processed dataset. (#191, !165)
* Added Rabi analysis subclass. (!159)
* fix for the multiple kwargs keys being passed into ax.text (!154)
* log the pip packages in the CI (same as in scheduler) (!168)
* UX improvements over current analysis flow control interrupt_before interface. (#183, !158)
* Allow providing an xarray dataset as input for analysis (#181, !156)
* Adds pytest fixture tmpdir_factory whenever possible. (!162)
* Fixes a bug with range-casting in the plot_fit function in the mpl_plotting module (!142)
* Utility function to handle the None edge case when converting lmfit pars to ufloat (#186, !160)
* T1 analysis (!137)
* Fixed a bug with loading settings onto an instrument (#166, !139)
* Storing quantities of interest in spectroscopy analysis simplified (!152)
* fix warning: Using a non-tuple sequence for multidimensional indexing is deprecated (!147)
* simplified header for all python files (#92, !146)
* Drop MeasurementControl soft_avg parameter in favor of MC.run(soft_avg=3) (!144)
* Better displaying of lmfit parameters and standard errors (!133)
* Plot duplicate setpoints in a 1D dataset (#173, !134)
* Downgrade and pin pyqt5 version (#170, !134)
* Sphinx autodoc function parameters and output types based on type hints!113
* Implemented :code:`numpy.bool_` patch for xarray 0.17.0 (temp fix for #161, !131)

Breaking changes
~~~~~~~~~~~~~~~~

* Analysis steps execution refactored and added optional arguments through `.run` (#184, !158)
    - Any analysis class now requires explicit execution of the steps with `.run()`.
    - One-liner still available `a_obj = MyAnalysisClass().run()`

* Analysis dataset variables and filename changed for consistency (!171):
    - `BaseAnalysis.dataset_raw` renamed to `BaseAnalysis.dataset`
    - `BaseAnalysis.dataset` renamed to `BaseAnalysis.dataset_processed`
    - "processed_dataset.hdf5" renamed to "dataset_processed.hdf5"
* The MeasurementControl soft_avg parameter has been removed. The same fucntionality is now available through MC.run(soft_avg=n) (!144)


0.3.2 (2021-03-17)
------------------

* Analysis framework beta version (limited documentation).
* Measurement control supports an inner loop in batched mode with outer iterative loops.
* Improvements to the dataset format (potentially breaking changes, see notes below).

    * Support of complex numbers and arrays in the dataset storage through `h5netcdf` engine.
    * Proper use of the coordinate property of xarray in quantify datasets.
* New data handling utilities: `load_dataset_from_path`, `locate_experiment_container`, `write_dataset`.
* Keyboard interrupt and Jupyter kernel interrupts are handled safely in MeasurementControl.
* Improved and more extensive documentation.
* Various bug fixes.


Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Dataset format changed to use Xarray Coordinates [no Dimensions] (!98)
* Added batched mode with outer iterative loops (!98)
* Switched default dataset engine to support complex numbers #150 (!114)
* Analysis class, framework, subclass examples #63 (!89, !122, !123)
* Cyclic colormaps auto-detect in 2D analysis (!118, !122)
* Safely handle Keyboard interrupt or Jupyter kernel interrupts (!125, !127)


Potentially breaking changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please see merge request !98 for a python snippet that will make all previous datasets compliant with this change!
Note that this is only required if you want to load old datasets in `quantify.visualization.pyqt_plotmon.PlotMonitor_pyqt`.

* Dataset format is updated to use Xarray Coordinates [no Dimensions] (!98)
* The TUID class is only a validator now to avoid issues with `h5netcdf`


0.3.1 (2021-02-15)
------------------

* Added function to load settings from previous experiments onto instruments (load_settings_onto_instrument).
* Added support for @property as attributes of Settables/Gettables.
* Migrated code style to black.
* Fixed support for python3.9.
* Significant improvements to general documentation.
* Improved installation instructions for windows and MacOS.
* Changed the dataset .unit attribute to .units to adopt xarray default (Breaking change!).
* Various minor bugfixes.


Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Windows install instr (!79)
* Load instrument settings (!29)
* Docs/general fixes (!82)
* updated copyright years (!84)
* Hotfix makefile docs (!83)
* Hot fix tuids max num (!85)
* added reqs for scipy, fixes #133 (!87)
* Added link on cross-fork collaboration (!90)
* Allow easy access to the tests datadir from a simple import (!95)
* Add custom css for rtd (!27)
* Dset units attr, closes #147 (!101)
* Add setGeometry method to instrument monitor and plotmon (!88)
* Enforce a datadir choice to avoid potential data loss (!86)
* Migrated code style to black (!93)
* Fixed support for python3.9 (!94)
* Added support for dynamic change of datadir for plotmon (!97)
* Added support for @property as attributes of Settables/Gettables (!100)
* "unit" attr of xarray variables in dataset changed to "units" for compatibility with xarray utilities. (!101)
* Updated numpy requirement (!104)
* Updated installation intructions for MacOS #142 (!99)
* Bugfix for get tuids containing method (!106)

Breaking changes
~~~~~~~~~~~~~~~~

Please see merge request !101 for a python snippet that will make all previous datasets compliant with this breaking change!

* "unit" attr of xarray variables in dataset changed to "units" for compatibility with xarray utilities. (!101)


0.3.0 (2020-12-17)
------------------

* Persistence mode feature added to the plotting monitor responsible for visualization during experiments, see also tutorial 4 in the docs.
* Instrument monitor feature added to support live snapshot monitoring during experiments.
* Renaming of [soft, hard]-loops to [iterative, batched]-loops respectively.
* Adds t_start and t_stop arguments to the function get_tuids_containing in quantify.data.handling.
* Various bug fixes and improvements to documentation.

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fix for pyqtgraph plotting and instrument monitor remote process sleeping !81.
* Plotting monitor is now running in a completely detached process !78.
* Persistence mode added to the plotting monitor !72.
* Adds explicit numpy version number (==1.19.2) requirement for windows in the setup. (!74).
* Improved documentation on how to set/get the datadirectory #100 (!71)
* Batched refactor. Closes #113 (!69).
* Instrument monitor feature added. Closes #62 (!65).
* Hot-fix for exception handling of gettable/settable in MC. Closes #101 (!64).
* Added t_start and t_stop arguments to get_tuids_containing function within quantify.data.handling. Closes #69 (!57, !62).
* Fix for the case when MC does not call finish on gettable. Closes #96 (!60).




0.2.0 (2020-10-16)
------------------

* Repository renamed to quantify-core.
* Scheduler functionality factored out into quantify-scheduler repository.

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* !11 Advanced MC, closed #13.
* First prototype of sequencer #16 (!13), moved to quantify-scheduler.
* Documentation of sequencer datatypes #19 (!13), moved to quantify-scheduler.
* Simplified settable gettable interface #32 (!15).
* Keyboard interrupt handler for Measurement Control #20 (!12).
* Documentation for gettable and settable #27 (!14).
* Sequencer hardening and cleanup (!16), moved to quantify-scheduler.
* CZ doc updates and rudimentary CZ implementation (!18), moved to quantify-scheduler.
* Pulsar asm backend (!17), moved to quantify-scheduler.
* Minor fixes sequencer (!19), moved to quantify-scheduler.
* Utility function to get_tuids_containing #48 (!22).
* Enable modulation bugfix #42 (!23), moved to quantify-scheduler.
* Added copyright notices to source files #36 (!25).
* Custom readthedocs theme to change column width, fixes #28 (!27).
* Amplitude limit on waveforms #41 (!24), moved to quantify-scheduler.
* Pulse diagram autoscaling bufix #49 (!26), moved to quantify-scheduler.
* Implementation of adaptive measurement loops in the measurement control #24 (!21)
* Load instrument settings utility function #21, !29.
* Support for data acquisition in sequencer (!28), moved to quantify-scheduler.
* Documentation for data storage, experiment containers and dataset #7 (!20).
* Function to create a plot monitor from historical data #56 (!32).
* Bugfix for buffersize in dynamically resized dataset (!35).
* Bugfix for adaptive experiments with n return variables (!34)
* Exteneded sequencer.rst tutorial to include QRM examples (!33), moved to quantify-scheduler.
* Refactor, Moved quantify-scheduler to new repository (!37).
* Gettable return variables made consistent for multiple gettables #68 (!38).
* Contribution guidelines updated #53 (!31).
* Bugfix for unexpected behaviour in keyboard interrupt for measurements #73 (!39)
* Documentation improvements #71 (!40).
* Improvements to tutorial !41.
* Removed visualization for scheduler !43.
* Fix broken links in install and contributions !44.
* Fixes bug in TUID validator #75 (42).
* Standardize use of numpydoc accross repo #67 (!46).
* Fix for online build on readthedocs !47.
* CI hardening, base python version for tests is 3.7 (minimum version) !50.
* New data folder structure (Breaking change!) #76 (!48).
* Updated installation guide #77 (!49).
* Minor changes to RTD displaying issues !51.
* Convert jupyter notebooks to .rst files with jupyter-execute (!52).
* Cleanup before opening repo #86 and #82 (!53)


0.1.1 (2020-05-25)
------------------
* Hotfix to update package label and fix PyPI


0.1.0 (2020-05-21)
------------------

* First release on PyPI.



.. note::

    * # denotes a closed issue.
    * ! denotes a merge request.
