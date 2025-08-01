# Changelog

## Release v0.8.3 (2025-07-23)

### 🐛 Bug Fixes and Closed Issues

- Fixed a bug where `ThresholdedAcquisitions` with `binMode.AVERAGE` would return 0 instead of the average results. ([!566](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/566) by [@Thomas Middelburg](https://gitlab.com/ThomasMiddelburg))

### New Features

- Option to not save fit results in Analysis classes ([!565](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/565) by [@Adam Lawrence](https://gitlab.com/adamorangeQS))



## Release v0.8.2 (2025-05-28)

### 🐛 Bug Fixes and Closed Issues
- Resolve "Broken pipelines for python 3.11 and 3.12" ([!563](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/563) by [@Olga Lebiga](https://gitlab.com/olebiga))
- Broken pipelines for python 3.11 and 3.12 ([#382](https://gitlab.com/quantify-os/quantify-core/-/issues/382) by [@Olga Lebiga](https://gitlab.com/olebiga))
- Correct qubit assignment with `ReadoutCalibrationAnalysis`. ([!561](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/561) by [@Rohit Navarathna](https://gitlab.com/rnavarathna))

### ✨ New Features
- Allow Lorentzian Model to also fit descending peaks ([!562](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/562) by [@Basak Ozcan](https://gitlab.com/bozcan))
- TUID timestamps are now processed 8 times faster ([!559](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/559) by [@Pieter Eendebak](https://gitlab.com/peendebak))


### 🔧 Other

- Change return typehint of `BaseAnalysis.run` to 'Self' ([!560](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/560) by [@Timo van Abswoude](https://gitlab.com/Timo_van_Abswoude))



## Release v0.8.1 (2025-03-20)

### 🚀 Enhancements
- Add ``WeightedThresholdedAcquisition`` protocol compatibility to ``MeasurementControl`` ([!556](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/556) by [@Adithyan](https://gitlab.com/adithyan2))

## Release v0.8.0

### 🐛 Bug Fixes and Closed Issues

- Change logging level in BaseAnalysis from 'info' to 'debug' (https://gitlab.com/quantify-os/quantify-core/-/merge_requests/548) by [@Timo van Abswoude](https://gitlab.com/Timo_van_Abswoude)
- Fix typo in measurement control documentation (https://gitlab.com/quantify-os/quantify-core/-/merge_requests/551) by [@Robert Sokolewicz](https://gitlab.com/rsokolewicz)
- 

### 🚀 Enhancements

- Add **repr** method in BaseAnalysis (https://gitlab.com/quantify-os/quantify-core/-/merge_requests/554 by [@Sibasish Mishra](https://gitlab.com/sibasish-orangeqs)
- Support compression for QCoDeS instruments snapshots in measurement control (https://gitlab.com/quantify-os/quantify-core/-/merge_requests/549) by [@Mahmut Cetin](https://gitlab.com/cetin-oqs)

### 🔧  Other

- Improve type hinting of optionals in base analysis (https://gitlab.com/quantify-os/quantify-core/-/merge_requests/553) by [@Timo van Abswoude](https://gitlab.com/Timo_van_Abswoude)


## Release v0.7.8 (2024-10-14)

### 🐛 Bug Fixes and Closed Issues

- Fix remove distutils dependency ([!542](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/542) by [@Robert Sokolewicz](https://gitlab.com/rsokolewicz))

## Release v0.7.8 (2024-10-14)

### 🐛 Bug Fixes and Closed Issues

- Fix remove distutils dependency ([!542](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/542) by [@Robert Sokolewicz](https://gitlab.com/rsokolewicz))


## Release v0.7.7 (2024-10-08)

### 🐛 Bug Fixes and Closed Issues
- Allow any callable for adaptive functions and raise TypeError if not valid ([!539](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/539) by [@Timo van Abswoude](https://gitlab.com/Timo_van_Abswoude))
- Fix xarray dimension mismatch in tutorial ([!533](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/533) by [@Robert Sokolewicz](https://gitlab.com/rsokolewicz))
- Fix name generation for parameter in ChannelTuple ([!516](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/516) by [@Eugene Huang](https://gitlab.com/eugenhu))
- Refactor keyboard interrupt handling ([!529](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/529) by [@Michiel Haye](https://gitlab.com/hayemj))
- Undesirable side effects of keyboard interrupt handling implementation ([#375](https://gitlab.com/quantify-os/quantify-core/-/issues/375) by [@Michiel Haye](https://gitlab.com/hayemj))
- Remove distutils dependency ([!541](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/541) by [@Robert Sokolewicz](https://gitlab.com/rsokolewicz))

### 🚀 Enhancements
- Improve performance of get_tuids_containing ([!536](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/536) by [@Pieter Eendebak](https://gitlab.com/peendebak))
- Add experiment_name property to the QuantifyExperiment ([!530](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/530) by [@Pieter Eendebak](https://gitlab.com/peendebak))
- Refactor keyboard interrupt handling ([!529](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/529) by [@Michiel Haye](https://gitlab.com/hayemj))

### 📦 Dependencies
- Add diff-cover to test dependecies ([!534](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/534) by [@Leon Wubben](https://gitlab.com/LeonQblox))
- Add pytest-mpl dependency ([!531](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/531) by [@Thomas Middelburg](https://gitlab.com/ThomasMiddelburg))

### 🔧 Other
- Allow any callable for adaptive functions and raise TypeError if not valid ([!539](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/539) by [@Timo van Abswoude](https://gitlab.com/Timo_van_Abswoude))
- Allow any callable for adaptive functions and raise TypeError if not valid ([!539](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/539) by [@Timo van Abswoude](https://gitlab.com/Timo_van_Abswoude))
- Fix DeprecationWarning from re ([!535](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/535) by [@Robert Sokolewicz](https://gitlab.com/rsokolewicz))
- Update MR template with changelog breaking change and link changes ([!537](https://gitlab.com/quantify-os/quantify-core/-/merge_requests/537) by [@Gábor Oszkár Dénes](https://gitlab.com/gdenes))

## 0.7.6 (2024-07-04)

### Merged branches and closed issues

- Requirements
  - Add scikit-learn as a dependency (!519)
  - Pin numpy<2.0.0 (!527)

- MeasurementControl 
  - In `measurement_description`, use `np.asarray` to convert the `_setpoints` to a numpy array before getting the shape. (!520)

- Analysis
  - Use the current default colormap for all 2D heatmaps generated by the `Basic2DAnalysis` (!521)
  - Add the `BaseAnalysis.adjust_cmap` method to change the colormap of a `matplotlib` axis (!521)
  - Overload `base_anysis.wrap_text` with signatures that return `str` (`None`) when the passed text is of type `str` (`None`) (!523)
  - Add an html representation `_repr_html_`  to `BaseAnalysis` that shows .svg images generated by an analysis class (!526)

## 0.7.5 (2024-04-11)

### Merged branches and closed issues

- SI utilities 
  - Cast large values for seconds to minutes or hours (!501)
  - Add option `auto_scale` to `set_xlabel` and `set_ylabel` (!512)

- Documentation
  - Update copyright notice to 2024. (!506)

- Analysis 
  - Added analysis class for resonator flux spectroscopy. (!493)
  - Add analysis class for qubit flux spectroscopy. (!473)
  - Add analysis class for readout calibration. (!474)
  - Add analysis class for the CZ conditional oscillation experiment. (!472)
  - Fix units of Rabi analysis amplitudes. (!511)
  - Remove an `xarray` `FutureWarning` in `to_gridded_dataset` when retrieving the dataset dimension names. (!510)
  - Remove an `xarray` `DeprecationWarning` about `argmin` and `argmax` in the `RabiAnalysis`. (!513)
  - Remove a `matplotlib` `MatplotlibDeprecationWarning` in `Basic2DAnalysis` when plotting a heatmap. (!510)

- GitLab
  - Make general Merge Request template the default. (!507)
  - Add documentation redirect instructions to Release merge request template. (!507)

- Tests
  - Small refactor of tests to get rid of `PytestRemovedIn8Warning`s. (!508)

- MeasurementControl
  - Fix overflow warning when running `MeasurementControl.run_adaptive`. (!515)
  - `grid_setpoints` now returns a list of M ndarrays, where M is the number of settables. Note that this is the transpose of the previous return type. The data type of the input setpoints is now preserved. (!505) 
  
- Linting
  - replaced deprecated numpy definitions with numpy2.0 compatible alternatives. (!517)

## 0.7.4 (2023-12-15)

### Merged branches and closed issues

- Documentation
  - Improve documentation build time and enable `sphinx-autobuild`. (!471)
  - Fix missing images in Jupyter cell outputs in documentation deployed using Gitlab Pages. (!480)
  - Switch to `pydata` sphinx theme on www.readthedocs.org. (!479)
  - Add colored terminal output when building documentation. (!502)

- Data 
  - The `long_name` and `name` attributes of dataset coordinates now contain information about the root instrument and submodules in addition to the settable. (!478, !486)

- Utilities 
  - Add `compare_snapshots` utility function for comparing snapshots. (!485)

- Analysis 
  - Add `TimeOfFlightAnalysis` analysis class for a time of flight measurement. (!494)

- Visualization
  - Fix `plot_2d_grid` plots so that the x and y axes are used in the correct order for both ascending and descending arrays. (!500, #369)
  - Handle `nan` values in `SI_utilities.value_precision` input, to improve plot monitor stability. (!487)

- MeasurementControl
  - Improve progress bar for iterative and batch run with `tqdm`. (!477, #346)
  - Prevent `numpy` deprecation warning, by explicit handling of `numpy.arrays` when converting a dimensional array into scalar. (!476)

## 0.7.3 (2023-08-17)

### Merged branches and closed issues

- Deprecation - Avoid deprecated code in validator.iter_errors (!475)
- Analysis - Added analysis class for qubit spectroscopy (!463)
- Deprecation - The `@deprecated` decorator now returns a function when decorating a function, instead of a class with a `__call__` method (!462).
- Documentation - Add install instructions for macOS users (!464)
- Documentation - Update broken urls in documentation (!458)
- Documentation - Remove `jupyter-sphinx` extension and port snippets formerly served by it into a How-To section in the documentation. (!460)
- Documentation - Fix broken xarray display in docs and resolve sphinx warnings (!470)
- Linting - Minor changes to `data/handing.py` to make it ruff compliant (!457)
- MeasurementControl - Add `get_idn()` method, without it will generate warnings in using recent versions of QCoDeS (!459)
- Replace usage of the deprecated `qcodes.plots` with `qcodes_loop.plots` in the remote plot monitor (!465)
- Visualization - Minor refactor to make visualization module compatible with `scipy>=1.11` (!466)

## 0.7.2 (2023-05-02)

### Merged branches and closed issues

- Analysis - Allow adding additional arguments to `create_figures()` method of classes that inherit from `BaseAnalysis` (#364, !454)
- Git - Change back to default merge strategy for CHANGELOG.md (!449)
- Linting - minor changes to satisfy pyright (!445)
- QCoDeS - Add qcodes-loop as dependency to ensure `InstrumentMonitor` runs correctly (!452)
- Utilities - Fix bug and clean up code in `load_settings_onto_instrument` by only trying to get and set parameters in `_try_to_set_par_safe` (!447, #360)
- Visualization - `set_xlabel`, `set_ylabel` and `set_cbarlabel` now add an offset to the values displayed on the tick labels, if that is needed to properly display small ranges with a large offset (!450, #165)

## 0.7.1 (2023-03-09)

### Breaking changes

- Requirements - `quantify-core` requires `qcodes>=0.37` (!439)

### Merged branches and closed issues

- Documentation - `mk_trace_for_iq_shot` now renders nicely in documentation (!429)
- Utilities - Fix bug where calling `without` to remove a key from a dict that did not have that key in the first place raises a `KeyError` (!438)
- Visualization - If provided, InstrumentMonitor displays instrument label (!369)

## 0.7.0 (2023-02-03)

### Breaking changes

- Requirements - `quantify-core` requires `qcodes>=0.35` now. (!427)
- Installation - Instead of `requirements.txt` and `requirements_dev.txt` `quantify-core` uses optional requirements. Use `pip install quantify-core[dev]` to install all of them. (!386)

### Merged branches and closed issues

- Documentation - Remove dependency on directory-tree (!431)
- Documentation - Sphinx build now compatible with qcodes==0.36.0 (!416, counterpart of quantify-scheduler!552)
- Analysis - Make all analysis classes available from `quantify_core.analysis` (!418)
- Data - Improve `set_datadir` and test style and readability (!419)
- Added AttributeError to the exceptions explictly handled in loading of datasets for the remote plotmon (#352, !442)
- Added exception handling for SI_utilities.value_precision (#350, !442)
- Installation - Refactor of setup configuration (!386, !433)
- Utilities - Remove `quantify_core.utilities.examples_support.mk_surface7_sched()` function that was used only for generating one non-essential figure in documentation, but caused dependency on `quantify-scheduler` to build it. This function is inlined in the documentation as an example. (!434)
- Utilities - Improved error handling for `quantify_core.utilities.experiment_helpers.load_settings_onto_instrument` (#351, #348, !425)

## 0.6.5 (2022-12-13)

### Merged branches and closed issues

- Data - Allow concatenation of datasets with different names (!389)
- Data - Allow concatenation of processed datasets (!394)
- Data - Update OS environment to disable HDF5 file locking (!404, !405)
- Documentation - Building sphinx documentation will now raise an error if one of the code cells fails to run (!407)
- Documentation - Fixed a typo in the figure text of the optimization analysis class. (!413)
- Fit - Add functions to load and save lmfit fit results from file (!286)
- Git - Changed git merge strategy to "union" for CHANGELOG.md and AUTHORS.md to reduce amount of merge conflicts (!399)
- Measurement Control - Add experiments_data submodule to measurement control (!393)
- Memory leak - Added ability to skip figure creation in analysis classes to prevent memory leaks in long-running nested experiments (!406)
- Memory leak - Fix a memory leak due to BaseAnalysis lru_cache (!390)
- QCoDeS - Create function `quantify_core.utilities.experiment_helpers.get_all_parents` to find all parents of QCoDeS submodule (!401)
- QCoDeS - Quantify now supports qcodes>=0.34.0 (!382)
- QuantifyFix - Bug in QuantifyExperiment when tuid is None (!396)
- Snapshots - Enable loading of settings of individual QCoDeS submodules and parameters from the saved snapshot (!384)
- Warnings - Deprecation warnings are now shown to end-users by default (by changing to `FutureWarning`) (!411)

## 0.6.4 (2022-10-13)

### Breaking changes

- MeasurementControl - Raise a `ValueError` exception if there are no setpoints (!370)
- Utilities - `make_hash` and `import_python_object_from_string` were removed, because they are not used in `quantify-core`. Use their counterparts from `quantify-scheduler`. (!371, quantify-os/quantify-scheduler!357)
- Docs - `notebook_to_jupyter_sphinx` sphinx extension has been removed (!378)
- Removed `BaseAnalysis.run_from()` and `BaseAnalysis.run_until()` functionality, which is almost unused by an average user. (!379)
- `BaseAnalysis.run()` function will return analysis object even if analysis has failed. Exceptions raised by analysis
  will be logged and not raised. (!379)
- The order of the arguments of `set_xlabel` and `set_ylabel` has been modified so that the `ax` is now optional argument (!376)

### Merged branches and closed issues

- Add functions to load and save lmfit fit results from file (!286)
- Remove various old temp requirement pins (counterpart of quantify-scheduler!447) (!368)
- Added support to SI_prefix_and_scale_factor for scaled units such as ns or GHz (!365, !373)
- InstrumentMonitor - Restore `initial_value` of `update_interval` param (removed in !324) (!375)
- Analysis - Use `long_name` instead of `name` XArray attribute as default for axis labels and plot messages (!380)
- Reduce quantify-core import time (!366)

## 0.6.3 (2022-08-05)

### Breaking changes

- MeasurementControl - Fix `print_progress` for 0 progress and do not raise when no setpoints (!363)

### Merged branches and closed issues

- Changes to InstrumentMonitor and PlotMonitor to fix errors due to racing conditions. (!358)
- Can save metadata with `QuantifyExperiment` (!355)
- Contribution guidelines updated, added: Versioning, Backward Compatibility and Deprecation Policy (!282)

## 0.6.2 (2022-06-30)

- Sanitize Markdown for proper display on PyPi.
- Data Handling - New function to extract parameters from snapshot, including submodules (!360)

## 0.6.1 (2022-06-30)

### Merged branches and closed issues

- Require lmfit >= 1.0.3 due to a typo within lmfit for the `guess_from_peak2d` function. (!346)
- Fixed calling `super().create_figures()` when inheriting analysis classes (issue was introduced by merge request !337). (#313)
- PlotMonitor - Fix crash in secondary plotmon when data is one-dimensional. (!349)
- Documentation sources are converted from restructured text format to MyST markdown. (!350)

## 0.6.0 (2022-05-25)

### Breaking changes

- Supported Python versions are now 3.8-3.10. Python 3.7 support is dropped. (!328)
- MeasurementControl - Removed the `instrument_monitor` (InstrumentRefParameter) from the `MeasurementControl` class. There is no need to couple the instrument monitor object with the `MeasurementControl` anymore. (!324)
- InstrumentMonitor - Made the `update` method private (renamed to `_update`). Also removed the `force` argument in the method. (!324)

### Merged branches and closed issues

- InstrumentMonitor - Display parameter values with non-number type nicely. Only for parameters without unit. (!336)
- Data Handling - Setting datadir to None now correctly sets datadir to default directory (~/quantify-data/). (!338)
- Data Handling - Added a DecodeToNumpy decoder class and function argument `list_to_ndarray` to the function `load_snapshot`, which enables `load_settings_onto_instrument` to support loading numpy arrays parameters from json lists. (!342, !343, #309)
- Data handling - Add API to save custom text files as experiment data. (!325)
- Data handling - Fix attributes handling when converting datasets with `to_gridded()` method. (!277, !302)
- Experiment helpers - Loading settings from snapshot now supports Instrument submodules. (#307, !338)
- Plotmon - Suppress warning about all-NaN datasets during plotting. (!314)
- Visualization - Added kwarg dicts to `plot_fit` to pass matplotlib keyword arguments and `plot_fit` returns list of matplotlib Line2D objects. (!334, !331 (closed))
- Analysis - CosineModel now guesses a frequency based on a Fourier transform of the data. (!335)
- Analysis - We do not store all Matplotlib figures and axes for all analysis objects in memory anymore. This fixes out-of-memory error for long measurement runs, when a lot of figures are created. (#298, !337, !345)
- MeasurementControl - Performance improvement in MeasurementControl data construction. (!333)
- Bugfix - Fix QHullError occurring in RemotePlotmon when supplying two uniformly spaced settables to MeasurementControl.setpoints(). (#305, !323)
- Packaging and distribution - Added support for PEP 561. (!322)

## 0.5.3 (2022-02-25)

### Merged branches and closed issues

- Analysis - Changed the metric of the AllXY analysis to use the mean of the absolute deviations. (!300)
- MeasurementControl - Added a `measurement_description` function in the MeasurementControl to return a serializable description of the latest measurement. (!279)
- MeasurementControl - Add option to run an experiment without saving data (!308)
- Infrastructure - Added utilities to support deprecation (!281)
- Bugfix - Fix qcodes 0.32.0 incompatibility by replacing all references of `qcodes.Instrument._all_instruments`. (!295)

## 0.5.2 (2021-12-08)

### Merged branches and closed issues

- Data - Introduced a QuantifyExperiment class within the data.experiment module to separate data handling responsibilities from MeasurementControl. (!273, !274)
- Docs - Added quantify logo to the documentation. (!263)
- Infrastructure - Fixes the latest tests. (Except for Sphinx issues) (!275)
- Infrastructure - Fixes the tests temporarily by pinning matplotlib 3.4.3 (!269)
- Infrastructure - Added prospector config file for mypy in codacy. (copy from quantify-scheduler) (!259)
- Bugfix - Fix a bug in adjust_axeslabels_SI. (!272)

## 0.5.1 (2021-11-01)

### Merged branches and closed issues

- Analysis - Automatically rotate Rabi data to the axis with the best SNR (#249, !223)
- Analysis - Added support for calibration points to rotate and scale data to a calibrated axis for single-qubit timedomain experiments (T1, Echo, Ramsey and AllXY) (#227,  !219)
- Analysis - Added extra constraints to fits for T1, Echo and Ramsey when using a calibrated axis (T1, Echo, Ramsey) (#236,  !219)
- Analysis - Removed requirement for data on which to perform timedomain analysis to be acquired in radial coordinates (#227, !213).
- Analysis - Removed positive amplitude constraint from Rabi analysis (!213).
- Analysis - Detect calibration points automatically for single qubit time-domain experiments (!234)
- Docs - Added bibliography with sphinxcontrib-bibtex extension (!207).
- Docs - Added notebook_to_jupyter_sphinx sphinx extension converter for tutorials writing (!220).
- Docs - Add qcodes parameters docs to sphinx build (!255)
- Docs - Adds a notebook to jupyter sphinx converter for tutorials writing. (!220)
- MeasurementControl - Added representation with summary of settables, gettables and setpoints (!222).
- MeasurementControl - Added lazy_set functionality to avoid setting settables to same value (#261, !233).
- InstrumentMonitor - Extended Instrument Monitor to handle submodules and channels (#213, !226).
- Data - Adopted new specification for dataset v2.0 format. (!224)
- Infrastructure - Adds additional pre-commit and pre-push hooks (!254)
- Infrastructure - Ensure line endings are always committed with unix-like style (!227)
- Visualization - Factor out plotmon refresh from MeasurementControl (!248)
- Bugfix - Solved a bug where a fit would fail for a Ramsey experiment with negative values (#246, !219)
- Bugfix - Rabi analysis for negative signal amplitudes can now converge. (!213)
- Bugfix - Fixed divide by 0 warning in resonator spectroscopy analysis (!216).
- Bugfix - Fixed snapshot failing for qcodes instruments with dead weakrefs (!221).
- Bugfix - load_settings_onto_instrument does not try to set parameters to None if they are already None (#232, !225)
- Bugfix - replace OrderedDict with dict (!237)
- Bugfix - Fixes to utilities.general and function rename (!232)
- Bugfix - Fixes temporarily the funcparserlib failing rtd. (!249)
- Bugfix - alpha settings_overwrite\["mpl_transparent_background"\] = False (!236)
- Bugfix - Ramsey analysis cal points (!235)
- Bugfix - Ensures MeasurementControl representation works even when instruments are closed/freshly instantiated. (follow up from !226) (!229)
- Bugfix - fix snapshot for dead instruments (!221)
- Bugfix - The load_settings_onto_instrument function no longer attempts to set a QCoDeS parameter to None in certain cases. (!225)
- Bugfix - Fix filelock logging (!238)
- Bugfix - Fix divide by 0 which gives warning in resonator analysis (!216)
- Bugfix - Fix a bug in adjust_axeslabels_SI where it would update a label if no unit was provided (!272)

## 0.5.0 (2021-08-06)

### Breaking changes

- Change of namespace from quantify.\* to quantify_core.\*

### Merged branches and closed issues

- Change namespace. (!195)
- Support xarray >0.18.0. (!198, #223)
- Unpinned pyqt5 version in setup to fix bug on macOS big sur. (!203)
- Added an example stopwatch gettable. (!187)
- Added new utility class quantify_core.utilities.inspect_utils. (!190, !192)
- Delete print(list) statement from locate_experiment_container. (!194)
- Allow for unit-aware printing of floats and other values with no error. (!167, #193)
- Plotmon: support non-linear (e.g., logarithmic space) for x and y coordinates. (!201)
- Consistency of naming conventions in analysis code. (!188)
- Ramsey analysis. (!166)
- Echo analysis. (!176)
- AllXY analysis. (!177)
- Interpolated 2D analysis and ND optimization analysis. (!180)
- Quantities of interest saving to JSON now supports more types, including uncertainties.ufloats. (!164, #152)

## 0.4.0 (2021-05-10)

- Release of the analysis framework including basic analyses, example classes and documentation on how to make a custom analysis class.
- Various bug fixes.
- First beta-release of quantify-core.

### Merged branches and closed issues

- Defined how to pass optional arguments for analysis subclasses. (#184, !158)
- Added warning when an analysis class returns a bad fit and improve UX. (!163)
- Renamed analysis variables `.dataset` and `.dataset_raw` for consistency. (#197, !171)
- add support for ufloat in format_value_string (!151)
- Provide methods for loading quantities of interest and processed dataset. (#191, !165)
- Added Rabi analysis subclass. (!159)
- fix for the multiple kwargs keys being passed into ax.text (!154)
- log the pip packages in the CI (same as in scheduler) (!168)
- UX improvements over current analysis flow control interrupt_before interface. (#183, !158)
- Allow providing an xarray dataset as input for analysis (#181, !156)
- Adds pytest fixture tmpdir_factory whenever possible. (!162)
- Fixes a bug with range-casting in the plot_fit function in the mpl_plotting module (!142)
- Utility function to handle the None edge case when converting lmfit pars to ufloat (#186, !160)
- T1 analysis (!137)
- Fixed a bug with loading settings onto an instrument (#166, !139)
- Storing quantities of interest in spectroscopy analysis simplified (!152)
- fix warning: Using a non-tuple sequence for multidimensional indexing is deprecated (!147)
- simplified header for all python files (#92, !146)
- Drop MeasurementControl soft_avg parameter in favor of MC.run(soft_avg=3) (!144)
- Better displaying of lmfit parameters and standard errors (!133)
- Plot duplicate setpoints in a 1D dataset (#173, !134)
- Downgrade and pin pyqt5 version (#170, !134)
- Sphinx autodoc function parameters and output types based on type hints!113
- Implemented {code}`numpy.bool_` patch for xarray 0.17.0 (temp fix for #161, !131)

### Breaking changes

- Analysis steps execution refactored and added optional arguments through `.run` (#184, !158)
  - Any analysis class now requires explicit execution of the steps with `.run()`.
  - One-liner still available `a_obj = MyAnalysisClass().run()`
- Analysis dataset variables and filename changed for consistency (!171):
  - `BaseAnalysis.dataset_raw` renamed to `BaseAnalysis.dataset`
  - `BaseAnalysis.dataset` renamed to `BaseAnalysis.dataset_processed`
  - "processed_dataset.hdf5" renamed to "dataset_processed.hdf5"
- The MeasurementControl soft_avg parameter has been removed. The same fucntionality is now available through MC.run(soft_avg=n) (!144)

## 0.3.2 (2021-03-17)

- Analysis framework beta version (limited documentation).
- Measurement control supports an inner loop in batched mode with outer iterative loops.
- Improvements to the dataset format (potentially breaking changes, see notes below).
  - Support of complex numbers and arrays in the dataset storage through `h5netcdf` engine.
  - Proper use of the coordinate property of xarray in quantify datasets.
- New data handling utilities: `load_dataset_from_path`, `locate_experiment_container`, `write_dataset`.
- Keyboard interrupt and Jupyter kernel interrupts are handled safely in MeasurementControl.
- Improved and more extensive documentation.
- Various bug fixes.

### Merged branches and closed issues

- Dataset format changed to use Xarray Coordinates \[no Dimensions\] (!98)
- Added batched mode with outer iterative loops (!98)
- Switched default dataset engine to support complex numbers #150 (!114)
- Analysis class, framework, subclass examples #63 (!89, !122, !123)
- Cyclic colormaps auto-detect in 2D analysis (!118, !122)
- Safely handle Keyboard interrupt or Jupyter kernel interrupts (!125, !127)

### Potentially breaking changes

Please see merge request !98 for a python snippet that will make all previous datasets compliant with this change!
Note that this is only required if you want to load old datasets in `quantify.visualization.pyqt_plotmon.PlotMonitor_pyqt`.

- Dataset format is updated to use Xarray Coordinates \[no Dimensions\] (!98)
- The TUID class is only a validator now to avoid issues with `h5netcdf`

## 0.3.1 (2021-02-15)

- Added function to load settings from previous experiments onto instruments (load_settings_onto_instrument).
- Added support for @property as attributes of Settables/Gettables.
- Migrated code style to black.
- Fixed support for python3.9.
- Significant improvements to general documentation.
- Improved installation instructions for windows and MacOS.
- Changed the dataset .unit attribute to .units to adopt xarray default (Breaking change!).
- Various minor bugfixes.

### Merged branches and closed issues

- Windows install instr (!79)
- Load instrument settings (!29)
- Docs/general fixes (!82)
- updated copyright years (!84)
- Hotfix makefile docs (!83)
- Hot fix tuids max num (!85)
- added reqs for scipy, fixes #133 (!87)
- Added link on cross-fork collaboration (!90)
- Allow easy access to the tests datadir from a simple import (!95)
- Add custom css for rtd (!27)
- Dset units attr, closes #147 (!101)
- Add setGeometry method to instrument monitor and plotmon (!88)
- Enforce a datadir choice to avoid potential data loss (!86)
- Migrated code style to black (!93)
- Fixed support for python3.9 (!94)
- Added support for dynamic change of datadir for plotmon (!97)
- Added support for @property as attributes of Settables/Gettables (!100)
- "unit" attr of xarray variables in dataset changed to "units" for compatibility with xarray utilities. (!101)
- Updated numpy requirement (!104)
- Updated installation intructions for MacOS #142 (!99)
- Bugfix for get tuids containing method (!106)

### Breaking changes

Please see merge request !101 for a python snippet that will make all previous datasets compliant with this breaking change!

- "unit" attr of xarray variables in dataset changed to "units" for compatibility with xarray utilities. (!101)

## 0.3.0 (2020-12-17)

- Persistence mode feature added to the plotting monitor responsible for visualization during experiments, see also tutorial 4 in the docs.
- Instrument monitor feature added to support live snapshot monitoring during experiments.
- Renaming of \[soft, hard\]-loops to \[iterative, batched\]-loops respectively.
- Adds t_start and t_stop arguments to the function get_tuids_containing in quantify.data.handling.
- Various bug fixes and improvements to documentation.

### Merged branches and closed issues

- Fix for pyqtgraph plotting and instrument monitor remote process sleeping !81.
- Plotting monitor is now running in a completely detached process !78.
- Persistence mode added to the plotting monitor !72.
- Adds explicit numpy version number (==1.19.2) requirement for windows in the setup. (!74).
- Improved documentation on how to set/get the datadirectory #100 (!71)
- Batched refactor. Closes #113 (!69).
- Instrument monitor feature added. Closes #62 (!65).
- Hot-fix for exception handling of gettable/settable in MC. Closes #101 (!64).
- Added t_start and t_stop arguments to get_tuids_containing function within quantify.data.handling. Closes #69 (!57, !62).
- Fix for the case when MC does not call finish on gettable. Closes #96 (!60).

## 0.2.0 (2020-10-16)

- Repository renamed to quantify-core.
- Scheduler functionality factored out into quantify-scheduler repository.

### Merged branches and closed issues

- !11 Advanced MC, closed #13.
- First prototype of sequencer #16 (!13), moved to quantify-scheduler.
- Documentation of sequencer datatypes #19 (!13), moved to quantify-scheduler.
- Simplified settable gettable interface #32 (!15).
- Keyboard interrupt handler for Measurement Control #20 (!12).
- Documentation for gettable and settable #27 (!14).
- Sequencer hardening and cleanup (!16), moved to quantify-scheduler.
- CZ doc updates and rudimentary CZ implementation (!18), moved to quantify-scheduler.
- Pulsar asm backend (!17), moved to quantify-scheduler.
- Minor fixes sequencer (!19), moved to quantify-scheduler.
- Utility function to get_tuids_containing #48 (!22).
- Enable modulation bugfix #42 (!23), moved to quantify-scheduler.
- Added copyright notices to source files #36 (!25).
- Custom readthedocs theme to change column width, fixes #28 (!27).
- Amplitude limit on waveforms #41 (!24), moved to quantify-scheduler.
- Pulse diagram autoscaling bufix #49 (!26), moved to quantify-scheduler.
- Implementation of adaptive measurement loops in the measurement control #24 (!21)
- Load instrument settings utility function #21, !29.
- Support for data acquisition in sequencer (!28), moved to quantify-scheduler.
- Documentation for data storage, experiment containers and dataset #7 (!20).
- Function to create a plot monitor from historical data #56 (!32).
- Bugfix for buffersize in dynamically resized dataset (!35).
- Bugfix for adaptive experiments with n return variables (!34)
- Exteneded sequencer.rst tutorial to include QRM examples (!33), moved to quantify-scheduler.
- Refactor, Moved quantify-scheduler to new repository (!37).
- Gettable return variables made consistent for multiple gettables #68 (!38).
- Contribution guidelines updated #53 (!31).
- Bugfix for unexpected behaviour in keyboard interrupt for measurements #73 (!39)
- Documentation improvements #71 (!40).
- Improvements to tutorial !41.
- Removed visualization for scheduler !43.
- Fix broken links in install and contributions !44.
- Fixes bug in TUID validator #75 (42).
- Standardize use of numpydoc accross repo #67 (!46).
- Fix for online build on readthedocs !47.
- CI hardening, base python version for tests is 3.7 (minimum version) !50.
- New data folder structure (Breaking change!) #76 (!48).
- Updated installation guide #77 (!49).
- Minor changes to RTD displaying issues (!51).
- Convert jupyter notebooks to .rst files with jupyter-execute (!52).
- Cleanup before opening repo #86 and #82 (!53)

## 0.1.1 (2020-05-25)

- Hotfix to update package label and fix PyPI

## 0.1.0 (2020-05-21)

- First release on PyPI.

---

🗈 **Note**

- \#  denotes a closed issue.
- ! denotes a merge request.

