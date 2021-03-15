===========
Changelog
===========

0.3.2 (2021-03-16)
------------------

Overview
^^^^^^^^

* Improved dataset format (breaking changes! see notes below)
* New features
* Various bug fixes
* Improved and more extensive documentation
* Analysis framework (Undocumented)

Breaking changes
^^^^^^^^^^^^^^^^

Please see merge request !98 for a python snippet that will make all previous datasets compliant with this breaking change!

* Dataset format changed to use Xarray Coordinates [no Dimensions] (!98)
* The TUID class is only a validator now to avoid issues with `h5netcdf`

New
^^^

* Data handling utilities: `load_dataset_from_path`, `locate_experiment_container`, `write_dataset`
* Support of complex numbers and arrays in the dataset through `h5netcdf` engine
* Batched mode with outer iterative loops

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Dataset format changed to use Xarray Coordinates [no Dimensions] (!98)
* Added batched mode with outer iterative loops (!98)
* Switched default dataset engine to support complex numbers #150 (!114)
* Analysis class, framework, subclass examples #63 (!89, !122, !123)
* Cyclic colormaps auto-detect in 2D analysis (!118, !122)

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
^^^^^^^^^^^^^^^^

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
