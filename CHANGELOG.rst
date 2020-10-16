===========
Changelog
===========

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