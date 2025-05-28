# Release Notes

## Release v0.8.2

This release comes with a few minor improvements and bug fixes.

- the package metadata now includes compatibility with python 3.11 and 3.12
- the `ReadoutCalibrationAnalysis` would sometimes assign the wrong state to a measured qubit. This is now fixed.
- the `LorentzianModel` can now fit descending peaks which is useful for certain spin qubit experiments.
- the `TUID` timestamps are now processed 8 times faster

For more details, see the {ref}`complete changelog <changelog>`.


## Release v0.8.1

This release comes with a small fix to make quantify-core compatible with quantify-scheduler v0.22.3, that introduced the `WeightedThresholdedAcquisition` operation.

For more details, see the {ref}`complete changelog <changelog>`.


## Release v0.7.8

This release comes with a small fix to make quantify-core compatible with python 3.12 after the distutils package was removed.

## Release v0.7.7

This release comes with a few minor improvements and bug fixes.

For more details, see the {ref}`complete changelog <changelog>`.


    