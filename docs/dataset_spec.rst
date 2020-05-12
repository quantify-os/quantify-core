
==========================
DataStorage specification
==========================

The quantify dataset is based on ideas from PycQED, QCoDeS and xarray.

xarray dataset

- Dataset -> arrays + attrs
- metadata -> snapshot etc.
.. _dataset-spec:

Introduction
============

The DataSet class is used in QCoDeS to hold measurement results.
It is the destination for measurement loops and the source for plotting and data analysis.
As such, it is a central component of QCoDeS.

The DataSet class should be usable on its own, without other QCoDeS components.
In particular, the DataSet class should not require the use of Loop and parameters, although it should integrate with those components seamlessly.
This will significantly improve the modularity of QCoDeS by allowing users to plug into and extend the package in many different ways.
As long as a DataSet is used for data storage, users can freely select the QCoDeS components they want to use.

Terminology
================

.. warning::

    Below is copied from the QCoDeS data spec. it needs to be updated for quantify.

Metadata
    Many items in this spec have metadata associated with them.
    In all cases, we expect metadata to be represented as a dictionary with string keys.
    While the values are arbitrary and up to the user, in many cases we expect metadata to be nested, string-keyed dictionaries
    with scalars (strings or numbers) as the final values.
    In some cases, we specify particular keys or paths in the metadata that other QCoDeS components may rely on.

Parameter
    A logically-single value input to or produced by a measurement.
    A parameter need not be a scalar, but can be an array or a tuple or an array of tuples, etc.
    A DataSet parameter corresponds conceptually to a QCoDeS parameter, but does not have to be defined by or associated with a QCoDeS Parameter .
    Roughly, a parameter represents a column in a table of experimental data.

Result
    A result is the collection of parameter values associated to a single measurement in an experiment.
    Roughly, a result corresponds to a row in a table of experimental data.

DataSet
    A DataSet is a QCoDeS object that stores the results of an experiment.
    Roughly, a DataSet corresponds to a table of experimental data, along with metadata that describes the data.
    Depending on the state of the experiment, a DataSet may be "in progress" or "completed".

ExperimentContainer
    An ExperimentContainer is a QCoDeS object that stores all information about an experiment.
    This includes items such as the equipment on which the experiment was run, the configuration of the equipment, graphs and other analytical output, and arbitrary notes, as well as the DataSet that holds the results of the experiment.
