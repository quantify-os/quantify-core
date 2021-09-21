# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities for dataset (python object) handling."""
# pylint: disable=too-many-instance-attributes
from __future__ import annotations

import json
from copy import deepcopy
from typing import Callable, Any
from abc import abstractmethod
import xarray as xr


class DatasetAdapter:
    """
    Generic interface for dataset adapters.

    Subclasses implementing this interface are intended to bridge to some specific
    object, function, interface, backend, etc.. We can refer to it as the "Target".

    The function ``.adapt()`` should return a dataset to be consumed by the Target.

    The function ``.recover()`` should receive a dataset generated by the Target.
    """

    @classmethod
    @abstractmethod
    def adapt(cls, dataset: xr.Dataset) -> xr.Dataset:
        """Converts the ``dataset`` to a format consumed by the target."""

    @classmethod
    @abstractmethod
    def recover(cls, dataset: xr.Dataset) -> xr.Dataset:
        """Inverts the action of the ``.adapt()`` method."""


class AdapterH5NetCDF(DatasetAdapter):
    """
    Quantify dataset adapter for the ``h5netcdf`` engine.

    It has the functionality of adapting the Quantify dataset to a format compatible
    with the ``h5netcdf`` xarray backend engine that is used to write the dataset to
    disk.

    .. warning::

        The ``h5netcdf`` engine has issue with two-way trip of the types of the
        attributes values. Namely list- and tuple- like objects are loaded as
        numpy arrays of ``dtype=object``.
    """

    @classmethod
    def adapt(cls, dataset: xr.Dataset) -> xr.Dataset:
        """
        Serializes to JSON designated dataset and variables attributes.

        The designated attributes for which this is performed must be listed inside an
        attribute named ``json_attrs`` (for each ``attrs`` dictionary).

        Parameters
        ----------
        dataset
            Dataset that needs to be adapted.

        Returns
        -------
        :
            Dataset in which the designated attributes have been replaced with their
            JSON string version.
        """

        return cls._transform(dataset, vals_converter=json.dumps)

    @classmethod
    def recover(cls, dataset: xr.Dataset) -> xr.Dataset:
        """
        Reverts the action of ``.adapt()``.

        The designated attributes for which this is performed must be listed inside an
        attribute named ``json_attrs`` (for each ``attrs`` dictionary).

        Parameters
        ----------
        dataset
            Dataset from which to recover the original format.

        Returns
        -------
        :
            Dataset in which the designated attributes have been replaced with their
            python objects version.
        """

        return cls._transform(dataset, vals_converter=json.loads)

    @staticmethod
    def attrs_convert(
        attrs: dict,
        inplace: bool = False,
        vals_converter: Callable[Any, Any] = json.dumps,
    ) -> dict:
        """
        Converts to/from JSON string the values of the keys that are listed in the
        ``json_attrs`` list.

        Parameters
        ----------
        attrs
            The input dictionary.
        inplace
            If ``True`` the value are replaced in place, otherwise a deepcopy of
            ``attrs`` is performed first.
        """
        json_attrs = attrs.get("json_attrs", None)
        # json_attrs might be a numpy array
        if json_attrs is not None and len(json_attrs):
            attrs = attrs if inplace else deepcopy(attrs)
            for attr_name in json_attrs:
                attrs[attr_name] = vals_converter(attrs[attr_name])

        return attrs

    @classmethod
    def _transform(
        cls, dataset: xr.Dataset, vals_converter: Callable[Any, Any] = json.dumps
    ) -> xr.Dataset:
        dataset = xr.Dataset(
            dataset,
            attrs=cls.attrs_convert(
                dataset.attrs, inplace=False, vals_converter=vals_converter
            ),
        )

        for var_name in dataset.variables.keys():
            # The new dataset generated above has already a deepcopy of the attributes.
            _ = cls.attrs_convert(
                dataset[var_name].attrs, inplace=True, vals_converter=vals_converter
            )

        return dataset
