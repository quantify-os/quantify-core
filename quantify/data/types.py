# -----------------------------------------------------------------------------
# Description:    Module containing the core data concepts of quantify.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
import datetime


class TUID:
    """
    A human readable unique identifier based on the timestamp.
    This class does not wrap the passed in object but simply verifies and returns it.

    A tuid is a string formatted as ``YYYYmmDD-HHMMSS-sss-******``.
    The tuid serves as a unique identifier for experiments in quantify.

    .. seealso:: The. :mod:`~quantify.data.handling` module.
    """

    __slots__ = ()  # avoid unnecessary overheads

    def __new__(cls, value) -> str:
        cls.is_valid(value)
        # NB instead of creating an instance of this class we just return the object
        # This avoids nasty type conversion issues when saving a dataset using the
        # `h5netcdf` engine (which we need to support complex numbers)
        return value

    @classmethod
    def datetime(cls, tuid: str = None):
        """
        Returns
        -------
        :class:`~python:datetime.datetime`
            object corresponding to the TUID
        """
        return datetime.datetime.strptime(tuid[:18], "%Y%m%d-%H%M%S-%f")

    @classmethod
    def uuid(cls, tuid: str):
        """
        Returns
        -------
        str
            the uuid (universally unique identifier) component of the TUID,
            corresponding to the last 6 characters.
        """
        return tuid[20:]

    @classmethod
    def is_valid(cls, tuid: str):
        """
        Test if tuid is valid.
        A valid tuid is a string formatted as ``YYYYmmDD-HHMMSS-sss-******``.

        Parameters
        ----------
        tuid : str
            a tuid string

        Returns
        -------
        bool
            True if the string is a valid TUID.

        Raises
        ------
        ValueError
            Invalid format
        """
        cls.datetime(tuid)  # verify date format
        if len(cls.uuid(tuid)) != 6:
            raise ValueError("Invalid format")

        return True
