# -----------------------------------------------------------------------------
# Description:    Module containing the core data concepts of quantify.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import datetime


class TUID(str):
    """
    A human readable unique identifier based on the timestamp.

    A tuid is a string formatted as ``YYYYmmDD-HHMMSS-sss-******``.
    The tuid serves as a unique identifier for experiments in quantify see also :mod:`~quantify.data.handling`.
    """

    def __init__(self, value):
        self.is_valid(value)

    def datetime(self):
        """
        Returns
        -------
        :class:`~python:datetime.datetime`
            object corresponding to the TUID
        """
        return datetime.datetime.strptime(self[:18], '%Y%m%d-%H%M%S-%f')

    def uuid(self):
        """
        Returns
        -------
        str
            the uuid (universally unique identifier) component of the TUID, corresponding to the last 6 characters.
        """
        return self[20:]

    @classmethod
    def is_valid(cls, tuid):
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
