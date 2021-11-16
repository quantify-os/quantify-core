# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Module containing the core data concepts of quantify."""
from __future__ import annotations

import datetime
from typing import Type, cast


class TUID(str):
    """
    A human readable unique identifier based on the timestamp.
    This class does not wrap the passed in object but simply verifies and returns it.

    A tuid is a string formatted as ``YYYYmmDD-HHMMSS-sss-******``.
    The tuid serves as a unique identifier for experiments in quantify.

    .. seealso:: The. :mod:`~quantify_core.data.handling` module.
    """

    __slots__ = ()  # avoid unnecessary overheads

    def __new__(cls: Type[TUID], value: str) -> TUID:
        assert cls.is_valid(value)
        # NB instead of creating an instance of this class we just return the object
        # This avoids nasty type conversion issues when saving a dataset using the
        # `h5netcdf` engine (which we need to support complex numbers)
        return cast(TUID, value)

    @classmethod
    def datetime(cls, tuid: str) -> datetime.datetime:
        """
        Returns
        -------
        :class:`~python:datetime.datetime`
            object corresponding to the TUID
        """
        return datetime.datetime.strptime(tuid[:18], "%Y%m%d-%H%M%S-%f")

    @classmethod
    def uuid(cls, tuid: str) -> str:
        """
        Returns
        -------
        str
            the uuid (universally unique identifier) component of the TUID,
            corresponding to the last 6 characters.
        """
        return tuid[20:]

    @classmethod
    def is_valid(cls, tuid: str) -> bool:
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

        uid = cls.uuid(tuid)

        if len(uid) != 6:
            raise ValueError(
                "Invalid format: uid has invalid length {len(uid)} (should be 6)."
            )
        if not uid.isalnum():
            raise ValueError("Invalid format: uid is not alphanumeric.")

        if tuid[8] != "-" or tuid[15] != "-" or tuid[19] != "-":
            raise ValueError(
                f"Invalid TUID format: seperator at positions 8, 15 and 19 should be '-'."
            )

        return True
