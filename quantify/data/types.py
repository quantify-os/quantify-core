# -----------------------------------------------------------------------------
# Description:    Module containing the core data concepts of quantify.
# Repository:     https://gitlab.com/qblox/packages/software/quantify/
# Copyright (C) Qblox BV (2020)
# -----------------------------------------------------------------------------
import datetime


class TUID(str):
    """
    A human readable unique identifier based on the timestamp.

    A tuid is a string formatted as ``YYYYMMDD-HHMMSS-fff-******``.
    The tuid serves as a unique identifier for experiments in quantify see also :mod:`~quantify.data.handling`.
    """

    def __init__(self, value):
        self.is_valid(value)

    def datetime(self):
        """
        Returns:
            :class:`~python:datetime.datetime`: object corresponding to the TUID.
        """
        return datetime.datetime.strptime(self[:18], '%Y%m%d-%H%M%S-%f')

    def uuid(self):
        """
        Returns:
            str: the uuid (universally unique identifier) component of the TUID,
            corresponding to the last 6 characters.
        """
        return self[20:]

    @classmethod
    def is_valid(cls, tuid):
        """
        Test if tuid is valid.

        Args:
            tuid (str): a tuid string

        Returns:
            bool: True if the string is a valid TUID.

        Raises:
            ValueError: Invalid format

        A valid tuid is a string formatted like ``YYYYMMDD-HHMMSS-fff-******``.
        """

        if not tuid[:8].isdigit():
            raise ValueError('Invalid timespec {}'.format(tuid))
        if not tuid[9:15].isdigit():
            raise ValueError('Invalid timespec {}'.format(tuid))
        if not tuid[16:18].isdigit():
            raise ValueError('Invalid timespec {}'.format(tuid))
        if not tuid[8] == '-' and not tuid[15] == '-' and not tuid[19] == '-':
            raise ValueError('Invalid timespec {}'.format(tuid))
        if not len(tuid) == 26:
            raise ValueError('Invalid uuid {}'.format(tuid))

        cls.datetime(tuid)  # verify date format

        return True
