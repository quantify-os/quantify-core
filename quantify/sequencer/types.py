"""
Module containing the core concepts of the sequencer.
"""
from collections import UserDict


class Schedule():
    """
    A collection of :class:`Operation` objects and timing contraints
    that define relations between the operations.

    Contains:
        a set of the operations used.
        a schedule
    """

    def __init__(self, name=''):
        self.name = name

    def add_operation(operation, time=0,
                      ref_op='last', ref_pt='end', label='auto') -> str:
        """
        Add an operation to the schedule.

        """
        if label == 'auto':



        return label


class Operation():
    """
    A is a JSON compatible data structure that contains information on
    how to represent the operation on the Gate, Pulse and/or Logical level.
    It also contains information on the :class:`Resource`s used.

    An operation always has the following attributes

    - duration  (float) : duration of the operation in seconds (can be 0)
    - hash      (str)   : an auto generated unique identifier.
    - name      (str)   : a readable identifer, does not have to be unique

    An operation can contain

    - Gate information (dict): This typically contains:
            - A unitary matrix describing the operation.
            - The target qubit(s), [the resource(s)].
            - Optional Latex code?

    - Pulse information (dict): This typically contains:
            - A function to generate the waveform
            - the arguments for that function
            - Numerical waveforms?
            - The AWG channels used [the resource(s)].
            - TODO: -> this spec needs to be defined, will take inspiration
            from the qiskit OpenPulse spec, QuPulse and some spefic
            ideas discussed with Martin.

    - Logical information (dict): This typically contains:

            .. warning::

                The instruction/logical information level is not clearly
                defined yet.


    .. note::

        Two different Operations containing the same information generate the
        same hash and are considered identical.


    """
    pass


class Resource():
    """
    A resource corresponds to a physical resource such as an AWG channel,
    a qubit, or a classical processor for e.g., feedback as a function of time.

    .. warning::

        The data types and interface of a Resource are not defined yet.
    """

    pass
