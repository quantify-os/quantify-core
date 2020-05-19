"""
Module containing the core concepts of the sequencer.
"""
from collections import UserDict
from quantify.utilities.general import make_hash


class Schedule(UserDict):

    def __init__(self, name: str, data: dict = None):
        """
        A collection of :class:`Operation` objects and timing contraints
        that define relations between the operations.

        Args:
            name (str) : name of the schedule
            data (dict): a dictionary containing a pre-existing schedule.

        The Schedule data structure is based on a dictionary.
        This dictionary contains:

            operation_dict     :  a hash table containing the unique :class:`Operation` s added to the schedule.
            timing_constraints : a list of all timing constraints added between operations.


        """

        # valiate the input data to ensure it is valid schedule data
        super().__init__()

        # ensure keys exist
        self.data['operation_dict'] = {}
        self.data['timing_constraints'] = {}

        if name is not None:
            self.data['name'] = name

    def __repr__(self):
        return 'Shedule containing ({}) {}  (unique) operations.'.format(
            len(self.data['operation_dict']), len(self.data['timing_constraints']))

    @classmethod
    def is_valid(cls, schedule)->bool:
        # NOT IMPLEMENTED

        return True

    def add_operation(operation, time=0,
                      ref_op='last', ref_pt='end', label='auto') -> str:
        """
        """
        assert isinstance(operation, Operation)

        #

        return label

    pass



class Operation(UserDict):
    """
    A JSON compatible data structure that contains information on
    how to represent the operation on the Gate, Pulse and/or Logical level.
    It also contains information on the :class:`Resource` s used.

    An operation always has the following attributes

    - duration  (float) : duration of the operation in seconds (can be 0)
    - hash      (str)   : an auto generated unique identifier.
    - name      (str)   : a readable identifer, does not have to be unique

    An Operation can contain information  on several levels of abstraction.
    This information is used when different representations. Note that when
    initializing an operation  not all of this information needs to be available
    as operations are typically modified during the compilation steps.


    TODO: converge on exactly what information is required in these specs
    - gate_info (dict): This typically contains:
            - A unitary matrix describing the operation.
            - The target qubit(s), [the resource(s)].
            - Optional Latex code?

    - pulse_info (dict): This typically contains:
            - A function to generate the waveform
            - the arguments for that function
            - Numerical waveforms?
            - The AWG channels used [the resource(s)].
            - TODO: -> this spec needs to be defined, will take inspiration
            from the qiskit OpenPulse spec, QuPulse and some spefic
            ideas discussed with Martin.

    - logic_info (dict): This typically contains:

            .. warning::

                The instruction/logical information level is not clearly
                defined yet.


    .. note::

        Two different Operations containing the same information generate the
        same hash and are considered identical.


    """

    def __init__(self, name: str, data: dict = None):
        super().__init__()

        # ensure keys exist
        self.data['gate_info'] = {}
        self.data['pulse_info'] = {}
        self.data['logic_info'] = {}

        if name is not None:
            self.data['name'] = name

    @property
    def hash(self):
        """
        A hash based on the contents of the Operation.
        """
        return make_hash(self.data)


class Resource():
    """
    A resource corresponds to a physical resource such as an AWG channel,
    a qubit, or a classical processor for e.g., feedback as a function of time.

    .. warning::

        The data types and interface of a Resource are not defined yet.
    """

    pass
