"""
Module containing the core concepts of the sequencer.
"""
from uuid import uuid4
from collections import UserDict
from quantify.utilities.general import make_hash


class Schedule(UserDict):
    """
    A collection of :class:`Operation` objects and timing contraints
    that define relations between the operations.

    The Schedule data structure is based on a dictionary.
    This dictionary contains:

        operation_dict     : a hash table containing the unique :class:`Operation` s added to the schedule.
        timing_constraints : a list of all timing constraints added between operations.
        resource_dict      : a dictionary containing the relevant :class:Resource` s

    """

    def __init__(self, name: str, data: dict = None):
        """
        Args:
            name (str) : name of the schedule
            data (dict): a dictionary containing a pre-existing schedule.
        """

        # valiate the input data to ensure it is valid schedule data
        super().__init__()

        # ensure keys exist
        self.data['operation_dict'] = {}
        self.data['timing_constraints'] = []
        self.data['resource_dict'] = {}
        self.data['name'] = 'nameless'

        if name is not None:
            self.data['name'] = name

        if data is not None:
            raise NotImplementedError

    @property
    def operations(self):
        """
        Operation dictionary
        """
        return self.data['operation_dict']

    @property
    def timing_constraints(self):
        """
        Timing constraints
        """
        return self.data['timing_constraints']

    def __repr__(self):
        return 'Schedule "{}" containing ({}) {}  (unique) operations.'.format(
            self.data['name'],
            len(self.data['operation_dict']), len(self.data['timing_constraints']))

    @classmethod
    def is_valid(cls, schedule) -> bool:
        # NOT IMPLEMENTED

        return True

    def add(self, operation, rel_time: float = 0,
            ref_op: str = None,
            ref_pt: str = 'end',
            ref_pt_new: str = 'start',
            label: str = None) -> str:
        """
        Add an Operation to the schedule and specify timing constraints.

        Args:
            operation (:class:`Operation`): The operation to add to the schedule
            rel_time (float) : relative time between the the reference operation and added operation.
            ref_op (str) : specifies the reference operation.
            ref_pt ('start', 'center', 'end') : reference point in reference operation.
            ref_pt_new ('start', 'center', 'end') : reference point in added operation.
            label  (str) : a label that can be used as an identifier when adding more operations.

        Returns:
            label (str): returns the unique identifier of the last added operation.

        """
        assert isinstance(operation, Operation)

        operation_hash = operation.hash

        if label is None:
            label = str(uuid4())

        # assert that the label of the operation does not exists in the
        # timing constraints.
        label_is_unique = len([item for item in self.data['timing_constraints']
                               if item['label'] == label]) == 0
        if not label_is_unique:
            raise ValueError('label "{}" must be unique'.format(label))

        # assert that the reference operation exists
        if ref_op is not None:
            ref_exists = len([item for item in self.data['timing_constraints']
                              if item['label'] == ref_op]) == 1
            if not ref_exists:
                raise ValueError(
                    'Reference "{}" does not exist in schedule.'.format(ref_op))

        self.data['operation_dict'][operation_hash] = operation
        timing_constr = {'label': label,
                         'rel_time': rel_time,
                         'ref_op': ref_op,
                         'ref_pt_new': ref_pt_new,
                         'ref_pt': ref_pt,
                         'operation_hash': operation_hash}

        self.data['timing_constraints'].append(timing_constr)

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
        if data is not None:
            self.data.update(data)

    @property
    def hash(self):
        """
        A hash based on the contents of the Operation.
        """
        return make_hash(self.data)


class Resource(UserDict):
    """
    A resource corresponds to a physical resource such as an AWG channel,
    a qubit, or a classical register.

    .. warning::

        The data types and interface of a Resource are not defined yet.

    """

    pass
