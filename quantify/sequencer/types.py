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
            ref_pt (str): reference point in reference operation must be one of ('start', 'center', 'end').
            ref_pt_new (str) : reference point in added operation must be one of ('start', 'center', 'end').
            label  (str) : a label that can be used as an identifier when adding more operations.

        Returns:

            (str) : returns the (unique) label of the last added operation.

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


    - gate_info (dict): This can contain the following items:

        - unitary (np.array) : A unitary matrix describing the operation.
        - qubits (list) : A list of string specifying the qubit names.
        - tex (str) : latex snippet for plotting
        - plot_func (str): reference to a function for plotting this operation
          in a circuit diagram. If not specified, defaults to using
          :func:`quantify.visualization.circuit_diagram.gate_box`
          A valid plot_func must accept the following arguments:

            - ax
            - time (float)
            - qubit_idxs (list)
            - tex (str)

    - pulse_info (dict): This can contain:

        - wf_func (str): reference to a function to generate the waveform.
        - keyword arguments for wf_func
        - channels (list) : a list of the channels used.

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
        self.data['gate_info'] = {
            'unitary': None,
            'tex': None,
            'plot_func': None,
            'qubits': None}
        self.data['pulse_info'] = {}
        self.data['logic_info'] = {}
        self.data['duration'] = None  # start as undefined

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

    def add_gate_info(self, gate_operation):
        """
        Updates self.data['gate_info'] with contents of gate_operation.

        Args:
            gate_operation (:class:`Operation`) : an operation containing gate_info.
        """
        self.data['gate_info'].update(gate_operation.data['gate_info'])

    def add_pulse_info(self, pulse_operation):
        """
        Updates self.data['pulse_info'] with contents of pulse_operation.

        Args:
            pulse_operation (:class:`Operation`) : an operation containing pulse_info.
        """
        self.data['pulse_info'].update(pulse_operation.data['pulse_info'])

        # pulse level duration information takes presedence
        self.data['duration'] = pulse_operation.data['duration']


class Resource(UserDict):
    """
    A resource corresponds to a physical resource such as an AWG channel,
    a qubit, or a classical register.

    .. warning::

        The data types and interface of a Resource are not defined yet.

    """

    pass
