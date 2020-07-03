"""
Module containing the core concepts of the sequencer.
"""
from os import path
from uuid import uuid4
from collections import UserDict
import json
import jsonschema
from quantify.utilities.general import make_hash, without


class Schedule(UserDict):
    """
    A collection of :class:`Operation` objects and timing contraints
    that define relations between the operations.

    The Schedule data structure is based on a dictionary.
    This dictionary contains:

        operation_dict     : a hash table containing the unique :class:`Operation` s added to the schedule.
        timing_constraints : a list of all timing constraints added between operations.


    .. jsonschema:: schemas/schedule.json

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
    def name(self):
        return self.data['name']

    @property
    def operations(self):
        """
        Operation dictionary, keys are the has of operations values are instances of :class:`Operation`.
        """
        return self.data['operation_dict']

    @property
    def timing_constraints(self):
        """
        A list of dictionaries describing timing constraints between operations.
        """
        return self.data['timing_constraints']

    @property
    def resources(self):
        """
        A dictionary containing resources. Keys are names (str), values are instances of :class:`Resource` .
        """

        return self.data['resource_dict']

    def add_resources(self, resources: list):
        for r in resources:
            self.add_resource(r)

    def add_resource(self, resource):
        """
        Add a resource such as a channel or qubit to the schedule.
        """
        assert Resource.is_valid(resource)
        self.data['resource_dict'][resource.name] = resource

    def __repr__(self):
        return 'Schedule "{}" containing ({}) {}  (unique) operations.'.format(
            self.data['name'],
            len(self.data['operation_dict']), len(self.data['timing_constraints']))

    @classmethod
    def is_valid(cls, operation):

        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(basepath,
                                          "schemas", "schedule.json"))
        with open(filepath) as json_file:
            scheme = json.load(json_file)
        jsonschema.validate(operation.data, scheme)
        return True  # if not exception was raised during validation

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

    .. jsonschema:: schemas/operation.json


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
            'tex': '',
            'plot_func': None,
            'qubits': []}
        self.data['pulse_info'] = []  # A list of pulses
        self.data['logic_info'] = {}

        if name is not None:
            self.data['name'] = name
        if data is not None:
            self.data.update(data)

    @property
    def name(self):
        return self.data['name']

    @property
    def duration(self):
        """
        Determine the duration of the operation based on the pulses described in pulse_info.

        If the operation contains no pulse info, it is assumed to be ideal and
        have zero duration.
        """
        duration = 0  # default to zero duration if no pulse content is specified.

        # Iterate over all pulses and take longest duration
        for p in self.data['pulse_info']:
            d = p['duration']+p['t0']
            if d > duration:
                duration = d

        return duration

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

    def add_filter(self, window):
        """
        Adds a filter to all pulses in this operation.

        Args:
            window (object): Window to filter with.  # todo improve
        """
        for pulse in self.data['pulse_info']:
            pulse['filter'] = window

    def add_pulse(self, pulse_operation):
        """
        Adds pulse_info of pulse_operation to self.

        Args:
            pulse_operation (:class:`Operation`)    : an operation containing pulse_info.
        """
        self.data['pulse_info'] += pulse_operation.data['pulse_info']

    @classmethod
    def is_valid(cls, operation):

        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(basepath,
                                          "schemas", "operation.json"))
        with open(filepath) as json_file:
            scheme = json.load(json_file)
        jsonschema.validate(operation.data, scheme)
        return True  # if not exception was raised during validation


class Resource(UserDict):
    """
    A resource corresponds to a physical resource such as an AWG channel,
    a qubit, or a classical register.


    .. jsonschema:: schemas/resource.json

    """

    @classmethod
    def is_valid(cls, operation):

        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(basepath,
                                          "schemas", "resource.json"))
        with open(filepath) as json_file:
            scheme = json.load(json_file)
        jsonschema.validate(operation.data, scheme)
        return True  # if not exception was raised during validation

    @property
    def name(self):
        return self.data['name']
