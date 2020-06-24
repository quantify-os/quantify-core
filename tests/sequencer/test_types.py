import pytest
import numpy as np
from quantify.sequencer import Schedule, Operation, Resource
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy, X, X90, Y, Y90, CZ
from quantify.sequencer.resources import QubitResource, CompositeResource, Pulsar_QCM_sequencer

def test_schedule_Bell():
    # Create an empty schedule
    sched = Schedule('Bell experiment')
    assert Schedule.is_valid(sched)

    assert len(sched.data['operation_dict']) == 0
    assert len(sched.data['timing_constraints']) == 0

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ('q0', 'q1')

    # Define the operations, these will be added to the circuit
    init_all = Reset(q0, q1)  # instantiates
    x90_q0 = Rxy(theta=90, phi=0, qubit=q0)

    # we use a regular for loop as we have to unroll the changing theta variable here
    for theta in np.linspace(0, 360, 21):
        sched.add(init_all)
        sched.add(x90_q0)
        sched.add(operation=CNOT(qC=q0, qT=q1))
        sched.add(Rxy(theta=theta, phi=0, qubit=q0))
        sched.add(Measure(q0, q1), label='M {:.2f} deg'.format(theta))

    assert len(sched.operations) == 24
    assert len(sched.timing_constraints) == 105

    assert Schedule.is_valid(sched)


def test_schedule_add_timing_constraints():
    sched = Schedule('my exp')
    test_lab = 'test label'
    x90_label = sched.add(Rxy(theta=90, phi=0, qubit='q0'), label=test_lab)
    assert x90_label == test_lab

    with pytest.raises(ValueError):
        x90_label = sched.add(Rxy(theta=90, phi=0, qubit='q0'), label=test_lab)

    uuid_label = sched.add(Rxy(theta=90, phi=0, qubit='q0'))
    assert uuid_label != x90_label

    # not specifying a label should work
    sched.add(Rxy(theta=90, phi=0, qubit='q0'), ref_op=None)

    # specifying existing label should work
    sched.add(Rxy(theta=90, phi=0, qubit='q0'), ref_op=x90_label)

    # specifying non-existing label should raise an error
    with pytest.raises(ValueError):
        sched.add(Rxy(theta=90, phi=0, qubit='q0'), ref_op='non-existing-operation')


    assert Schedule.is_valid(sched)

def test_valid_resources():

    q0 = QubitResource('q0')
    assert Resource.is_valid(q0)

    s0 = Pulsar_QCM_sequencer(name='s0', instrument_name='qcm1', seq_idx=0)
    s1 = Pulsar_QCM_sequencer(name='s1', instrument_name='qcm1', seq_idx=1)
    assert Resource.is_valid(s0)
    assert Resource.is_valid(s1)

    with pytest.raises(TypeError):
        qcm1 = CompositeResource('qcm1', [s0, s1])

    qcm1 = CompositeResource('qcm1', [s0.name, s1.name])
    assert Resource.is_valid(qcm1)



def test_gates_valid():
    init_all = Reset('q0', 'q1')  # instantiates
    x90_q0 = Rxy(theta=124, phi=23.9, qubit='q5')
    x = X('q0')
    x90 = X90('q1')
    y = Y('q0')
    y90 = Y90('q1')

    cz = CZ('q0', 'q1')
    cnot = CNOT('q0', 'q6')

    measure = Measure('q0', 'q9')

    assert Operation.is_valid(init_all)
    assert Operation.is_valid(x90_q0)
    assert Operation.is_valid(x)
    assert Operation.is_valid(x90)
    assert Operation.is_valid(y)
    assert Operation.is_valid(y90)
    assert Operation.is_valid(cz)
    assert Operation.is_valid(cnot)
    assert Operation.is_valid(measure)



def test_schedule_add_resource():

    q0 = QubitResource('q0')
    assert Resource.is_valid(q0)
    s0 = Pulsar_QCM_sequencer(name='s0', instrument_name='qcm1', seq_idx=0)

    sched = Schedule('my exp')
    sched.add_resource(q0)
    set(sched.resources.keys()) == {'q0'}





