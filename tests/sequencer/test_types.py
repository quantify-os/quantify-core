import numpy as np
from quantify.sequencer import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy


def test_schedule_Bell():
    # Create an empty schedule
    sched = Schedule('Bell experiment')

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

    assert len(sched.data['operation_dict']) == 24
    assert len(sched.data['timing_constraints']) == 105
