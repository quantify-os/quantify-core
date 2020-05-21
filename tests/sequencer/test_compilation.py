import pytest
import numpy as np
from quantify.sequencer import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy
from quantify.sequencer.compilation import determine_absolute_timing


def test_determine_absolute_timing_ideal_clock():

    sched = Schedule('Test experiment')

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ('q0', 'q1')

    ref_label_1 = 'my_label'

    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(operation=CNOT(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')

    assert len(sched.data['operation_dict']) == 4
    assert len(sched.data['timing_constraints']) == 5

    for constr in sched.data['timing_constraints']:
        assert 'abs_time' not in constr.keys()
        assert constr['rel_time'] == 0

    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time']
                 for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4]

    # add a pulse and schedule simultaneous with the second pulse
    sched.add(Rxy(90, 0, qubit=q1), ref_pt='start', ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time']
                 for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='start', ref_op='M0')
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time']
                 for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='end', ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time']
                 for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='center', ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time']
                 for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2, 1.5]
