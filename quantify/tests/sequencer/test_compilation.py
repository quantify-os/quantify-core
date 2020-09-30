import pytest
import numpy as np
import json
from quantify.scheduler import Schedule
from quantify.scheduler.gate_library import Reset, Measure, CNOT, Rxy, CZ
from quantify.scheduler.pulse_library import SquarePulse
from quantify.scheduler.compilation import _determine_absolute_timing, validate_config, _add_pulse_information_transmon, qcompile
from quantify.scheduler.resources import QubitResource, PortResource
from quantify.scheduler.types import Operation, Resource


import pathlib
cfg_f = pathlib.Path(__file__).parent.parent.absolute() / 'test_data' / 'transmon_test_config.json'
with open(cfg_f, 'r') as f:
    DEVICE_TEST_CFG = json.load(f)


def test__determine_absolute_timing_ideal_clock():
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

    timed_sched = _determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4]

    # add a pulse and schedule simultaneous with the second pulse
    sched.add(Rxy(90, 0, qubit=q1), ref_pt='start', ref_op=ref_label_1)
    timed_sched = _determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='start', ref_op='M0')
    timed_sched = _determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='end', ref_op=ref_label_1)
    timed_sched = _determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='center', ref_op=ref_label_1)
    timed_sched = _determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2, 1.5]

    bad_sched = Schedule('no good')
    bad_sched.add(Rxy(180, 0, qubit=q1))
    bad_sched.add(Rxy(90, 0, qubit=q1), ref_pt='bad')
    with pytest.raises(NotImplementedError):
        _determine_absolute_timing(bad_sched)


def test_missing_ref_op():
    sched = Schedule('test')
    q0, q1 = ('q0', 'q1')
    ref_label_1 = 'test_label'
    with pytest.raises(ValueError):
        sched.add(operation=CNOT(qC=q0, qT=q1), ref_op=ref_label_1)


def test_config_spec():
    validate_config(DEVICE_TEST_CFG, scheme_fn='transmon_cfg.json')


def test_compile_transmon_program():
    sched = Schedule('Test schedule')

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ('q0', 'q1')
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')
    # pulse information is added
    sched = _add_pulse_information_transmon(sched, device_cfg=DEVICE_TEST_CFG)
    sched = _determine_absolute_timing(sched, clock_unit='physical')


def test_missing_edge():
    sched = Schedule('Bad edge')
    bad_cfg = DEVICE_TEST_CFG.copy()
    del bad_cfg['edges']['q0-q1']

    q0, q1 = ('q0', 'q1')
    sched.add(operation=CZ(qC=q0, qT=q1))
    with pytest.raises(ValueError, match="Attempting operation 'CZ' on qubits q1 and q0 which lack a connective edge."):
        _add_pulse_information_transmon(sched, device_cfg=bad_cfg)


def test_empty_sched():
    sched = Schedule('empty')
    with pytest.raises(ValueError, match="schedule 'empty' contains no operations"):
        _determine_absolute_timing(sched)


def test_bad_gate():
    class NotAGate(Operation):
        def __init__(self, q):
            data = {'gate_info': {'unitary': np.array([[1, 1, 1, 1],
                                                       [1, 1, 1, 1],
                                                       [1, 1, 1, 1],
                                                       [1, 1, 1, 1]]),
                                  'tex': r'bad',
                                  'plot_func': 'quantify.visualization.circuit_diagram.cnot',
                                  'qubits': q,
                                  'operation_type': 'bad'}}
            super().__init__('bad ({})'.format(q), data=data)

    min_config = {
        "qubits": {
            "q0": {"init_duration": 250e-6}
        },
        "edges": {}
    }

    sched = Schedule('Bell experiment')
    q0 = QubitResource('q0')
    sched.add_resource(q0)
    sched.add(Reset(q0.name))
    sched.add(NotAGate(q0.name))
    with pytest.raises(NotImplementedError, match='Operation type "bad" not supported by backend'):
        _add_pulse_information_transmon(sched, min_config)


def test_resource_resolution():
    # i think ultimately explicit resources for qubits/ports is no bueno, just adds an extra layer with no purpose
    # addressing scheme of q0:mw:qcm0:s0 makes much more sense
    # resolve leftwards

    # walk_address -> return the right most

    sched = Schedule('resource_resolution')
    q0 = QubitResource('q0')
    q0_mw = PortResource('q0:mw_ch')
    qcm0_s0 = Resource({'name': 'qcm0.s0', 'type': 'qcm'})

    sched.add(Rxy(90, 0, q0))
    sched.add(SquarePulse(0.8, 20e-9, q0_mw))
    sched.add(SquarePulse(0.6, 20e-9, 'q0:mw_ch'))

    sched.add_resources([q0, q0_mw, qcm0_s0])
    sched = qcompile(sched, DEVICE_TEST_CFG)

    print(sched)
