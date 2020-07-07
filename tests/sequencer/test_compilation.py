import pytest
import numpy as np
import json
from quantify.sequencer.backends.visualization import pulse_diagram_plotly
from quantify.sequencer import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy, CZ
from quantify.sequencer.pulse_library import SquarePulse, IdlePulse
from quantify.sequencer.compilation import determine_absolute_timing, validate_config, add_pulse_information_transmon
from quantify.sequencer.resources import QubitResource
from quantify.sequencer.types import Operation


try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from .. import test_data  # relative-import the *package* containing the templates

DEVICE_TEST_CFG = json.loads(pkg_resources.read_text(
    test_data, 'transmon_test_config.json'))


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

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4]

    # add a pulse and schedule simultaneous with the second pulse
    sched.add(Rxy(90, 0, qubit=q1), ref_pt='start', ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='start', ref_op='M0')
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='end', ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt='center', ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, clock_unit='ideal')

    abs_times = [constr['abs_time'] for constr in timed_sched.data['timing_constraints']]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2, 1.5]

    bad_sched = Schedule('no good')
    bad_sched.add(Rxy(180, 0, qubit=q1))
    bad_sched.add(Rxy(90, 0, qubit=q1), ref_pt='bad')
    with pytest.raises(NotImplementedError):
        determine_absolute_timing(bad_sched)


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
    sched.add(operation=CZ(qC=q0, qT=q1)) # not implemented in config
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')
    # pulse information is added
    sched = add_pulse_information_transmon(sched, device_cfg=DEVICE_TEST_CFG)
    sched = determine_absolute_timing(sched, clock_unit='physical')


def test_empty_sched():
    sched = Schedule('empty')
    with pytest.raises(ValueError, match="schedule 'empty' contains no operations"):
        determine_absolute_timing(sched)


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
        add_pulse_information_transmon(sched, min_config)
