import pytest
import numpy as np
from quantify.sequencer.backends import pulse_diagram_plotly
from quantify.sequencer import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy
from quantify.sequencer.compilation import determine_absolute_timing, validate_config, add_pulse_information_transmon


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


device_test_cfg = {
    'qubits':
    {
        'q0': {'mw_amp180': .75, 'mw_motzoi': -.25, 'mw_duration': 20e-9,
               'mw_modulation_freq': 50e6, 'mw_ef_amp180': .87, 'mw_ch_I': 0, 'mw_ch_Q': 1,
               'ro_pulse_ch_I': 5, 'ro_pulse_ch_Q': 6, 'ro_pulse_amp': .5, 'ro_pulse_modulation_freq': 80e6,
               'ro_pulse_type': 'square', 'ro_pulse_duration': 540e-9,
               'ro_acq_ch_I': 7, 'ro_acq_ch_Q': 8, 'ro_acq_delay': 120e-9, 'ro_acq_integration_time': 700e-9,
               'ro_acq_weigth_type': 'SSB',
               'init_duration': 250e-6,
               },

        'q1': {'mw_amp180': .45, 'mw_motzoi': -.15, 'mw_duration': 20e-9,
               'mw_modulation_freq': 20e6, 'mw_ef_amp180': .27, 'mw_ch_I': 2, 'mw_ch_Q': 3,
               'ro_pulse_ch_I': 5, 'ro_pulse_ch_Q': 6, 'ro_pulse_amp': .5, 'ro_pulse_modulation_freq': -23e6,
               'ro_pulse_type': 'square', 'ro_pulse_duration': 540e-9,
               'ro_acq_ch_I': 7, 'ro_acq_ch_Q': 8, 'ro_acq_delay': 120e-9, 'ro_acq_integration_time': 700e-9,
               'ro_acq_weigth_type': 'SSB',
               'init_duration': 250e-6, }
    },
    'edges':
    {
    }
}


def test_config_spec():
    validate_config(device_test_cfg, scheme_fn='transmon_cfg.json')


def test_compile_transmon_program():
    sched = Schedule('Test schedule')

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ('q0', 'q1')
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    # sched.add(operation=CNOT(qC=q0, qT=q1)) # not implemented in config
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')
    # pulse information is added
    sched = add_pulse_information_transmon(sched, device_cfg=device_test_cfg)
    sched = determine_absolute_timing(sched, clock_unit='physical')
