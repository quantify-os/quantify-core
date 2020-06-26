import pathlib
import numpy as np
from quantify.sequencer.types import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy
from quantify.sequencer.backends.pulsar_backend import prepare_waveforms_for_q1asm, construct_q1asm_pulse_operations, generate_sequencer_cfg
from quantify.sequencer.pulse_library import SquarePulse, DRAGPulse
from quantify.sequencer.resources import QubitResource, CompositeResource, Pulsar_QCM_sequencer


def test_prepare_waveforms_for_q1asm():
    pulse_data = {
        'gdfshdg45': [0, 0.2, 0.6],
        '6h5hh5hyj': [-1, 0, 1, 0],
    }
    sequence_cfg = prepare_waveforms_for_q1asm(pulse_data)
    assert len(sequence_cfg['waveforms'])
    wf_1 = sequence_cfg['waveforms']['gdfshdg45']
    wf_2 = sequence_cfg['waveforms']['6h5hh5hyj']
    assert wf_1['data'] == [0, 0.2, 0.6]
    assert wf_1['index'] == 0
    assert wf_2['data'] == [-1, 0, 1, 0]
    assert wf_2['index'] == 1


def test_construct_q1asm_pulse_operations():
    pulse_timings = [
        (0, SquarePulse(amp=1.0, duration=4, ch='ch1')),
        (4, DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=4, freq_mod=15e6, ch='ch1')),
        (16, SquarePulse(amp=2.0, duration=4, ch='ch1')),
    ]

    pulse_data = {
        pulse_timings[0][1].hash: {'data': [0, 1, 0], 'index': 0},
        pulse_timings[1][1].hash: {'data': [-1, 0, -1], 'index': 1},
        pulse_timings[2][1].hash: {'data': [-1, 1, -1], 'index': 2}
    }

    program_str = construct_q1asm_pulse_operations(pulse_timings, pulse_data)
    with open(pathlib.Path(__file__).parent.joinpath('ref_test_construct_q1asm_pulse_operations'), 'rb') as f:
        assert program_str.encode('utf-8') == f.read()


def test_generate_sequencer_cfg():
    pulse_timings = [
        (0, SquarePulse(amp=1.0, duration=4, ch='ch1')),
        (4, DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=4, freq_mod=15e6, ch='ch1')),
        (16, SquarePulse(amp=2.0, duration=4, ch='ch1')),
    ]

    pulse_data = {
        pulse_timings[0][1].hash: [0, 1, 0],
        pulse_timings[1][1].hash: [-1, 0, -1],
        pulse_timings[2][1].hash: [-1, 1, -1],
    }

    sequence_cfg = generate_sequencer_cfg(pulse_data, pulse_timings)
    assert sequence_cfg['waveforms'][pulse_timings[0][1].hash] == {'data': [0, 1, 0], 'index': 0}
    assert sequence_cfg['waveforms'][pulse_timings[1][1].hash] == {'data': [-1, 0, -1], 'index': 1}
    assert sequence_cfg['waveforms'][pulse_timings[2][1].hash] == {'data': [-1, 1, -1], 'index': 2}
    assert len(sequence_cfg['program'])


def test_pulsar_assembler_backend():
    """
    This test uses a full example of compilation for a simple Bell experiment.
    This test can be made simpler the more we clean up the code.
    """
    # Create an empty schedule
    sched = Schedule('Bell experiment')

    # define the resources
    q0, q1 = (QubitResource('q0'), QubitResource('q1'))

    sched.add_resource(q0)
    sched.add_resource(q1)

    # Define the operations, these will be added to the circuit
    init_all = Reset(q0.name, q1.name)  # instantiates
    x90_q0 = Rxy(theta=90, phi=0, qubit=q0.name)

    # we use a regular for loop as we have to unroll the changing theta variable here
    for theta in np.linspace(0, 360, 21):
        sched.add(init_all)
        sched.add(x90_q0)
        sched.add(operation=CNOT(qC=q0.name, qT=q1.name))
        sched.add(Rxy(theta=theta, phi=0, qubit=q0.name))
        sched.add(Measure(q0.name, q1.name),
                  label='M {:.2f} deg'.format(theta))

    # Add the resources for the pulsar qcm channels

    qcm1 = CompositeResource('qcm1', ['qcm1.s0', 'qcm1.s1'])
    qcm1_s0 = Pulsar_QCM_sequencer(
        'qcm1.s0', instrument_name='qcm1', seq_idx=0)
    qcm1_s1 = Pulsar_QCM_sequencer(
        'qcm1.s1', instrument_name='qcm1', seq_idx=1)

    qcm2 = CompositeResource('qcm2', ['qcm2.s0', 'qcm2.s1'])
    qcm2_s0 = Pulsar_QCM_sequencer(
        'qcm2.s0', instrument_name='qcm2', seq_idx=0)
    qcm2_s1 = Pulsar_QCM_sequencer(
        'qcm2.s1', instrument_name='qcm2', seq_idx=1)

    qrm1 = CompositeResource('qrm1', ['qrm1.s0', 'qrm1.s1'])
    # Currently mocking a readout module using an acquisition module
    qrm1_s0 = Pulsar_QCM_sequencer(
        'qrm1.s0', instrument_name='qrm1', seq_idx=0)
    qrm1_s1 = Pulsar_QCM_sequencer(
        'qrm1.s1', instrument_name='qrm1', seq_idx=1)

    sched.add_resources([qcm1, qcm1_s0, qcm1_s1, qcm2, qcm2_s0, qcm2_s1])

    device_test_cfg = {
        'qubits':
        {
            'q0': {'mw_amp180': .75, 'mw_motzoi': -.25, 'mw_duration': 20e-9,
                   'mw_modulation_freq': 50e6, 'mw_ef_amp180': .87, 'mw_ch_I': 'ch0', 'mw_ch_Q': 'ch1',
                   'ro_pulse_ch_I': 'ch5.0', 'ro_pulse_ch_Q': 'ch6.0', 'ro_pulse_amp': .5, 'ro_pulse_modulation_freq': 80e6,
                   'ro_pulse_type': 'square', 'ro_pulse_duration': 150e-9,
                   'ro_acq_ch_I': 'acq_ch1', 'ro_acq_ch_Q': 'acq_ch2', 'ro_acq_delay': 120e-9, 'ro_acq_integration_time': 700e-9,
                   'ro_acq_weigth_type': 'SSB',
                   'init_duration': 250e-6,
                   },

            'q1': {'mw_amp180': .45, 'mw_motzoi': -.15, 'mw_duration': 20e-9,
                   'mw_modulation_freq': 80e6, 'mw_ef_amp180': .27, 'mw_ch_I': 'ch2', 'mw_ch_Q': 'ch3',
                   'ro_pulse_ch_I': 'ch5.1', 'ro_pulse_ch_Q': 'ch6.1', 'ro_pulse_amp': .5, 'ro_pulse_modulation_freq': -23e6,
                   'ro_pulse_type': 'square', 'ro_pulse_duration': 100e-9,
                   'ro_acq_ch_I': 'acq_ch1', 'ro_acq_ch_Q': 'acq_ch2', 'ro_acq_delay': 120e-9, 'ro_acq_integration_time': 700e-9,
                   'ro_acq_weigth_type': 'SSB',
                   'init_duration': 250e-6, }
        },
        'edges':
        {
            'q0-q1': {}  # TODO
        }
    }
