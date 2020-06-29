import json
import pathlib
import numpy as np
from quantify.sequencer.types import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy
from quantify.sequencer.backends.pulsar_backend import prepare_waveforms_for_q1asm, construct_q1asm_pulse_operations, generate_sequencer_cfg
from quantify.sequencer.pulse_library import SquarePulse, DRAGPulse
from quantify.sequencer.resources import QubitResource, CompositeResource, Pulsar_QCM_sequencer


try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from ... import test_data  # relative-import the *package* containing the templates


DEVICE_TEST_CFG = json.loads(pkg_resources.read_text(
    test_data, 'transmon_test_config.json'))


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


    # current description:
    pulse_data = {
        pulse_timings[0][1].hash: {'data': [0, 1, 0], 'index': 0},
        pulse_timings[1][1].hash: {'data': [-1, 0, -1], 'index': 1},
        pulse_timings[2][1].hash: {'data': [-1, 1, -1], 'index': 2}
    }
    # intended description:

    # pulse_data_us = {'pulse_id': {'data': np.array (complex)}


    # pulse_data_hardware = {'pulse_id_I': {'data': np.array, 'index': int,
    #                        'pulse_id_Q': {'data': np.array, 'index': int}





    program_str = construct_q1asm_pulse_operations(pulse_timings, pulse_data)
    with open(pathlib.Path(__file__).parent.joinpath('ref_test_construct_q1asm_pulse_operations'), 'rb') as f:
        assert program_str.encode('utf-8') == f.read()


def test_generate_sequencer_cfg():
    # Pulse timings should already contain the tuple of wf + pulse_ID
    pulse_timings = [
        (0, SquarePulse(amp=1.0, duration=4, ch='ch1')),
        (4, DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=4, freq_mod=15e6, ch='ch1')),
        (16, SquarePulse(amp=2.0, duration=4, ch='ch1')),
    ]

    # pulse_timings hash property is not guaranteed to be unique as it i
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


