import json
import pathlib
import pytest
import numpy as np
from quantify.sequencer.types import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy
from quantify.sequencer.backends.pulsar_backend import prepare_waveforms_for_q1asm, \
    construct_q1asm_pulse_operations, generate_sequencer_cfg, pulsar_assembler_backend
from quantify.sequencer.pulse_library import SquarePulse, DRAGPulse
from quantify.sequencer.resources import QubitResource, CompositeResource, Pulsar_QCM_sequencer
from quantify.sequencer.compilation import add_pulse_information_transmon, determine_absolute_timing

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

@pytest.mark.skip('none')
def test_construct_q1asm_pulse_operations():

    # Input I want to provide for function 3, contents will change, data types and schema will not.
    # pulse timings example ID's can change
    pulse_timings = [
        (0, 'square_id'),
        (4, 'drag_ID'),
        (16, 'drag_ID5'),
        (20, 'square_id')
    ]

    # new/intended pulse_darta
    # pulse_data = {'pulse_id': np.array (complex),
    #                'square_id': np.ones(20)} # imaginary part is implicitly zero here

    # provided pulse_dict:
    pulse_data = {
        'square_id': np.ones(8),
        # 'drag_ID':   some_np_complex_array,
        'drag_ID5': np.ones(5)}

    # function 1
    # take pulse_data and turn it into the pulse_data required for the json spec (now same name, confusing)
    # function 2
    # loop over the timing tuples, being aware of the pulse_data for hardware config to get indices and produce valid assembly

    # this loop over timing tuples consisint of (t0, pulse_id) needs to specify two waveforms
    pulse_dict_hardware = {
        'pulse_id': 'square_id_I',  'data': np.ones(5), 'index': 0,
        'pulse_id': 'square_id_Q',  'data': np.zeros(5), 'index': 1,
        'pulse_id': 'drag_I',  'data': np.random.rand(5), 'index': 2,
        'pulse_id': 'drag_Q',  'data': np.random.rand(5), 'index': 3, }

    # function 3
    # combine function 1 and 2.

    program_str = construct_q1asm_pulse_operations(pulse_timings, pulse_data)
    # program_str should be a valid JSON containing both the pulse data and the assembly program.

    with open(pathlib.Path(__file__).parent.joinpath('ref_test_construct_q1asm_pulse_operations'), 'rb') as f:
        assert program_str.encode('utf-8') == f.read()


def test_generate_sequencer_cfg():
    # Pulse timings should already contain the tuple of wf + pulse_ID
    pulse_timings = [
        (0, SquarePulse(amp=1.0, duration=4, ch='ch1')),
        (4, DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3,
                      duration=4, freq_mod=15e6, ch='ch1')),
        (16, SquarePulse(amp=2.0, duration=4, ch='ch1')),
    ]

    # pulse_timings hash property is not guaranteed to be unique as it i
    pulse_data = {
        pulse_timings[0][1].hash: [0, 1, 0],
        pulse_timings[1][1].hash: [-1, 0, -1],
        pulse_timings[2][1].hash: [-1, 1, -1],
    }

    sequence_cfg = generate_sequencer_cfg(pulse_data, pulse_timings)
    assert sequence_cfg['waveforms'][pulse_timings[0]
                                     [1].hash] == {'data': [0, 1, 0], 'index': 0}
    assert sequence_cfg['waveforms'][pulse_timings[1]
                                     [1].hash] == {'data': [-1, 0, -1], 'index': 1}
    assert sequence_cfg['waveforms'][pulse_timings[2]
                                     [1].hash] == {'data': [-1, 1, -1], 'index': 2}
    assert len(sequence_cfg['program'])

@pytest.mark.skip('Not Implemented')
def test_pulsar_assembler_backend_missing_pulse_info():
    # should raise an exception
    pass

@pytest.mark.skip('Not Implemented')
def test_pulsar_assembler_backend_missing_timing_info():
    pass


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
        # sched.add(operation=CNOT(qC=q0.name, qT=q1.name))
        sched.add(Rxy(theta=theta, phi=0, qubit=q0.name))
        sched.add(Rxy(theta=90, phi=0, qubit=q1.name))
        sched.add(Measure(q0.name, q1.name),
                  label='M {:.2f} deg'.format(theta))

    # Add the resources for the pulsar qcm channels
    qcm0 = CompositeResource('qcm0', ['qcm0.s0', 'qcm0.s1'])
    qcm0_s0 = Pulsar_QCM_sequencer(
        'qcm0.s0', instrument_name='qcm0', seq_idx=0)
    qcm0_s1 = Pulsar_QCM_sequencer(
        'qcm0.s1', instrument_name='qcm0', seq_idx=1)

    qcm1 = CompositeResource('qcm1', ['qcm1.s0', 'qcm1.s1'])
    qcm1_s0 = Pulsar_QCM_sequencer(
        'qcm1.s0', instrument_name='qcm1', seq_idx=0)
    qcm1_s1 = Pulsar_QCM_sequencer(
        'qcm1.s1', instrument_name='qcm1', seq_idx=1)

    qrm0 = CompositeResource('qrm0', ['qrm0.s0', 'qrm0.s1'])
    # Currently mocking a readout module using an acquisition module
    qrm0_s0 = Pulsar_QCM_sequencer(
        'qrm0.s0', instrument_name='qrm0', seq_idx=0)
    qrm0_s1 = Pulsar_QCM_sequencer(
        'qrm0.s1', instrument_name='qrm0', seq_idx=1)

    # using qcm sequencing modules to fake a readout module
    qrm0_r0 = Pulsar_QCM_sequencer(
        'qrm0.r0', instrument_name='qrm0', seq_idx=0)
    qrm0_r1 = Pulsar_QCM_sequencer(
        'qrm0.r1', instrument_name='qrm0', seq_idx=1)

    sched.add_resources([qcm0, qcm0_s0, qcm0_s1, qcm1, qcm1_s0, qcm1_s1, qrm0, qrm0_s0, qrm0_s1,  qrm0_r0, qrm0_r1])

    sched = add_pulse_information_transmon(sched, DEVICE_TEST_CFG)
    sched = determine_absolute_timing(sched)

    seq_config_dict = pulsar_assembler_backend(sched)

    assert len(sched.resources['qcm0.s0'].timing_tuples) == int(21*2)
    assert len(qcm0_s0.timing_tuples) == int(21*2)
    assert len(qcm0_s1.timing_tuples) == 0
    assert len(qcm1_s0.timing_tuples) == 21
    assert len(qcm1_s1.timing_tuples) == 0

    # assert right keys.
    # assert right content of the config files.
