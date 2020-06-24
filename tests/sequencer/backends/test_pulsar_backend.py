import pathlib
from quantify.sequencer.backends.pulsar_backend import prepare_waveforms_for_q1asm, construct_q1asm_pulse_operations
from quantify.sequencer.pulse_library import SquarePulse, DRAGPulse


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
        (4, DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=4, freq_mod=15e6, ch_I='ch1', ch_Q='ch2')),
        (16, SquarePulse(amp=2.0, duration=4, ch='ch1')),
    ]

    pulse_data = {
        pulse_timings[0][1].hash: {'data': [0, 1, 0], 'index': 0},
        pulse_timings[1][1].hash: {'data': [-1, 0, -1], 'index': 1},
        pulse_timings[2][1].hash: {'data': [-1, 1, -1], 'index': 2}
    }

    program_str = construct_q1asm_pulse_operations(pulse_timings, pulse_data)
    with open(pathlib.Path(__file__).parent.joinpath('ref_test_construct_q1asm_pulse_operations.q1asm'), 'rb') as f:
        assert program_str.encode('utf-8') == f.read()

