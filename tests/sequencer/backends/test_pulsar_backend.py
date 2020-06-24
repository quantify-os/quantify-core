from quantify.sequencer.backends.pulsar_backend import prepare_waveforms_for_q1asm


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
    pass
    # # this will be operation objects
    # pulse_timings = [
    #     (0, SquarePulse(amp=1.0, duration=4, ch='ch1')),
    #     (4, IdlePulse(4)),
    #     (16, SquarePulse(amp=1.0, duration=4, ch='ch1')),
    # ]
