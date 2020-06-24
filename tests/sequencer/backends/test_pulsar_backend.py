

def test_prepare_waveforms_for_q1asm():
    pulse_data = {
        'SquarePulse': [0, 0.2, 0.6],
        'Idle': [-1, 0, 1, 0],
    }


def test_construct_q1asm_pulse_operations():
    pass
    # # this will be operation objects
    # pulse_timings = [
    #     (0, SquarePulse(amp=1.0, duration=4, ch='ch1')),
    #     (4, IdlePulse(4)),
    #     (16, SquarePulse(amp=1.0, duration=4, ch='ch1')),
    # ]
