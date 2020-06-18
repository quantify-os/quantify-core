import pytest
import numpy as np
from quantify.sequencer import Operation
from quantify.sequencer.gate_library import X90
from quantify.sequencer.pulse_library import SquarePulse, DRAGPulse, IdlePulse, ModSquarePulse

def test_operation_duration_single_pulse():
    dgp = DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=20e-9,
                    freq_mod=15e6,
                    ch_I='ch1', ch_Q='ch2')
    assert dgp.duration == pytest.approx(20e-9)
    idle = IdlePulse(50e-9)
    assert idle.duration == pytest.approx(50e-9)


def test_operation_duration_single_pulse_delayed():
    dgp = DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=10e-9,
                    freq_mod=15e6,
                    ch_I='ch1', ch_Q='ch2', t0=3.4e-9)

    assert dgp.duration == pytest.approx(13.4e-9)


def test_operation_add_pulse():

    dgp1 = DRAGPulse(G_amp=.8, D_amp=-.3, phase=0, duration=20e-9,
                     freq_mod=15e6,
                     ch_I='ch1', ch_Q='ch2', t0=0)

    assert len(dgp1['pulse_info']) == 1
    dgp1.add_pulse(dgp1)
    assert len(dgp1['pulse_info']) == 2

    x90 = X90('q1')
    assert len(x90['pulse_info']) == 0
    dgp = DRAGPulse(G_amp=.8, D_amp=-.3, phase=0, duration=20e-9,
                    freq_mod=15e6,
                    ch_I='ch1', ch_Q='ch2', t0=0)
    x90.add_pulse(dgp)
    assert len(x90['pulse_info']) == 1


def test_operation_duration_composite_pulse():

    dgp1 = DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=10e-9,
                     freq_mod=15e6,
                     ch_I='ch1', ch_Q='ch2', t0=0)

    assert dgp1.duration == pytest.approx(10e-9)

    dgp2 = DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=7e-9,
                     freq_mod=15e6,
                     ch_I='ch1', ch_Q='ch2', t0=2e-9)

    # Adding a shorter pulse is not expected to change the duration
    dgp1.add_pulse(dgp2)
    assert dgp1.duration == pytest.approx(10e-9)

    # adding a longer pulse is expected to change the duration
    dgp3 = DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=12e-9,
                     freq_mod=15e6,
                     ch_I='ch3', ch_Q='ch4', t0=3.4e-9)
    dgp1.add_pulse(dgp3)
    assert dgp3.duration == pytest.approx(15.4e-9)
    assert dgp1.duration == pytest.approx(15.4e-9)


def test_pulses_valid():

    sqp = SquarePulse(amp=.5, duration=300e-9, ch='ch')
    msqp = ModSquarePulse(amp=.5, duration=300e-9, ch_I='ch5', ch_Q='ch6')
    dgp = DRAGPulse(G_amp=.8, D_amp=-.3, phase=24.3, duration=20e-9,
                    freq_mod=15e6,
                    ch_I='ch1', ch_Q='ch2')

    idle = IdlePulse(duration=50e-9)
    assert Operation.is_valid(sqp)
    assert Operation.is_valid(msqp)
    assert Operation.is_valid(dgp)
    assert Operation.is_valid(idle)
