import pytest
import numpy as np
from quantify.sequencer.backends import pulse_diagram_plotly, circuit_diagram_matplotlib
from quantify.sequencer import Schedule
from quantify.sequencer.compilation import determine_absolute_timing
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy
from quantify.sequencer.compilation import determine_absolute_timing, validate_config, add_pulse_information_transmon
import matplotlib.pyplot as plt


def test_circuit_diagram_matplotlib():
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

    sched = determine_absolute_timing(sched, clock_unit='ideal')
    f, ax = circuit_diagram_matplotlib(sched)


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


def test_pulse_diagram_plotly():
    sched = Schedule('Test schedule')

    # define the resources
    q0, q1 = ('q0', 'q1')
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    # sched.add(operation=CNOT(qC=q0, qT=q1)) # not implemented in config
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')
    # pulse information is added
    sched = add_pulse_information_transmon(sched, device_cfg=device_test_cfg)
    sched = determine_absolute_timing(sched, clock_unit='physical')

    # It should be possible to generate this visualization after compilation
    fig = pulse_diagram_plotly(
        sched, ch_list=['ch0', 'ch5.0', 'ch6.0', 'acq_ch1'])
    # and with auto labels
    fig = pulse_diagram_plotly(
        sched)
