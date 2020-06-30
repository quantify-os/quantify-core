import json
import pathlib
import pytest
from quantify import set_datadir
import numpy as np
from qcodes.instrument.base import Instrument
from quantify.sequencer.types import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy
from quantify.sequencer.backends.pulsar_backend import build_waveform_dict, build_q1asm, generate_sequencer_cfg, pulsar_assembler_backend
from quantify.sequencer.resources import QubitResource, CompositeResource, Pulsar_QCM_sequencer
from quantify.sequencer.compilation import add_pulse_information_transmon, determine_absolute_timing

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

try:
    from pulsar_qcm.pulsar_qcm import pulsar_qcm_dummy
    PULSAR_ASSEMBLER = True
except ImportError:
    PULSAR_ASSEMBLER = False

from ... import test_data  # relative-import the *package* containing the templates


DEVICE_TEST_CFG = json.loads(pkg_resources.read_text(
    test_data, 'transmon_test_config.json'))


def test_build_waveform_dict():
    real = np.random.random(int(4e3))
    complex_vals = real + (np.random.random(int(4e3)) * 1.0j)

    pulse_data = {
        'gdfshdg45': complex_vals,
        '6h5hh5hyj': real,
    }
    sequence_cfg = build_waveform_dict(pulse_data)
    assert len(sequence_cfg['waveforms']) == 2 * len(pulse_data)
    wf_1 = sequence_cfg['waveforms']['gdfshdg45_I']
    wf_2 = sequence_cfg['waveforms']['gdfshdg45_Q']
    wf_3 = sequence_cfg['waveforms']['6h5hh5hyj_I']
    wf_4 = sequence_cfg['waveforms']['6h5hh5hyj_Q']
    np.testing.assert_array_equal(wf_1['data'], complex_vals.real)
    assert wf_1['index'] == 0
    np.testing.assert_array_equal(wf_2['data'], complex_vals.imag)
    assert wf_2['index'] == 1
    np.testing.assert_array_equal(wf_3['data'], real)
    assert wf_3['index'] == 2
    np.testing.assert_array_equal(wf_4['data'], np.zeros(len(wf_4['data'])))
    assert wf_4['index'] == 3


def test_bad_pulse_timings():
    too_close_pulse_timings = [
        (0, 'square_id'),
        (2, 'drag_ID')
    ]

    short_pulse_timings = [
        (0, 'drag_ID'),
        (4, 'square_id')
    ]

    short_wait_timings = [
        (0, 'square_id'),
        (6, 'square_id')
    ]

    dummy_pulse_data = {
        'square_id_I': {'data': np.ones(4), 'index': 0},
        'square_id_Q': {'data': np.zeros(4), 'index': 1},
        'drag_ID_I': {'data': np.ones(2), 'index': 2},
        'drag_ID_Q': {'data': np.ones(2), 'index': 3}
    }

    with pytest.raises(ValueError) as e:
        build_q1asm(too_close_pulse_timings, dummy_pulse_data, too_close_pulse_timings[-1][0] + 4)
        e.match(r'Timings.*0.*2.*too close.*must be at least 4ns')

    with pytest.raises(ValueError) as e:
        build_q1asm(short_pulse_timings, dummy_pulse_data, short_pulse_timings[-1][0] + 4)
        e.match(r'Pulse.*drag_ID.*at timing.*0.*is too short.*must be at least 4ns')

    with pytest.raises(ValueError) as e:
        build_q1asm(short_wait_timings, dummy_pulse_data, short_wait_timings[-1][0] + 4)
        e.match(r'Insufficient wait period between pulses.*square_ID.*and.*square_ID.*timings.*0.*6.*square_ID.*'
                r'duration of 4ns necessitating a wait of duration 2ns.*must be at least 4ns')


def test_build_q1asm():
    real = np.random.random(4)
    complex_vals = real + (np.random.random(4) * 1.0j)

    pulse_timings = [
        (0, 'square_id'),
        (4, 'drag_ID'),
        (16, 'square_id')
    ]

    pulse_data = {
        'square_id_I': {'data': real, 'index': 0},
        'square_id_Q': {'data': np.zeros(len(real)), 'index': 1},
        'drag_ID_I': {'data': complex_vals.real, 'index': 2},
        'drag_ID_Q': {'data': complex_vals.imag, 'index': 3}
    }

    program_str = build_q1asm(pulse_timings, pulse_data, 20)
    with open(pathlib.Path(__file__).parent.joinpath('ref_test_build_q1asm'), 'rb') as f:
        assert program_str.encode('utf-8') == f.read()

    program_str_sync = build_q1asm(pulse_timings, pulse_data, 30)
    with open(pathlib.Path(__file__).parent.joinpath('ref_test_build_q1asm_sync'), 'rb') as f:
        assert program_str_sync.encode('utf-8') == f.read()

    with pytest.raises(ValueError) as e:
        build_q1asm(pulse_timings, pulse_data, 4)
        e.match(r'Provided sequence_duration 4 is less than the total runtime of this sequence (20).')

    # sequence_duration greater than final timing but less than total runtime
    with pytest.raises(ValueError) as e:
        build_q1asm(pulse_timings, pulse_data, 18)
        e.match(r'Provided sequence_duration 18 is less than the total runtime of this sequence (20).')

    with pytest.raises(ValueError) as e:
        build_q1asm(pulse_timings, pulse_data, 22)
        e.match(r'Insufficient sync period between final timing.*16.*and sequence_duration.*22.*must be at least 4ns')


def test_generate_sequencer_cfg():
    pulse_timings = [
        (0, 'square_1'),
        (4, 'drag_1'),
        (16, 'square_2'),
    ]

    real = np.random.random(4)
    complex_vals = real + (np.random.random(4) * 1.0j)
    pulse_data = {
        "square_1": [0.0, 1.0, 0.0, 0.0],
        "drag_1": complex_vals,
        "square_2": real,
    }

    def check_waveform(entry, exp_data, exp_idx):
        assert exp_idx == entry['index']
        np.testing.assert_array_equal(exp_data, entry['data'])

    sequence_cfg = generate_sequencer_cfg(pulse_data, pulse_timings, 20)
    check_waveform(sequence_cfg['waveforms']["square_1_I"], [0.0, 1.0, 0.0, 0.0], 0)
    check_waveform(sequence_cfg['waveforms']["square_1_Q"], np.zeros(4), 1)
    check_waveform(sequence_cfg['waveforms']["drag_1_I"], complex_vals.real, 2)
    check_waveform(sequence_cfg['waveforms']["drag_1_Q"], complex_vals.imag, 3)
    check_waveform(sequence_cfg['waveforms']["square_2_I"], real, 4)
    check_waveform(sequence_cfg['waveforms']["square_2_Q"], np.zeros(4), 5)
    assert len(sequence_cfg['program'])


@pytest.mark.skip('Not Implemented')
def test_pulsar_assembler_backend_missing_pulse_info():
    # should raise an exception
    pass


@pytest.mark.skip('Not Implemented')
def test_pulsar_assembler_backend_missing_timing_info():
    pass

class TestAssemblerBackend:
    @classmethod
    def setup_class(cls):
        if PULSAR_ASSEMBLER:
            _pulsars = []
            for qcm_name in ['qcm0', 'qcm1', 'qrm0', 'qrm1']:
                _pulsars.append(pulsar_qcm_dummy(qcm_name))
        # ensures the default datadir is used which is excluded from git
        set_datadir(None)


    @classmethod
    def teardown_class(self):
        for instr_name in list(Instrument._all_instruments):
            try:
                inst = Instrument.find_instrument(instr_name)
                inst.close()
            except KeyError:
                pass
        set_datadir(None)



    def test_pulsar_assembler_backend(self):
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


        seq_config_dict = pulsar_assembler_backend(sched,
            configure_hardware=PULSAR_ASSEMBLER)


        assert len(sched.resources['qcm0.s0'].timing_tuples) == int(21*2)
        assert len(qcm0_s0.timing_tuples) == int(21*2)
        assert len(qcm0_s1.timing_tuples) == 0
        assert len(qcm1_s0.timing_tuples) == 21
        assert len(qcm1_s1.timing_tuples) == 0


        # if PULSAR_ASSEMBLER:
        #     seq_config_dict = pulsar_assembler_backend(sched, program_sequencers=True)


        # assert right keys.
        # assert right content of the config files.


    @pytest.mark.skipif(not PULSAR_ASSEMBLER, reason="requires pulsar_qcm assembler to be installed")
    def test_configure_pulsar_sequencers(self):

        pass
