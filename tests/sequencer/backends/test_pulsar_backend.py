import json
import pathlib
import pytest
from quantify import set_datadir
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils.helpers import NumpyJSONEncoder
from quantify.sequencer.types import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CNOT, CZ, Rxy
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
    short_final_wait = [
        (0, 'square_id'),
        (4, 'square_id')
    ]

    dummy_pulse_data = {
        'square_id_I': {'data': np.ones(4), 'index': 0},
        'square_id_Q': {'data': np.zeros(4), 'index': 1},
        'drag_ID_I': {'data': np.ones(2), 'index': 2},
        'drag_ID_Q': {'data': np.ones(2), 'index': 3}
    }

    with pytest.raises(ValueError, match="Pulse square_id at 0 has duration 4 but next timing is 2."):
        build_q1asm(too_close_pulse_timings, dummy_pulse_data, too_close_pulse_timings[-1][0] + 4)

    with pytest.raises(ValueError, match="Generated wait for '0':'drag_ID' caused exception 'duration 2ns < "
                                         "cycle time 4ns'"):
        build_q1asm(short_pulse_timings, dummy_pulse_data, short_pulse_timings[-1][0] + 4)

    with pytest.raises(ValueError, match="Generated wait for '0':'square_id' caused exception 'duration 2ns < "
                                         "cycle time 4ns'"):
        build_q1asm(short_wait_timings, dummy_pulse_data, 10)

    with pytest.raises(ValueError, match="Generated wait for '4':'square_id' caused exception 'duration 2ns < "
                                         "cycle time 4ns'"):
        build_q1asm(short_final_wait, dummy_pulse_data, 10)


def deploy_q1asm(cfg_path, messages=None, delete_file=False):
    qcm = pulsar_qcm_dummy('test')
    qcm.sequencer0_waveforms_and_program(str(cfg_path))
    if not messages:
        assert 'assembler finished successfully' in qcm.get_assembler_log()
    else:
        print(qcm.get_assembler_log())
        for message in messages:
            assert message in qcm.get_assembler_log()
    if delete_file:
        cfg_path.unlink()


def test_overflowing_instruction_times():
    real = np.random.random(129380)
    pulse_timings = [
        (0, 'square_ID')
    ]
    pulse_data = {
        'square_ID_I': {'data': real, 'index': 0},
        'square_ID_Q': {'data': np.zeros(len(real)), 'index': 1}
    }
    program_str = build_q1asm(pulse_timings, pulse_data, len(real))
    with open(pathlib.Path(__file__).parent.joinpath('ref_test_large_plays_q1asm'), 'rb') as f:
        assert program_str.encode('utf-8') == f.read()

    pulse_timings.append((229380 + pow(2, 16), 'square_ID'))
    program_str = build_q1asm(pulse_timings, pulse_data, 524296)
    with open(pathlib.Path(__file__).parent.joinpath('ref_test_large_waits_q1asm'), 'rb') as f:
        assert program_str.encode('utf-8') == f.read()


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

    err = r"Provided sequence_duration.*4.*less than the total runtime of this sequence.*20"
    with pytest.raises(ValueError, match=err):
        build_q1asm(pulse_timings, pulse_data, 4)

    # sequence_duration greater than final timing but less than total runtime
    err = r"Provided sequence_duration.*18.*less than the total runtime of this sequence.*20"
    with pytest.raises(ValueError, match=err):
        build_q1asm(pulse_timings, pulse_data, 18)


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

    if PULSAR_ASSEMBLER:
        with open('tmp.json', 'w') as f:
            f.write(json.dumps(sequence_cfg, cls=NumpyJSONEncoder))
        qcm = pulsar_qcm_dummy('test')
        qcm.sequencer0_waveforms_and_program('tmp.json')
        assert 'assembler finished successfully' in qcm.get_assembler_log()
        pathlib.Path('tmp.json').unlink()


@pytest.mark.skip('Not Implemented')
def test_pulsar_assembler_backend_missing_pulse_info():
    # should raise an exception
    pass


@pytest.mark.skip('Not Implemented')
def test_pulsar_assembler_backend_missing_timing_info():
    pass


@pytest.fixture
def dummy_pulsars():
    if PULSAR_ASSEMBLER:
        _pulsars = []
        for qcm_name in ['qcm0', 'qcm1', 'qrm0', 'qrm1']:
            _pulsars.append(pulsar_qcm_dummy(qcm_name))
    else:
        _pulsars = []

    # ensures the default datadir is used which is excluded from git
    set_datadir(None)
    yield _pulsars

    # teardown
    for instr_name in list(Instrument._all_instruments):
        try:
            inst = Instrument.find_instrument(instr_name)
            inst.close()
        except KeyError:
            pass


def test_pulsar_assembler_backend(dummy_pulsars):
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
        sched.add(operation=CZ(qC=q0.name, qT=q1.name))
        sched.add(Rxy(theta=theta, phi=0, qubit=q0.name))
        sched.add(Rxy(theta=90, phi=0, qubit=q1.name))
        sched.add(Measure(q0.name, q1.name), label='M {:.2f} deg'.format(theta))

    # Add the resources for the pulsar qcm channels
    qcm0 = CompositeResource('qcm0', ['qcm0.s0', 'qcm0.s1'])
    qcm0_s0 = Pulsar_QCM_sequencer('qcm0.s0', instrument_name='qcm0', seq_idx=0)
    qcm0_s1 = Pulsar_QCM_sequencer('qcm0.s1', instrument_name='qcm0', seq_idx=1)

    qcm1 = CompositeResource('qcm1', ['qcm1.s0', 'qcm1.s1'])
    qcm1_s0 = Pulsar_QCM_sequencer('qcm1.s0', instrument_name='qcm1', seq_idx=0)
    qcm1_s1 = Pulsar_QCM_sequencer('qcm1.s1', instrument_name='qcm1', seq_idx=1)

    qrm0 = CompositeResource('qrm0', ['qrm0.s0', 'qrm0.s1'])
    # Currently mocking a readout module using an acquisition module
    qrm0_s0 = Pulsar_QCM_sequencer('qrm0.s0', instrument_name='qrm0', seq_idx=0)
    qrm0_s1 = Pulsar_QCM_sequencer('qrm0.s1', instrument_name='qrm0', seq_idx=1)

    # using qcm sequencing modules to fake a readout module
    qrm0_r0 = Pulsar_QCM_sequencer('qrm0.r0', instrument_name='qrm0', seq_idx=0)
    qrm0_r1 = Pulsar_QCM_sequencer('qrm0.r1', instrument_name='qrm0', seq_idx=1)

    sched.add_resources([qcm0, qcm0_s0, qcm0_s1, qcm1, qcm1_s0, qcm1_s1, qrm0, qrm0_s0, qrm0_s1,  qrm0_r0, qrm0_r1])

    sched = add_pulse_information_transmon(sched, DEVICE_TEST_CFG)
    sched = determine_absolute_timing(sched)

    seq_config_dict = pulsar_assembler_backend(sched, configure_hardware=PULSAR_ASSEMBLER)

    assert len(sched.resources['qcm0.s0'].timing_tuples) == int(21*3)
    assert len(qcm0_s0.timing_tuples) == int(21*3)
    assert len(qcm0_s1.timing_tuples) == 0
    assert len(qcm1_s0.timing_tuples) == 21
    assert len(qcm1_s1.timing_tuples) == 0

    assert sched.resources['qcm0.s0']['nco_freq'] == DEVICE_TEST_CFG["qubits"]["q0"]["mw_modulation_freq"]
    assert sched.resources['qrm0.s0']['nco_freq'] == DEVICE_TEST_CFG["qubits"]["q0"]["ro_pulse_modulation_freq"]
    assert sched.resources['qrm0.r0']['nco_freq'] == DEVICE_TEST_CFG["qubits"]["q0"]["ro_pulse_modulation_freq"]
    assert sched.resources['qcm1.s0']['nco_freq'] == DEVICE_TEST_CFG["qubits"]["q1"]["mw_modulation_freq"]
    assert sched.resources['qrm0.s1']['nco_freq'] == DEVICE_TEST_CFG["qubits"]["q1"]["ro_pulse_modulation_freq"]
    assert sched.resources['qrm0.r1']['nco_freq'] == DEVICE_TEST_CFG["qubits"]["q1"]["ro_pulse_modulation_freq"]

    # if PULSAR_ASSEMBLER:
    #     seq_config_dict = pulsar_assembler_backend(sched, program_sequencers=True)

    # assert right keys.
    # assert right content of the config files.


def test_mismatched_mod_freq():
    bad_config = {
        "qubits": {
            "q0": {"mw_amp180": 0.75, "mw_motzoi": -0.25, "mw_duration": 20e-9, "mw_modulation_freq": 50e6,
                   "mw_ef_amp180": 0.87, "mw_ch": "qcm0.s0"},
            "q1": {"mw_amp180": 0.75, "mw_motzoi": -0.25, "mw_duration": 20e-9, "mw_modulation_freq": 70e6,
                   "mw_ef_amp180": 0.87, "mw_ch": "qcm0.s0"}
        },
        "edges": {
            "q0-q1": {}
        }
    }
    sched = Schedule('Mismatched mod freq')
    q0, q1 = (QubitResource('q0'), QubitResource('q1'))
    sched.add_resource(q0)
    sched.add_resource(q1)
    sched.add(Rxy(theta=90, phi=0, qubit=q0.name))
    sched.add(Rxy(theta=90, phi=0, qubit=q1.name))
    qcm0_s0 = Pulsar_QCM_sequencer('qcm0.s0', instrument_name='qcm0', seq_idx=0)
    sched.add_resource(qcm0_s0)
    sched = add_pulse_information_transmon(sched, bad_config)
    sched = determine_absolute_timing(sched)
    with pytest.raises(ValueError, match=r'pulse.*\d+ on channel qcm0.s0 has divergent modulation frequency: '
                                         r'expected 50000000 but was 70000000'):
        pulsar_assembler_backend(sched, configure_hardware=PULSAR_ASSEMBLER)


@pytest.mark.skipif(not PULSAR_ASSEMBLER, reason="requires pulsar_qcm assembler to be installed")
def test_configure_pulsar_sequencers():

    pass


@pytest.mark.skip('no reason')
def test_rounding_errors_in_timing():

    pass
