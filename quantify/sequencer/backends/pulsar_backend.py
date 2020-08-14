"""
-----------------------------------------------------------------------------
Description:    Compiler backend for the Pulsar QCM.
Repository:     https://gitlab.com/qblox/packages/software/quantify/
Copyright (C) Qblox BV (2020)
-----------------------------------------------------------------------------
"""
import os
import inspect
import json
from qcodes.utils.helpers import NumpyJSONEncoder
from columnar import columnar
from qcodes import Instrument
import numpy as np
from quantify.data.handling import gen_tuid, create_exp_folder
from quantify.utilities.general import make_hash, without, import_func_from_string


class Q1ASMBuilder:
    """
    Generates a q1asm program instruction by instruction. Handles splitting overflowing operation times.
    """
    IMMEDIATE_SZ = pow(2, 16) - 1
    CYCLE_TIME_ns = 4

    def __init__(self):
        self.rows = []

    def get_str(self):
        """
        Returns:
            The program in string format
        """
        return columnar(self.rows, no_borders=True)

    @staticmethod
    def _iff(label):
        return '{}:'.format(label) if label else ''

    def _split_playtime(self, duration):
        split = []
        while duration > self.IMMEDIATE_SZ:
            split.append(self.IMMEDIATE_SZ)
            duration -= self.IMMEDIATE_SZ
        split.append(duration)
        return split

    def _check_playtime(self, duration):
        if duration < self.CYCLE_TIME_ns:
            raise ValueError('duration {}ns < cycle time {}ns'.format(duration, self.CYCLE_TIME_ns))
        return duration

    def line_break(self):
        self.rows.append(['', '', '', ''])

    def wait_sync(self):
        self.rows.append(['', 'wait_sync', '4', '#sync'])

    def move(self, label, source, target, comment):
        self.rows.append([self._iff(label), 'move', '{},{}'.format(source, target), comment])

    def play(self, label, I_idx, Q_idx, playtime, comment):
        for duration in self._split_playtime(playtime):
            args = '{},{},{}'.format(I_idx, Q_idx, duration)
            row = [self._iff(label), 'play', args, comment]
            label = None
            self.rows.append(row)

    def acquire(self, label, I_idx, Q_idx, playtime, comment):
        for duration in self._split_playtime(playtime):
            args = '{},{},{}'.format(I_idx, Q_idx, duration)
            row = [self._iff(label), 'acquire', args, comment]
            label = None
            self.rows.append(row)

    def wait(self, label, playtime, comment):
        """
        Insert a wait instruction.

        Args:
             label (str):       Label to add
             playtime (int):    Duration to wait. Will be split into multiple instructions if greater than max size.
             comment (str):     Comment to add
        """
        for duration in self._split_playtime(playtime):
            row = [label if label else '', 'wait', self._check_playtime(duration), comment]
            label = None
            self.rows.append(row)

    def jmp(self, label, target, comment):
        self.rows.append([self._iff(label), 'jmp', '@{}'.format(target), comment])


def pulsar_assembler_backend(schedule, tuid=None, configure_hardware=False, debug=False):
    """
    Create sequencer configuration files for multiple Qblox pulsar modules.

    Sequencer configuration files contain assembly, a waveform dictionary and the
    parameters to be configured for every pulsar sequencer.

    The sequencer configuration files are stored in the quantify datadir
    (see :func:`~quantify.data.handling.get_datadir`)


    Parameters
    ------------
    schedule : :class:`~quantify.sequencer.types.Schedule` :
        The schedule to convert into assembly.

    tuid : :class:`~quantify.data.types.TUID` :
        a tuid of the experiment the schedule belongs to. If set to None, a new TUID will be generated to store
        the sequencer configuration files.

    configure_hardware : bool
        if True will configure the hardware to run the specified schedule.

    debug : bool
        if True will produce extra debug output

    Returns
    ----------
    config_dict : dict
        of sequencer names as keys with json filenames as values


    .. note::

        Currently only supports the Pulsar_QCM module.
        Does not yet support the Pulsar_QRM module.
    """

    max_seq_duration = 0
    acquisitions = set()
    for pls_idx, t_constr in enumerate(schedule.timing_constraints):
        op = schedule.operations[t_constr['operation_hash']]

        if len(op['pulse_info']) == 0:
            # this exception is raised when no pulses have been added yet.
            raise ValueError('Operation {} has no pulse info'.format(op.name))

        for p in op['pulse_info']:
            if 'abs_time' not in t_constr:
                raise ValueError("Absolute timing has not been determined for the schedule '{}'".format(schedule.name))

            t0 = t_constr['abs_time']+p['t0']
            pulse_id = make_hash(without(p, 't0'))

            if p['channel'] is None:
                continue  # pulses with None channel will be ignored by this backend

            if p['channel'][-8:] == '_READOUT':
                acquisitions.add(pulse_id)
                p['channel'] = p['channel'][:-8]

            # Assumes the channel exists in the resources available to the schedule
            if p['channel'] not in schedule.resources.keys():
                raise KeyError('Resource "{}" not available in "{}"'.format(p['channel'], schedule))

            ch = schedule.resources[p['channel']]
            ch.timing_tuples.append((round(t0*ch['sampling_rate']), pulse_id))

            # determine waveform
            if pulse_id not in ch.pulse_dict.keys():
                if 'freq_mod' in p:
                    if ch['type'] == 'Pulsar_QCM_sequencer' and ch['nco_freq'] != 0 and p['freq_mod'] != ch['nco_freq']:
                        raise ValueError('pulse {} on channel {} has an inconsistent modulation frequency: expected {} '
                                         'but was {}'
                                         .format(pulse_id, ch['name'], int(ch['nco_freq']), int(p['freq_mod'])))
                    if ch['nco_freq'] == 0:
                        ch['nco_freq'] = p['freq_mod']

                # the pulsar backend makes use of real-time pulse modulation
                t = np.arange(0, 0+p['duration'], 1/ch['sampling_rate'])
                wf_func = import_func_from_string(p['wf_func'])

                # select the arguments for the waveform function that are present in pulse info
                par_map = inspect.signature(wf_func).parameters
                wf_kwargs = {}
                for kw in par_map.keys():
                    if kw in p.keys():
                        wf_kwargs[kw] = p[kw]
                # Calculate the numerical waveform using the wf_func
                wf = wf_func(t=t, **wf_kwargs)
                ch.pulse_dict[pulse_id] = wf

            seq_duration = ch.timing_tuples[-1][0] + len(ch.pulse_dict[pulse_id])
            max_seq_duration = max_seq_duration if max_seq_duration > seq_duration else seq_duration

    # Creating the files
    if tuid is None:
        tuid = gen_tuid()
    # Should use the folder of the matching file if tuid already exists
    exp_folder = create_exp_folder(tuid=tuid, name=schedule.name+'_schedule')
    seq_folder = os.path.join(exp_folder, 'schedule')
    os.makedirs(seq_folder, exist_ok=True)

    # Convert timing tuples and pulse dicts for each seqeuncer into assembly configs
    config_dict = {}
    for resource in schedule.resources.values():
        if hasattr(resource, 'timing_tuples'):
            seq_cfg = generate_sequencer_cfg(
                pulse_info=resource.pulse_dict,
                timing_tuples=sorted(resource.timing_tuples),
                sequence_duration=max_seq_duration,
                acquisitions=acquisitions
            )
            seq_cfg['instr_cfg'] = resource.data

            if debug:
                qasm_dump = os.path.join(seq_folder, '{}_sequencer.q1asm'.format(resource.name))
                with open(qasm_dump, 'w') as f:
                    f.write(seq_cfg['program'])

            seq_fn = os.path.join(seq_folder, '{}_sequencer_cfg.json'.format(resource.name))
            with open(seq_fn, 'w') as f:
                json.dump(seq_cfg, f, cls=NumpyJSONEncoder, indent=4)
            config_dict[resource.name] = seq_fn

    if configure_hardware:
        configure_pulsar_sequencers(config_dict)

    return config_dict


def configure_pulsar_sequencers(config_dict: dict):
    """
    Configures multiple pulsar modules based on a configuration dictionary.

    Parameters
    ------------
    config_dict: dict
        Dictionary with resource_names as keys and filenames of sequencer config json files as values.
    """
    for resource, config_fn in config_dict.items():
        with open(config_fn) as seq_config:
            data = json.load(seq_config)
            instr_cfg = data['instr_cfg']
            pulsar = Instrument.find_instrument(instr_cfg['instrument_name'])
            is_qrm = instr_cfg['type'] == "Pulsar_QRM_sequencer"

            # configure settings
            seq_idx = instr_cfg['seq_idx']
            if seq_idx > 0:
                continue  # multiple sequencers not supported yet

            pulsar.set("sequencer{}_sync_en".format(seq_idx), True)
            pulsar.set('sequencer{}_nco_freq'.format(seq_idx), instr_cfg['nco_freq'])
            pulsar.set('sequencer{}_nco_phase'.format(seq_idx), instr_cfg['nco_phase'])
            mod_enable = True if instr_cfg['nco_freq'] != 0 or instr_cfg['nco_phase'] != 0 else False
            pulsar.set('sequencer{}_mod_en_awg'.format(seq_idx), mod_enable)
            for path in (0, 1):
                awg_path = "_awg_path{}".format(path)
                pulsar.set('sequencer{}_cont_mode_en{}'.format(seq_idx, awg_path), False)
                pulsar.set('sequencer{}_cont_mode_waveform_idx{}'.format(seq_idx, awg_path), 0)
                pulsar.set('sequencer{}_upsample_rate{}'.format(seq_idx, awg_path), 0)
                pulsar.set('sequencer{}_gain{}'.format(seq_idx, awg_path), 1)
                pulsar.set('sequencer{}_offset{}'.format(seq_idx, awg_path), 0)

            if is_qrm:
                # todo check with Jordy wrt Callum&Marijn found involving acquisitions and wait_sync
                pulsar.set("sequencer{}_trigger_mode_acq_path0".format(seq_idx), True)
                pulsar.set("sequencer{}_trigger_mode_acq_path1".format(seq_idx), True)

            # configure sequencer
            pulsar.set('sequencer{}_waveforms_and_program'.format(seq_idx), config_fn)


def build_waveform_dict(pulse_info: dict, acquisitions: set) -> dict:
    """
    Allocates numerical pulse representation to indices and formats for sequencer JSON.

    Args:
        pulse_info (dict): Pulse ID to array-like numerical representation
        acquisitions (set): set of pulse_IDs which are acquisitions

    Returns:
        Dictionary mapping pulses to numerical representation and memory index
    """
    sequencer_cfg = {"waveforms": {"awg": {}, "acq": {}}}
    for pulse_id, data in pulse_info.items():
        arr = np.array(data)
        I = arr.real  # noqa: E741
        Q = arr.imag  # real-valued arrays automatically evaluate to an array of zeros
        device = 'awg' if pulse_id not in acquisitions else 'acq'
        sequencer_cfg["waveforms"][device]["{}_I".format(pulse_id)] = {
            "data": I,
            "index": len(sequencer_cfg["waveforms"][device])
        }
        sequencer_cfg["waveforms"][device]["{}_Q".format(pulse_id)] = {
            "data": Q,
            "index": len(sequencer_cfg["waveforms"][device])
        }
    return sequencer_cfg


# todo this needs a serious clean up
def build_q1asm(timing_tuples: list, pulse_dict: dict, sequence_duration: int, acquisitions: set) -> str:
    """
    Converts operations and waveforms to a q1asm program. This function verifies these hardware based constraints:

        * Each pulse must run for at least the INSTRUCTION_CLOCK_TIME
        * Each operation must have a timing separation of at least INSTRUCTION_CLOCK_TIME

    .. warning:
        The above restrictions apply to any generated WAIT instructions.

    Args:
        timing_tuples (list): A sorted list of tuples matching timings to pulse_IDs.
        pulse_dict (dict): pulse_IDs to numerical waveforms with registered index in waveform memory.
        sequence_duration (int): maximum runtime of this sequence
        acquisitions (set): set of pulse_IDs which are acquisitions

    Returns:
        A q1asm program in a string.
    """

    def get_pulse_runtime(pulse_id):
        device = 'awg' if pulse_id not in acquisitions else 'acq'
        return len(pulse_dict[device]["{}_I".format(pulse_id)]['data'])

    def get_pulse_finish_time(pulse_idx):
        start_time = timing_tuples[pulse_idx][0]
        runtime = get_pulse_runtime(timing_tuples[pulse_idx][1])
        return start_time + runtime

    # Checks if our automatically generated 'sync' waits are too short.
    def auto_wait(label, duration, comment, previous):
        try:
            if duration > 0:
                q1asm.wait(label, duration, comment)
        except ValueError as e:
            raise ValueError("Generated wait for '{}':'{}' caused exception '{}'"
                             .format(previous[0], previous[1], str(e)))

    q1asm = Q1ASMBuilder()
    q1asm.wait_sync()
    q1asm.move('start', len(pulse_dict['awg']), 'R0', '#Waveform count register')

    if timing_tuples and get_pulse_finish_time(-1) > sequence_duration:
        raise ValueError("Provided sequence_duration '{}' is less than the total runtime of this sequence ({})."
                         .format(sequence_duration, get_pulse_finish_time(-1)))

    previous = None  # previous pulse
    clock = 0  # current execution time

    for idx, (timing, pulse_id) in enumerate(timing_tuples):
        # check if we must wait before beginning our next section
        wait_duration = timing - clock
        device = 'awg' if pulse_id not in acquisitions else 'acq'
        auto_wait('', wait_duration, '#Wait', previous)

        I = pulse_dict[device]["{}_I".format(pulse_id)]['index']  # noqa: E741
        Q = pulse_dict[device]["{}_Q".format(pulse_id)]['index']
        q1asm.line_break()

        # duration should be the pulse length or next start time
        next_timing = timing_tuples[idx+1][0] if idx < len(timing_tuples) - 1 else np.Inf
        pulse_runtime = get_pulse_runtime(pulse_id)  # duration in nanoseconds, QCM sample rate is # 1Gsps
        duration = min(next_timing, pulse_runtime)
        if device == 'awg':
            q1asm.play('', I, Q, duration, '#Play {}'.format(pulse_id))
        else:
            q1asm.acquire('', I, Q, duration, '#Play {}'.format(pulse_id))

        previous = (timing, pulse_id)
        clock += duration + wait_duration

    # check if we must wait to sync up with fellow sequencers
    final_wait_duration = sequence_duration - clock
    if timing_tuples:
        auto_wait('', final_wait_duration, '#Sync with other sequencers', timing_tuples[-1])

    q1asm.jmp('', 'start', '#Loop back to start')
    return q1asm.get_str()


def generate_sequencer_cfg(pulse_info, timing_tuples, sequence_duration: int, acquisitions: set):
    """
    Generate a JSON compatible dictionary for defining a sequencer configuration. Contains a list of waveforms and a
    program in a q1asm string.

    Args:
        pulse_info (dict): mapping of pulse IDs to numerical waveforms
        timing_tuples (list): time ordered list of tuples containing the absolute starting time and pulse ID
        sequence_duration (int): maximum runtime of this sequence
        acquisitions (set): set of pulse IDs which are acquisitions

    Returns:
        Sequencer configuration
    """
    cfg = build_waveform_dict(pulse_info, acquisitions)
    cfg['program'] = build_q1asm(timing_tuples, cfg['waveforms'], sequence_duration, acquisitions)
    return cfg
