import os
import inspect
import json
import logging
from qcodes.utils.helpers import NumpyJSONEncoder
from columnar import columnar
from qcodes import Instrument
import numpy as np
from quantify.data.handling import gen_tuid, create_exp_folder
from quantify.utilities.general import make_hash, without, import_func_from_string


class Q1ASMBuilder:
    CYCLE_TIME_ns = 4  # 250MHz processor
    WAVEFORM_IDX_SZ = pow(2, 10) - 1
    IMMEDIATE_SZ = pow(2, 16) - 1

    def __init__(self):
        self.rows = []

    def get_str(self):
        return columnar(self.rows, no_borders=True)

    @staticmethod
    def _iff(label):
        return '{}:'.format(label) if label else ''

    def _check_wave_idx(self, idx):
        if idx > self.WAVEFORM_IDX_SZ:
            raise ValueError("waveform_idx must be less than {}, actual: {}".format(self.WAVEFORM_IDX_SZ, idx))
        return idx

    def _check_immediate(self, val):
        if val > self.IMMEDIATE_SZ:
            raise ValueError("immediate value must be less than {}, actual: {}".format(self.IMMEDIATE_SZ, val))
        return val

    def _check_playtime(self, duration):
        if duration < self.CYCLE_TIME_ns:
            raise ValueError("instruction time must be greater than {}, actual: {}"
                             .format(self.CYCLE_TIME_ns, duration))
        split = []
        while duration > self.IMMEDIATE_SZ:
            split.append(self.IMMEDIATE_SZ)
            duration -= self.IMMEDIATE_SZ
        split.append(duration)
        return split

    def line_break(self):
        self.rows.append(['', '', '', ''])

    def wait_sync(self):
        self.rows.append(['', '', '', '#Wait for all sequencers to be ready'])

    def move(self, label, source, target, comment):
        if isinstance(source, int):
            self._check_immediate(source)
        self.rows.append([self._iff(label), 'move', '{},{}'.format(source, target), comment])

    def play(self, label, I_idx, Q_idx, playtime, comment):
        self._check_wave_idx(I_idx)
        self._check_wave_idx(Q_idx)
        for duration in self._check_playtime(playtime):
            args = '{},{},{}'.format(I_idx, Q_idx, duration)
            row = [self._iff(label), 'play', args, comment]
            label = None
            self.rows.append(row)

    def wait(self, label, playtime, comment):
        for duration in self._check_playtime(playtime):
            row = [label if label else '', 'wait', duration, comment]
            label = None
            self.rows.append(row)

    def jmp(self, label, target, comment):
        self.rows.append([self._iff(label), 'jmp', '@{}'.format(target), comment])


def pulsar_assembler_backend(schedule, tuid=None, configure_hardware=False):
    """
    Create sequencer configuration files for multiple Qblox pulsar modules.

    Sequencer configuration files contain assembly, a waveform dictionary and the
    parameters to be configured for every pulsar sequencer.

    The sequencer configuration files are stored in the quantify datadir (see :func:`~quantify.data.handling.get_datadir`)


    Parameters
    ------------
    schedule : :class:`~quantify.sequencer.types.Schedule` :
        The schedule to convert into assembly.

    tuid : :class:`~quantify.data.types.TUID` :
        a tuid of the experiment the schedule belongs to. If set to None, a new TUID will be generated to store
        the sequencer configuration files.

    configure_hardware : bool
        if True will configure the hardware to run the specified schedule.

    Returns
    ----------
    config_dict : dict
        a dictionary containing


    .. note::

        Currently only supports the Pulsar_QCM module.
        Does not yet support the Pulsar_QRM module.
    """

    for pls_idx, t_constr in enumerate(schedule.timing_constraints):
        op = schedule.operations[t_constr['operation_hash']]

        if len(op['pulse_info']) == 0:
            # this exception is raised when no pulses have been added yet.
            raise ValueError('Operation {} has no pulse info'.format(op))

        for p in op['pulse_info']:

            t0 = t_constr['abs_time']+p['t0']
            pulse_id = make_hash(without(p, 't0'))

            if p['channel'] is None:
                continue  # pulses with None channel will be ignored by this backend

            # Assumes the channel exists in the resources available to the schedule
            if p['channel'] not in schedule.resources.keys():
                raise KeyError('Resource "{}" not available in "{}"'.format(
                    p['channel'], schedule))

            ch = schedule.resources[p['channel']]
            ch.timing_tuples.append((int(t0*ch['sampling_rate']), pulse_id))

            # determine waveform
            if pulse_id not in ch.pulse_dict.keys():
                # TODO: configure the settings (modulation freq etc. for each seqeuncer)

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

    # Creating the files
    if tuid is None:
        tuid = gen_tuid()
    # Should use the folder of the matching file if tuid already exists
    exp_folder = create_exp_folder(tuid=tuid, name=schedule.name+'_schedule')
    seq_folder = os.path.join(exp_folder, 'schedule')
    os.makedirs(seq_folder, exist_ok=True)

    # Find the longest running schedule (other sequences must wait before repeating to stay in sync)
    # todo shame to sort the timing_tuples again here
    max_sequence_duration = 0
    for resource in schedule.resources.values():
        if hasattr(resource, 'timing_tuples') and resource.timing_tuples:
            final_pulse = sorted(resource.timing_tuples)[-1]
            final_timing = final_pulse[0] + len(resource.pulse_dict[final_pulse[1]])
            max_sequence_duration = final_timing if final_timing > max_sequence_duration else max_sequence_duration

    # Convert timing tuples and pulse dicts for each seqeuncer into assembly configs
    config_dict = {}
    for resource in schedule.resources.values():
        if hasattr(resource, 'timing_tuples'):
            seq_cfg = generate_sequencer_cfg(
                pulse_info=resource.pulse_dict,
                timing_tuples=sorted(resource.timing_tuples),
                sequence_duration=max_sequence_duration)
            seq_cfg['instr_cfg'] = resource.data

            seq_fn = os.path.join(
                seq_folder, '{}_sequencer_cfg.json'.format(resource.name))
            with open(seq_fn, 'w') as f:
                json.dump(seq_cfg, f, cls=NumpyJSONEncoder, indent=4)
            config_dict[resource.name] = seq_fn

    if configure_hardware:
        configure_pulsar_sequencers(config_dict)

    # returns a dict of sequencer names as keys with json filenames as values.
    # add bool option to program immediately?
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
            qcm = Instrument.find_instrument(instr_cfg['instrument_name'])

            if instr_cfg['seq_idx'] == 0:
                # configure settings
                qcm.set('sequencer{}_mod_enable'.format(
                    instr_cfg['seq_idx']), instr_cfg['mod_enable'])
                qcm.set('sequencer{}_nco_freq'.format(
                    instr_cfg['seq_idx']), instr_cfg['nco_freq'])
                qcm.set('sequencer{}_cont_mode_en'.format(
                    instr_cfg['seq_idx']), False)
                qcm.set('sequencer{}_cont_mode_waveform_idx'.format(
                    instr_cfg['seq_idx']), 0)
                qcm.set('sequencer{}_upsample_rate'.format(
                    instr_cfg['seq_idx']), 0)

                # configure sequencer
                qcm.set('sequencer{}_waveforms_and_program'.format(instr_cfg['seq_idx']),
                        config_fn)
            else:
                logging.warning(
                    'Not Implemented, awaiting driver for more than one seqeuncer')


def build_waveform_dict(pulse_info):
    """
    Allocates numerical pulse representation to indices and formats for sequencer JSON.

    Args:
        pulse_info (dict): Pulse ID to array-like numerical representation

    Returns:
        Dictionary mapping pulses to numerical representation and memory index
    """
    sequencer_cfg = {"waveforms": {}}
    idx_offset = 0
    for idx, (pulse_id, data) in enumerate(pulse_info.items()):
        arr = np.array(data)

        I = arr.real
        Q = arr.imag  # real-valued arrays automatically evaluate to an array of zeros

        sequencer_cfg["waveforms"]["{}_I".format(pulse_id)] = {
            "data": I,
            "index": idx + idx_offset
        }
        idx_offset += 1
        sequencer_cfg["waveforms"]["{}_Q".format(pulse_id)] = {
            "data": Q,
            "index": idx + idx_offset
        }
    return sequencer_cfg


def build_q1asm(timing_tuples, pulse_dict, sequence_duration):
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

    Returns:
        A q1asm program in a string.
    """

    def get_pulse_runtime(pulse_id):
        return len(pulse_dict["{}_I".format(pulse_id)]['data'])

    def get_pulse_finish_time(pulse_idx):
        start_time = timing_tuples[pulse_idx][0]
        runtime = get_pulse_runtime(timing_tuples[pulse_idx][1])
        return start_time + runtime

    def attempt(operation, args, time, id):
        try:
            operation(*args)
        except ValueError as e:
            raise ValueError("'{}', at {}:{}, sequence_duration: {}".format(str(e), time, id, sequence_duration))

    q1asm = Q1ASMBuilder()
    q1asm.move('start', len(pulse_dict), 'R0', '#Waveform count register')

    if timing_tuples and get_pulse_finish_time(-1) > sequence_duration:
        raise ValueError("Provided sequence_duration '{}' is less than the total runtime of this sequence ({})."
                         .format(sequence_duration, get_pulse_finish_time(-1)))

    previous = None  # previous pulse
    clock = 0  # current execution time
    for timing, pulse_id in timing_tuples:
        """
        # check each operation has the minimum required timing
        if previous:
            try:
                q1asm._check_playtime(timing - previous[0])
            except ValueError as e:
                raise ValueError("Timings '{}' and '{}' are too close (must be at least {} ns)"
                                 .format(previous[0], timing, q1asm.CYCLE_TIME_ns))
        """

        # check if we must wait before beginning our next section
        wait_duration = timing - clock
        if previous and wait_duration > 0:
            # if the previous operation is not contiguous to the current, we must wait for period
            attempt(q1asm.wait, ['', wait_duration, '#Wait'], previous[0], previous[1])

        I = pulse_dict["{}_I".format(pulse_id)]['index']
        Q = pulse_dict["{}_Q".format(pulse_id)]['index']
        q1asm.line_break()

        duration = get_pulse_runtime(pulse_id)  # duration in nanoseconds, QCM sample rate is 1Gsps
        q1asm.play('', I, Q, duration, '#Play {}'.format(pulse_id))

        previous = (timing, pulse_id)
        clock += duration + wait_duration

    # check if we must wait to sync up with fellow sequencers
    final_wait = sequence_duration - clock
    if final_wait > 0:
        attempt(q1asm.wait, ['', final_wait, '#Sync with other sequencers'], timing_tuples[-1][0], timing_tuples[-1][1])

    q1asm.jmp('', 'start', '#Loop back to start')
    return q1asm.get_str()


def generate_sequencer_cfg(pulse_info, timing_tuples, sequence_duration: int):
    """
    Generate a JSON compatible dictionary for defining a sequencer configuration. Contains a list of waveforms and a
    program in a q1asm string.

    Args:
        pulse_info (dict): mapping of pulse IDs to numerical waveforms
        timing_tuples (list): time ordered list of tuples containing the absolute starting time and pulse ID
        sequence_duration (int): maximum runtime of this sequence

    Returns:
        Sequencer configuration
    """
    cfg = build_waveform_dict(pulse_info)
    cfg['program'] = build_q1asm(timing_tuples, cfg['waveforms'], sequence_duration)
    return cfg
