import os
import inspect
import json
import logging
from qcodes.utils.helpers import NumpyJSONEncoder
from columnar import columnar
from collections import Counter
import numpy as np
from quantify.data.handling import gen_tuid, create_exp_folder
from quantify.utilities.general import make_hash, without, import_func_from_string


INSTRUCTION_CLOCK_TIME = 4  # 250MHz processor


def pulsar_assembler_backend(schedule, tuid=None):
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
                raise KeyError('Resource "{}" not available in "{}"'.format(p['channel'], schedule))

            ch = schedule.resources[p['channel']]
            ch.timing_tuples.append((int(t0*ch['sampling_rate']), pulse_id))

            # determine waveform
            if pulse_id not in ch.pulse_dict.keys():

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

    # Convert timing tuples and pulse dicts for each seqeuncer into assembly configs
    config_dict = {}
    for resource in schedule.resources.values():
        if hasattr(resource, 'timing_tuples'):
            seq_cfg = generate_sequencer_cfg(
                pulse_info=resource.pulse_dict,
                pulse_timings=sorted(resource.timing_tuples))
            seq_cfg['instr_cfg'] = resource.data

            seq_fn = os.path.join(seq_folder, '{}_sequencer_cfg.json'.format(resource.name))
            with open(seq_fn, 'w') as f:
                json.dump(seq_cfg, f, cls=NumpyJSONEncoder, indent=4)
            config_dict[resource.name] = seq_fn

            # TODO: configure the settings (modulation freq etc. for each seqeuncer)

    # returns a dict of sequencer names as keys with json filenames as values.
    # add bool option to program immediately?
    return config_dict


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
        if np.iscomplex(arr).any():
            I = arr.real
            Q = arr.imag
        else:
            I = arr
            Q = np.zeros(len(arr))
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


def check_pulse_long_enough(duration):
    return duration >= INSTRUCTION_CLOCK_TIME


def build_q1asm(ordered_operations, pulse_dict):
    """
    Converts operations and waveforms to a q1asm program. This function verifies these hardware based constraints:

        * Each pulse must run for at least the INSTRUCTION_CLOCK_TIME
        * Each operation must have a timing separation of at least INSTRUCTION_CLOCK_TIME

    .. warning:
        The above restrictions apply to any generated WAIT instructions.

    Args:
        ordered_operations (list): A sorted list of tuples matching timings to pulse_IDs.
        pulse_dict (dict): pulse_IDs to numerical waveforms with registered index in waveform memory.

    Returns:
        A q1asm program in a string.
    """
    rows = []
    rows.append(['start:', 'move', '{},R0'.format(len(pulse_dict)), '#Waveform count register'])

    previous = None  # previous pulse
    clock = 0  # current execution time
    labels = Counter()  # for unique labels, suffixed with a count in the case of repeats
    for timing, pulse_id in ordered_operations:
        # check each operation has the minimum required timing
        if previous and not check_pulse_long_enough(timing - previous[0]):
            raise ValueError("Timings '{}' and '{}' are too close (must be at least {}ns)"
                             .format(previous[0], timing, INSTRUCTION_CLOCK_TIME))

        # check if we must wait before beginning our next section
        wait_duration = timing - clock
        if previous and wait_duration > 0:
            # if the previous operation is not contiguous to the current, we must wait for period
            # check this period is at least the minimum time
            if not check_pulse_long_enough(wait_duration):
                previous_duration = len(pulse_dict['{}_I'.format(previous[1])]['data'])
                raise ValueError("Insufficient wait period between pulses '{}' and '{}' with timings '{}' and '{}'."
                                 "{} has a duration of {}ns necessitating a wait of duration {}ns "
                                 "(must be at least {}ns)."
                                 .format(previous[1], pulse_id, previous[0], timing, pulse_id, previous_duration,
                                         wait_duration, INSTRUCTION_CLOCK_TIME))
            rows.append(['', 'wait', '{}'.format(wait_duration), '#Wait'])

        I = pulse_dict["{}_I".format(pulse_id)]['index']
        Q = pulse_dict["{}_Q".format(pulse_id)]['index']

        rows.append(['', '', '', ''])
        label = '{}_{}'.format(pulse_id, labels[pulse_id])
        labels.update([pulse_id])
        duration = len(pulse_dict["{}_I".format(pulse_id)]['data'])  # duration in nanoseconds, QCM sample rate is 1Gsps
        # ensure pulse runs for at least the minimum time
        if not check_pulse_long_enough(duration):
            raise ValueError("Pulse '{}' at timing '{}' is too short (must be at least {}ns)"
                             .format(pulse_id, timing, INSTRUCTION_CLOCK_TIME))
        rows.append(['{}:'.format(label), 'play', '{},{},{}'.format(I, Q, duration), '#Play {}'.format(pulse_id)])
        previous = (timing, pulse_id)
        clock += duration

    table = columnar(rows, no_borders=True)
    return table


def generate_sequencer_cfg(pulse_info, pulse_timings):
    """
    Generate a JSON compatible dictionary for defining a sequencer configuration. Contains a list of waveforms and a
    program in a q1asm string

    Args:
        pulse_info (dict): mapping of pulse IDs to numerical waveforms
        pulse_timings (list): time ordered list of tuples containing the absolute starting time and pulse ID

    Returns:
        Sequencer configuration
    """
    cfg = build_waveform_dict(pulse_info)
    cfg['program'] = build_q1asm(pulse_timings, cfg['waveforms'])
    return cfg
