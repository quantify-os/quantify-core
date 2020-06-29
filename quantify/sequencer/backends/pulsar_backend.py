import inspect
import logging
from columnar import columnar
from collections import Counter
import numpy as np
from quantify.utilities.general import make_hash, without, import_func_from_string


def pulsar_assembler_backend(schedule):
    """
    Create assembly input for multiple Qblox pulsar modules.

    Parameters
    ------------
    schedule : :class:`~quantify.sequencer.types.Schedule` :
        The schedule to convert into assembly.

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


    # Convert timing tuples and pulse dicts for each seqeuncer into assembly configs
    for ch in schedule.resources.values():
        if hasattr(ch, 'timing_tuples'):
            seq_cfg = generate_sequencer_cfg(
                pulse_info=ch.pulse_dict,
                pulse_timings=sorted(ch.timing_tuples))

            # TODO: write these seq_cfg to json files

            # TODO: configure the settings (modulation freq etc. for each seqeuncer)

    # returns a dict of sequencer names as keys with json filenames as values.
    # add bool option to program immediately?
    return {}

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


def build_q1asm(ordered_operations, pulse_dict):
    rows = []
    rows.append(['start:', 'move', '{},R0'.format(len(pulse_dict)), '#Waveform count register'])

    clock = 0  # current execution time
    labels = Counter()  # for unique labels, suffixed with a count in the case of repeats
    for timing, pulse_id in ordered_operations:
        I = pulse_dict["{}_I".format(pulse_id)]['index']
        Q = pulse_dict["{}_Q".format(pulse_id)]['index']
        # check if we must wait before beginning our next section
        if clock < timing:
            rows.append(['', 'wait', '{}'.format(timing - clock), '#Wait'])
        rows.append(['', '', '', ''])
        label = '{}_{}'.format(pulse_id, labels[pulse_id])
        labels.update([pulse_id])
        duration = len(pulse_dict["{}_I".format(pulse_id)]['data'])  # duration in nanoseconds, QCM sample rate is 1Gsps
        rows.append(['{}:'.format(label), 'play', '{},{},{}'.format(I, Q, duration), '#Play {}'.format(pulse_id)])
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
