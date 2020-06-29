from columnar import columnar
from collections import Counter
import numpy as np


def pulsar_assembler_backend(schedule):
    """
    Create assembly input for a Qblox pulsar module.

    Parameters
    ------------
    schedule : :class:`~quantify.sequencer.types.Schedule` :
        The schedule to convert into assembly.


    .. note::

        Currently only supports the Pulsar_QCM module.
        Does not yet support the Pulsar_QRM module.
    """

    timing_tuples_dict = {}  # keys correspond to resources

    for pls_idx, t_constr in enumerate(schedule.timing_constraints):
        op = schedule.operations[t_constr['operation_hash']]

        for p in op['pulse_info']:
            t0 = t_constr['abs_time']+p['t0']
            # Assumes the channel exists in the resources available to the schedule
            ch = schedule.resources[p['channel']]


            ch.pulse_dict
            ch.timing_tuples.append((t0, pulse_id))









    # This is the master function that calls the other ones

    # for all operation in schedule.timing_constraints:
    # add operation to separate lists for each resource
    # add pulses to pulse_dict per resource (similar to operation dict)

    # for resource in resources:
    #     sort operation lists

    # Convert the code for each resource to assembly

    # returns a dict of sequencer names as keys with json filenames as values.
    # add bool option to program immediately?
    pass


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
            sequencer_cfg["waveforms"]["{}_I".format(pulse_id)] = {
                "data": arr.real,
                "index": idx + idx_offset
            }
            idx_offset += 1
            sequencer_cfg["waveforms"]["{}_Q".format(pulse_id)] = {
                "data": arr.imag,
                "index": idx + idx_offset
            }
        else:
            sequencer_cfg["waveforms"][pulse_id] = {
                "data": data,
                "index": idx + idx_offset
            }
    return sequencer_cfg


def construct_q1asm_pulse_operations(ordered_operations, pulse_dict):
    rows = []
    rows.append(['start:', 'move', '{},R0'.format(len(pulse_dict)), '#Waveform count register'])

    clock = 0  # current execution time
    labels = Counter()  # for unique labels, suffixed with a count in the case of repeats
    for timing, operation in ordered_operations:
        pulse_idx = pulse_dict[operation.hash]['index']
        # check if we must wait before beginning our next section
        if clock < timing:
            rows.append(['', 'wait', '{}'.format(timing - clock), '#Wait'])
        rows.append(['', '', '', ''])
        label = '{}_{}'.format(operation.name, labels[operation.name])
        labels.update([operation.name])
        rows.append(['{}:'.format(label), 'play', '{},{},{}'.format(pulse_idx, pulse_idx, operation.duration),
                     '#Play {}'.format(operation.name)])
        clock += operation.duration

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
    cfg = prepare_waveforms_for_q1asm(pulse_info)
    cfg['program'] = construct_q1asm_pulse_operations(pulse_timings, cfg['waveforms'])
    return cfg
