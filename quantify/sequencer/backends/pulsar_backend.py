from columnar import columnar
from collections import Counter


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

    # This is the master function that calls the other ones

    # for all operation in schedule.timing_constraints:
    # add operation to separate lists for each resource
    # add pulses to pulse_dict per resource (similar to operation dict)

    # for resource in resources:
    #     sort operation lists

    # Convert the code for each resource to assembly
    pass


def prepare_waveforms_for_q1asm(pulse_info):
    sequencer_cfg = {"waveforms": {}, "program": ""}
    for idx, (pulse, data) in enumerate(pulse_info.items()):
        sequencer_cfg["waveforms"][pulse] = {
            "data": data,
            "index": idx
        }
    return sequencer_cfg


def construct_q1asm_pulse_operations(ordered_operations, pulse_dict):
    rows = []
    rows.append(['start:', 'move', '{},R0'.format(len(pulse_dict)), '#Waveform count register'])

    clock = 0  # current execution time
    labels = Counter()  # for unique labels, suffixed with a count in the case of repeats
    for timing, operation in ordered_operations:
        pulse_idx = pulse_dict[operation.name]['index']
        # check if we must wait before beginning our next section
        if clock < timing:
            rows.append(['', 'wait', '{}'.format(timing - clock), '#Wait'])
        rows.append(['', '', '', ''])
        label = '{}_{}'.format(operation.name, labels[operation.name])
        labels.update([operation.name])
        rows.append(['{}:'.format(label), 'play', '{},{}'.format(pulse_idx, operation.duration), '#Play {}'.format(operation.name)])
        clock += operation.duration

    table = columnar(rows, no_borders=True)
    return table


def generate_sequencer_cfg(pulse_info, pulse_timings):
    """
    Needs docstring
    """
    top_level = prepare_waveforms_for_q1asm(pulse_info)
    program_str = construct_q1asm_pulse_operations(pulse_timings, top_level['waveforms'])
    top_level['program'] = program_str
    return top_level
