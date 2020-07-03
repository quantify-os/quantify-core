"""
This module contains compilation steps for the quantify sequencer.

A compilation step is a function that takes a :class:`~quantify.sequencer.types.Schedule`
and returns a new (modified) :class:`~quantify.sequencer.types.Schedule`.
"""
import logging
import jsonschema
from scipy.signal import hanning
from quantify.sequencer.pulse_library import ModSquarePulse, DRAGPulse, IdlePulse, SquarePulse
from quantify.sequencer.windows import Hanning
from quantify.utilities.general import load_json_schema


def determine_absolute_timing(schedule, clock_unit='physical'):
    """
    Determines the absolute timing of a schedule based on the timing constraints.

    Parameters
    ----------
    schedule : :class:`~quantify.sequencer.Schedule`
        The schedule for which to determine timings.
    clock_unit : str
        Must be ('physical', 'ideal') : whether to use physical units to determine the
        absolute time or ideal time.
        When clock_unit == "physical" the duration attribute is used.
        When clock_unit == "ideal" the duration attribute is ignored and treated as if it is 1.


    Returns
    ----------
    schedule : :class:`~quantify.sequencer.Schedule`
        a new schedule object where the absolute time for each operation has been determined.


    This function determines absolute timings for every operation in the
    :attr:`~quantify.sequencer.Schedule.timing_constraints`. It does this by:

        1. iterating over all and elements in the timing_constraints.
        2. determining the absolute time of the reference operation.
        3. determining the of the start of the operation based on the rel_time and duration of operations.

    """

    # iterate over the objects in the schedule.
    last_constr = schedule.timing_constraints[0]
    last_op = schedule.operations[last_constr['operation_hash']]

    last_constr['abs_time'] = 0

    # 1. loop over all operations in the schedule and
    for t_constr in schedule.data['timing_constraints'][1:]:
        curr_op = schedule.operations[t_constr['operation_hash']]
        if t_constr['ref_op'] is None:
            ref_constr = last_constr
            ref_op = last_op
        else:
            # this assumes the reference op exists. This is ensured in schedule.add
            ref_constr = next(item for item in schedule.timing_constraints if item['label'] == t_constr['ref_op'])
            ref_op = schedule.operations[ref_constr['operation_hash']]

        # duration = 1 is useful when e.g., drawing a circuit diagram.
        duration_ref_op = ref_op.duration if clock_unit == 'physical' else 1

        # determine
        if t_constr['ref_pt'] == 'start':
            t0 = ref_constr['abs_time']
        elif t_constr['ref_pt'] == 'center':
            t0 = ref_constr['abs_time'] + duration_ref_op/2
        elif t_constr['ref_pt'] == 'end':
            t0 = ref_constr['abs_time'] + duration_ref_op
        else:
            raise NotImplementedError(
                'Timing "{}" not supported by backend'.format(ref_constr['abs_time']))

        duration_new_op = curr_op.duration if clock_unit == 'physical' else 1

        if t_constr['ref_pt_new'] == 'start':
            t_constr['abs_time'] = t0 + t_constr['rel_time']
        elif t_constr['ref_pt_new'] == 'center':
            t_constr['abs_time'] = t0 + \
                t_constr['rel_time'] - duration_new_op/2
        elif t_constr['ref_pt_new'] == 'end':
            t_constr['abs_time'] = t0 + t_constr['rel_time'] - duration_new_op

        # update last_constraint and operation for next iteration of the loop
        last_constr = t_constr
        last_op = curr_op

    return schedule


def add_pulse_information_transmon(schedule, device_cfg: dict):
    """
    Adds pulse information specified in the device config to the schedule.

    Parameters
    ------------
    schedule : :class:`~quantify.sequencer.Schedule`
        The schedule for which to add pulse information.

    device_cfg: dict
        A dictionary specifying the required pulse information.
        The device_cfg schema is specified in `sequencer/schemas/transmon_cfg.json` see also below.


    Returns
    ----------
    schedule : :class:`~quantify.sequencer.Schedule`
        a new schedule object where the pulse information has been added.


    .. rubric:: Supported operations


    The following gate type operations are supported by this compilation step.

    - :class:`~quantify.sequencer.gate_library.Rxy`
    - :class:`~quantify.sequencer.gate_library.Reset`
    - :class:`~quantify.sequencer.gate_library.Measure`
    - :class:`~quantify.sequencer.gate_library.CZ`


    .. rubric:: Configuration specification

    .. jsonschema:: schemas/transmon_cfg.json

    """
    validate_config(device_cfg, scheme_fn='transmon_cfg.json')

    for op in schedule.operations.values():
        if op['gate_info']['operation_type'] == 'measure':
            for q in op['gate_info']['qubits']:
                q_cfg = device_cfg['qubits'][q]
                # readout pulse
                if q_cfg['ro_pulse_type'] == 'square':
                    op.add_pulse(ModSquarePulse(amp=q_cfg['ro_pulse_amp'],
                                                duration=q_cfg['ro_pulse_duration'],
                                                ch=q_cfg['ro_pulse_ch'],
                                                freq_mod=q_cfg['ro_pulse_modulation_freq'],
                                                t0=0))
                    # acquisition integration window
                    op.add_pulse(ModSquarePulse(amp=1,
                                                duration=q_cfg['ro_acq_integration_time'],
                                                ch=q_cfg['ro_acq_ch'],
                                                freq_mod=-q_cfg['ro_pulse_modulation_freq'],
                                                t0=q_cfg['ro_acq_delay']))

        elif op['gate_info']['operation_type'] == 'Rxy':
            q = op['gate_info']['qubits'][0]
            # read info from config
            q_cfg = device_cfg['qubits'][q]

            G_amp = q_cfg['mw_amp180']*op['gate_info']['theta'] / 180
            D_amp = G_amp * q_cfg['mw_motzoi']

            pulse = DRAGPulse(
                G_amp=G_amp, D_amp=D_amp, phase=op['gate_info']['phi'],
                ch=q_cfg['mw_ch'],  duration=q_cfg['mw_duration'],
                freq_mod=q_cfg['mw_modulation_freq'])
            op.add_pulse(pulse)

        elif op['gate_info']['operation_type'] == 'CNOT':
            # These methods don't raise exceptions as they will be implemented shortly
            logging.warning("Not Implemented yet")
            logging.warning('Operation type "{}" not supported by backend'.format(op['gate_info']['operation_type']))

        elif op['gate_info']['operation_type'] == 'CZ':
            q0 = op['gate_info']['qubits'][0]
            q1 = op['gate_info']['qubits'][1]
            q0_cfg = device_cfg['qubits'][q0]
            q1_cfg = device_cfg['qubits'][q1]

            amp = q0_cfg['mw_amp180']

            pulse = SquarePulse(amp=amp, duration=q0_cfg['mw_duration'], ch=q0_cfg['mw_ch'])
            pulse.add_filter(Hanning(2))
            op.add_pulse(pulse)
        elif op['gate_info']['operation_type'] == 'reset':
            # Initialization through relaxation
            qubits = op['gate_info']['qubits']
            init_times = []
            for q in qubits:
                init_times.append(device_cfg['qubits'][q]['init_duration'])
            op.add_pulse(IdlePulse(max(init_times)))

        else:
            raise NotImplementedError('Operation type "{}" not supported by backend'
                                      .format(op['gate_info']['operation_type']))

    return schedule


def validate_config(config: dict, scheme_fn: str):
    """
    Validate a configuration using a schema.

    Parameters
    ------------
    config : dict
        The configuration to validate
    scheme_fn : str
        The name of a json schema in the quantify.sequencer.schemas folder.

    Returns
    ----------
        valid : bool

    """
    scheme = load_json_schema(__file__, scheme_fn)
    jsonschema.validate(config, scheme)
    return True
