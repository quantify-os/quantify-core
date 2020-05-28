"""
This module contains compilation step for the quantify sequencer.

A compilation step is a function that takes a :class:`~quantify.sequencer.types.Schedule`
and returns a new (modified) :class:`~quantify.sequencer.types.Schedule`.
"""


def determine_absolute_timing(schedule, clock_unit='physical'):
    """
    Determines the absolute timing of a schedule based on the timing constraints.

    Parameters
    ----------
    schedule : :class:`~quantify.sequencer.Schedule`
        The schedule for which to determine timings.
    clock_unit : str
        Must be ('physical', 'ideal') : wheter to use physical units to determine the
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
            ref_constr = [item for item in schedule.timing_constraints
                          if item['label'] == t_constr['ref_op']][0]
            ref_op = schedule.operations[ref_constr['operation_hash']]

        # duration = 1 is useful when e.g., drawing a circuit diagram.
        duration_ref_op = ref_op['duration'] if clock_unit == 'physical' else 1

        # determine
        if t_constr['ref_pt'] == 'start':
            t0 = ref_constr['abs_time']
        elif t_constr['ref_pt'] == 'center':
            t0 = ref_constr['abs_time'] + duration_ref_op/2
        elif t_constr['ref_pt'] == 'end':
            t0 = ref_constr['abs_time'] + duration_ref_op

        duration_new_op = curr_op['duration'] if clock_unit == 'physical' else 1

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
