

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
