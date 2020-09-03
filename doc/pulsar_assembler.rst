
===============================
quantify assembler backend
===============================

.. note::

    This is part of the sequencer documentation that is under active development.

Here I document how the assembler backend should function.
I intend this documentation to evolve together with the implementation as we write it.

The starting point for the assembler is a :class:`~quantify.sequencer.Schedule` containing several operations.
The case we are considering does not include any "timeline-breaking" instructions such as hardware loops and feedback.
We do not include any kind of classical logic.

.. note::

    I envision timeline-breaking operations to be dealt with by allowing nesting of schedules.
    This idea needs to be worked out in more detail.

In order to compile this to something that can be executed on the hardware we need to add the follwing information

- gate to pulse info mapping.
- determine the timing constraints.
- information on the resources used.

To describe the compilation steps taken by the assembler I take a Schedule object from :ref:`Compilation onto a transmon backend`.



Compilation steps (psuedocode)
---------------------------------


.. code::

    sched =  add_pulse_information(sched)
    sched = determine_absolute_timing(sched)

    for all operation in schedule.timing_constraints:
        add operation to separate lists for each resource
        add pulses to pulse_dict per resource (similar to operation dict)

    for resource in resources:
        sort operation lists

    Convert the code for each resource to assembly

Problems:
- Need a clear definition of a "resource"


Resources
---------------------------------

The pulsar QCM has two fundamental types of resource.
The output (dac) channel, and the sequencer.
The user should specify what output channel to use.
The compiler backend should assign the different assemblers.

Looking at the input format, a sequencer resource should at least incluce information on what QCM instrument and which sequencer it uses.
We should probably also include some of the settings (like the modulation frequency) in this resource object.



Valid assembler input format
-------------------------------

The latest version of the demo by Jordy programs a QCM sequencer unit by passing a single JSON file (per sequencer) e.g., `qcm.sequencer0_waveforms_and_program` .

The contents of this file are
- A dictionary of waveforms
- A "program" , a string containing the assembly code.

We should make a proper JSON schema to validate and document this input format.

Of note are several things that are not included in this file but that are relevant.

.. code:: python

    qcm.sequencer0_cont_mode_en(False)        # Don't know what this does
    qcm.sequencer0_cont_mode_waveform_idx(0)  # Don't know what this does
    qcm.sequencer0_upsample_rate(0)           # Don't know what this does
    qcm.sequencer0_mod_enable(True)
    qcm.sequencer0_nco_freq(10e6)


