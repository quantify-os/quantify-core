Sequencer
===============

The :mod:`quantify.sequencer` can be used to schedule operations on the control hardware.

The quantify seqeuncer is designed to allow access to low-level (hardware) functionally at the highest level of abstraction if required, while simultaneously allowing the user to ignore these aspects when this level of detail is not required.

The :mod:`quantify.sequencer` is build around the :class:`~quantify.sequencer.Schedule`, a data structure to which :class:`~quantify.sequencer.Operation` s are added with timing constraints.
An :class:`~quantify.sequencer.Operation` object contains information on how to represent the operation at the gate, pulse and/or instruction level as well as the :class:`~quantify.sequencer.Resource` (s) used.
When adding an :class:`~quantify.sequencer.Operation` to a :class:`~quantify.sequencer.Schedule`, the user is not expected to supply all this information at once.
This should be taken care of during the *compilation* steps.

A compilation step is a transformation of the :class:`~quantify.sequencer.Schedule` (add figure from presentation here).

The benefit of allowing the user to mix the high-level gate description of a circuit with the lower-level pulse description can be understood through an example.
Below we first give an example of basic usage using `Bell violations`.
We next show the `Chevron` experiment in which the user is required to mix gate-type and pulse-type information when define the :class:`~quantify.sequencer.Schedule`.


Example circuit diagram -> a visual representation of a schedule using the gate-type information.
Pulse sequence -> a visual representation of a schedule using the pulse-type information.
Key idea is to provide access at highest level

Examples
-----------



Bell violation circuit (change angle)
    - Show input how to create
    - Visualization circuit diagram
    - Visualization pulse sequence (waveforms per channel)
    - Visualization combined
    - Show underlying data structures

Chevron experiment
    - Show input how to create