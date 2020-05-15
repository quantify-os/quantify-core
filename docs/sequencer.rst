Sequencer
===============

The :mod:`quantify.sequencer` can be used to schedule operations on the control hardware.

The quantify seqeuncer is designed to allow access to low-level (hardware) functionally at the highest level of abstraction if required, while simultaneously allowing the user to ignore these aspects when this level of detail is not required.

The :mod:`quantify.sequencer` is build around the :class:`~quantify.sequencer.Schedule`, a data structure to which :class:`~quantify.sequencer.Operation` s are added with timing constraints.
An :class:`~quantify.sequencer.Operation` object contains information on how to represent the operation at the gate, pulse and/or instruction level as well as the :class:`~quantify.sequencer.Resource` (s) used.
When adding an :class:`~quantify.sequencer.Operation` to a :class:`~quantify.sequencer.Schedule`, the user is not expected to supply all this information at once.
This should be taken care of during the *compilation* steps.
Each compilation step is a transformation of the :class:`~quantify.sequencer.Schedule` and results in a new :class:`~quantify.sequencer.Schedule`.
Examples of valid compliation steps are ..... TODO
A final compilation step translates the :class:`~quantify.sequencer.Schedule` into a format compatible with the desired backend. VALID BACKENDS TODO


.. blockdiag::

    blockdiag sequencer {

      qf_input [label="Embedded  language"];
      ext_input [label="Q A S M-like\nformats", stacked];
      vis_bck [label="Visualization \nbackends", stacked];
      hw_bck [label="Hardware\nbackends", stacked];
      sim_bck [label="Simulator\nbackends", stacked];
      ext_fmts [label="Q A S M-like\n formats", stacked];


      qf_input, ext_input -> Schedule;
      Schedule -> Schedule [label="Compile"];
      Schedule -> vis_bck;
      Schedule -> hw_bck;
      Schedule -> sim_bck ;
      Schedule -> ext_fmts;

      group {
        label= "Input formats";
        qf_input
        ext_input
        color="#90EE90"
        }


      group {

        Schedule
        color=red
        label="Compilation"
        }

      group {
        label = "Backends";
        color = orange;
        vis_bck, hw_bck, sim_bck, ext_fmts
        }
    }


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