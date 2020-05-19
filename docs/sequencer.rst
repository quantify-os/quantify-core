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

Example the Bell experiment
----------------------------------------

As the first example, we want to perform the  `Bell experiment <https://en.wikipedia.org/wiki/Bell%27s_theorem>`_ .
In this example, we will go quite deep into the internals of the sequencer to show how the data strutures work.

The goal of the Bell experiment is to create a Bell state :math:`|\Phi ^+\rangle=\frac{1}{2}(|00\rangle+|11\rangle)` followed by a measurement and observe violations of the CSHS inequality.

By changing the basis in one which one of the detectors measures, we can observe an oscillation which should result in a violation of Bell's inequality.
If everything is done properly, one should observe this oscillation:

.. figure:: https://upload.wikimedia.org/wikipedia/commons/e/e2/Bell.svg
  :figwidth: 50%





Bell circuit
~~~~~~~~~~~~~~~~
Below is the QASM code used to perform this experiment in the `Quantum Inspire <http://>`_  [quantum inspire](https://www.quantum-inspire.com/) and a circuit diagram representation.
We will be creating this same experiment using the Quantify sequencer.

.. code::

    version 1.0

    # Bell experiment

    qubits 2

    .Init
    prep_z q[0:1]


    .Entangle
    X90 q[0]
    cnot q[0],q[1]

    .Rotate
    # change the value to change the basis of the detector
    Rx q[0], 0.15

    .Measurement
    Measure_all


.. figure:: /figures/bell_circuit_QI.png
  :figwidth: 50%


Creating the experiment using the quantify sequencer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start by initializing an empty :class:`~quantify.sequencer.Schedule`

.. jupyter-execute::

  from quantify.sequencer import Schedule
  sched = Schedule('Bell experiment')
  sched

Under the hood, the :class:`~quantify.sequencer.Schedule` is based on a dictionary that can be serialized

.. jupyter-execute::

  sched.data

We also need to define the resources. For now these are just strings because I have not implemented them properly yet.

.. jupyter-execute::

  # define the resources
  # q0, q1 = Qubits(n=2) # assumes all to all connectivity
  q0, q1 = ('q0', 'q1') # we use strings because Resources have not been implemented yet


We will now add some operations to the schedule.
Because this experiment is most conveniently described on the gate level, we use operations defined in the :mod:`quantify.sequencer.gate_library` .


.. jupyter-execute::

  import quantify.sequencer.gate_library as gl
  # Define the operations, these will be added to the circuit
  init_all = gl.Reset(q0, q1)
  x90_0 = gl.X90(q0)
  cnot = gl.CNOT(qC=q0, qT= q1)
  Rxy_theta = gl.Rxy(theta=23, phi=0, qubit=q0) # this will be replaced in our loop

Similar to the schedule, :class:`~quantify.sequencer.Operation` are also based on dicts.


.. jupyter-execute::
    # Rxy_theta  # produces the same output
    Rxy_theta.data


Now we create the Bell experiment, including observing the oscillation in a simple for loop.



Bell violation circuit (change angle)
    - Show input how to create
    - Visualization circuit diagram
    - Visualization pulse sequence (waveforms per channel)
    - Visualization combined
    - Show underlying data structures

Chevron experiment
    - Show input how to create