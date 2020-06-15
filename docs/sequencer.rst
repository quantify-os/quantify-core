Sequencer
===============

.. warning::

  Under active development!


The :mod:`quantify.sequencer` can be used to schedule operations on the control hardware.
The :mod:`quantify.sequencer` is designed to allow access to low-level (hardware) functionally at the highest level of abstraction if required, while simultaneously allowing the user to ignore these aspects when this level of detail is not required.

The :mod:`quantify.sequencer` is build around the :class:`~quantify.sequencer.Schedule`, a data structure containing :attr:`~quantify.sequencer.Schedule.operations` , :attr:`~quantify.sequencer.Schedule.timing_constraints` , and :attr:`~quantify.sequencer.Schedule.resources` .

An :class:`~quantify.sequencer.Operation` contains information on how to *represent* the operation at the gate, pulse and/or instruction level as well as the :class:`~quantify.sequencer.Resource` (s) required.
When adding an :class:`~quantify.sequencer.Operation` to a :class:`~quantify.sequencer.Schedule`, the user is not expected to provide this information at once.
Only when specific information is required by a backend such as a simulator or a hardware backend does the information need to be provided.

A compilation step is a transformation of the :class:`~quantify.sequencer.Schedule` and results in a new :class:`~quantify.sequencer.Schedule`.
A compilation step can be used to e.g., add pulse information to operations containing only a gate-level representation or to determine the absolute timing based on timing constraints.
A final compilation step translates the :class:`~quantify.sequencer.Schedule` into a format compatible with the desired backend.


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

.. code-block:: python

    version 1.0

    # Bell experiment

    qubits 2

    .init
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

    from quantify.sequencer.gate_library import Reset, Measure, CNOT, Rxy, X90

    # Define the operations, these will be added to the circuit
    init_all = Reset(q0, q1) # instantiates
    x90_q0 = Rxy(theta=90, phi=0, qubit=q0)
    cnot = CNOT(qC=q0, qT= q1)
    Rxy_theta = Rxy(theta=23, phi=0, qubit=q0) # will be not be used in the experiment loop.
    meass_all = Measure(q0, q1)


Similar to the schedule, :class:`~quantify.sequencer.Operation` are also based on dicts.


.. jupyter-execute::

    # Rxy_theta  # produces the same output
    Rxy_theta.data


Now we create the Bell experiment, including observing the oscillation in a simple for loop.

.. jupyter-execute::

    import numpy as np

    # we use a regular for loop as we have to unroll the changing theta variable here
    for theta in np.linspace(0, 360, 21):
        sched.add(init_all)
        sched.add(x90_q0)
        sched.add(operation=CNOT(qC=q0, qT= q1))
        sched.add(Rxy(theta=theta, phi=0, qubit=q0))
        sched.add(Measure(q0, q1), label='M {:.2f} deg'.format(theta))


.. note::

  This experiment should also be wrapped in a "Quantum loop" with a symbolic variable to set the loop counter and determine the number of averages. (not implemented yet).
  Making that variable hardware controllable is interesting to include in our high level description in an elegant way.
  It depends a bit on how this would work in the hardware (using a register to set the number of loops) how we want to represent this in the sequencer.
  Intuitively this feels like a concept that would allow super awesome variational algorithms.


Let's take a look at the internals of the :class:`~quantify.sequencer.Schedule`.

.. jupyter-execute::

    sched

We can see that the number of unique operations is 24 corresponding to 4 operations that occur in every loop and 21 unique rotations for the different theta angles. (21+4 = 25 so we are missing something.





.. jupyter-execute::

    sched.data.keys()


The schedule consists of a hash table containing all the operations.
This allows effecient loading of pulses or gates to memory and also enables efficient adding of pulse type information as a compilation step.

.. jupyter-execute::

    from itertools import islice
    # showing the first 5 elements of the operation dict
    dict(islice(sched.data['operation_dict'].items(), 5))

The timing constraints are stored as a list of pulses.
Because

.. jupyter-execute::

  sched.data['timing_constraints'][:6]

Turning the constraints into a timed experiment, would require iterating over all elements in the timing constraints list.
This is identical to how the pycqed pulsar works.
Compilation efficiency is not an issue for "small" experiments but will be something we encounter in the future.


Creating a circuit diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far we have only defined timing constraints.
Duration of pulses not known, but can create circuit diagram.

For this purpose we do our first compilation step.

.. jupyter-execute::

  from quantify.sequencer.compilation import determine_absolute_timing
  # We modify the schedule in place adding timing information
  # setting clock_unit='ideal' ignores the duration of operations and sets it to 1.
  determine_absolute_timing(sched, clock_unit='ideal')

And we can use this to create a default visualizaton.

.. jupyter-execute::

  %matplotlib inline

  from quantify.sequencer import backends
  f, ax = backends.circuit_diagram_matplotlib(sched)
  # all gates are plotted, but it doesn't all fit in a matplotlib figure
  ax.set_xlim(-.5, 9.5)

Compilation onto a transmon backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. jupyter-execute::

  device_test_cfg = {
      'qubits':
      {
          'q0': {'mw_amp180': .75, 'mw_motzoi': -.25, 'mw_duration': 20e-9,
                 'mw_modulation_freq': 50e6, 'mw_ef_amp180': .87, 'mw_ch_I': 'ch0', 'mw_ch_Q': 'ch1',
                 'ro_pulse_ch_I': 'ch5.0', 'ro_pulse_ch_Q': 'ch6.0', 'ro_pulse_amp': .5, 'ro_pulse_modulation_freq': 80e6,
                 'ro_pulse_type': 'square', 'ro_pulse_duration': 150e-9,
                 'ro_acq_ch_I': 'acq_ch1', 'ro_acq_ch_Q': 'acq_ch2', 'ro_acq_delay': 120e-9, 'ro_acq_integration_time': 700e-9,
                 'ro_acq_weigth_type': 'SSB',
                 'init_duration': 250e-6,
                 },

          'q1': {'mw_amp180': .45, 'mw_motzoi': -.15, 'mw_duration': 20e-9,
                 'mw_modulation_freq': 80e6, 'mw_ef_amp180': .27, 'mw_ch_I': 'ch2', 'mw_ch_Q': 'ch3',
                 'ro_pulse_ch_I': 'ch5.1', 'ro_pulse_ch_Q': 'ch6.1', 'ro_pulse_amp': .5, 'ro_pulse_modulation_freq': -23e6,
                 'ro_pulse_type': 'square', 'ro_pulse_duration': 100e-9,
                 'ro_acq_ch_I': 'acq_ch1', 'ro_acq_ch_Q': 'acq_ch2', 'ro_acq_delay': 120e-9, 'ro_acq_integration_time': 700e-9,
                 'ro_acq_weigth_type': 'SSB',
                 'init_duration': 250e-6, }
      },
      'edges':
      {
      }
  }



Compilation is happening here

.. jupyter-execute::

  from quantify.sequencer.compilation import add_pulse_information_transmon
  sched = add_pulse_information_transmon(sched, device_test_cfg)
  sched = determine_absolute_timing(sched)



And here we plot the resulting experiment using plotly


.. jupyter-execute::

  from quantify.sequencer.backends import pulse_diagram_plotly
  fig = pulse_diagram_plotly(sched)
  fig.show()


By default :func:`quantify.sequencer.backends.pulse_diagram_plotly` shows the first 8 channels encountered in in a schedule, but by specifying a list of channels, a more compact visualization can be created.


.. jupyter-execute::


  fig = pulse_diagram_plotly(sched, ch_list=['ch0', 'ch5.0', 'ch6.0', 'acq_ch1'])
  fig.show()



.. note::

  This is it for now! Let's discuss.



Bell violation circuit (change angle)

    - [x] Show input how to create
    - [x] Visualization circuit diagram
    - [x] Visualization pulse sequence (waveforms per channel)

Chevron experiment

    - Show input how to create