=========
Schedules
=========

The fundamental type of the scheduler is the Schedule. The Schedule contains Operations and Resources, which together
describe *what* should be done *where*. Operations and Resources are added to the schedule with ordering operators. The
schedule is then compiled, producing a set of configuration files to be uploaded to the control hardware.

Operation
~~~~~~~~~

An Operation is an action to be taken by the control hardware, typically playing a pulse. Operations at the Gate level
of abstraction are Gates and Operations at the Pulse level of abstraction are pulses. These Operations also typically
contain parameterisation information, such as amplitudes, durations and so forth.

*Gates* operate on qubits. A single gate often consists of multiple pulses to various ports on the Qubit.

*Pulses* operate on ports. Ports are typically attached to qubits but can also be attached to other devices.

Resource
~~~~~~~~

A Resource is a device accessible to the control hardware. This could be an output channel of the hardware, a register
on the device or even a qubit itself.

Compilation
~~~~~~~~~~~

To compile a program, the user must specify some configuration on:

- the type and design of quantum chip
- the control hardware

*Timing is determined*
Operations in the schedule are defined relatively. This step converts those relative timings into clock time ordered
operations.

*Gates are translated into Pulses*
The Compiler then converts Gates into their equivalent pulse representation.

*Pulses are translated into Waveforms*
The Backend then converts these pulses into waveforms and generates QASM like code for triggering them.

*Waveforms are distributed to Resources*
The set of generated programs are uploaded to the control hardware.

**For a practical example, check out the provided tutorials!**
