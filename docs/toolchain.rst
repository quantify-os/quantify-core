
==========
Toolchains
==========

Compilers
~~~~~~~~~

The compiler converts an input Schedule into a time ordered list of pulses. It is responsible for converting any higher
level abstractions (such as Gates) into these pulses. As such a compiler is specific to the type of quantum chip in use.

The compiler also resolves connections between resources, translating virtual addresses into physical ones.

Quantify currently supports:

- Transmons

Assembler Backends
~~~~~~~~~~~~~~~~~~

The backend converts the output of the compiler (a time ordered list of operations) into a program the control hardware
will run. This typically involves generating the requisite waveforms and an assembly language describing the program
itself.

Quantify currently supports:

- QBlox Pulsar range
