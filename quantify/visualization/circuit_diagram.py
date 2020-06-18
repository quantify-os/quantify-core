"""
Plotting functions used in the visualization backend of the sequencer.

Mostly thin wrappers around the functions in :mod:`~quantify.visualization.pulse_scheme`
with default values to ensure that they match the required call signature.
"""

import quantify.visualization.pulse_scheme as ps


def gate_box(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    A box for a single gate containing a label.
    """

    for qubit_idx in qubit_idxs:
        ps.box_text(ax, x0=time, y0=qubit_idx, text=tex,
                    fillcolor='C0', w=.8, h=.5, **kw)


def meter(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    A simple meter to depict a measurement.
    """

    for qubit_idx in qubit_idxs:
        ps.meter(ax, x0=time, y0=qubit_idx,
                 fillcolor='C4', y_offs=0, w=.8, h=.5, **kw)


def cnot(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    Markers to denote a CNOT gate between two qubits.
    """
    ax.plot([time, time], qubit_idxs, marker='o',
            markersize=15, color='C1')
    ax.plot([time], qubit_idxs[1], marker='+',
            markersize=12, color='white')


def cz(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    Markers to denote a CZ gate between two qubits.
    """
    ax.plot([time, time], qubit_idxs, marker='o',
            markersize=15, color='C1')


def reset(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    A broken line to denote qubit initialization.
    """

    for qubit_idx in qubit_idxs:
        ps.box_text(ax, x0=time, y0=qubit_idx, text=tex,
                    color='white',
                    fillcolor='white', w=.4, h=.5, **kw)
