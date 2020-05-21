"""
Backends for the quantify sequencer.

A backend takes a :class:`~quantify.sequencer.types.Schedule` object as input
and produces output in a different format.
Examples of backends are a visualization, simulator input formats, or a hardware input format.
"""
from quantify.visualization.pulse_scheme import new_pulse_fig
from quantify.utilities.general import import_func_from_string


def circuit_diagram_matplotlib(schedule, figsize=None):
    # qubit map should be obtained from the schedule object
    qubit_map = {'q0': 0, 'q1': 1}

    qubits = ('q0', 'q1')

    if figsize is None:
        figsize = (10, len(qubit_map))
    f, ax = new_pulse_fig(figsize=(10, 1.5))
    ax.set_title(schedule.data['name'])
    ax.set_aspect('equal')

    ax.set_ylim(-.5, len(qubit_map)-.5)
    for q in qubits:
        ax.axhline(qubit_map[q], color='.75')

    for t_constr in schedule.timing_constraints:
        op = schedule.operations[t_constr['operation_hash']]
        plot_func = import_func_from_string(op['gate_info']['plot_func'])
        time = t_constr['abs_time']
        idxs = [qubit_map[q] for q in op['gate_info']['qubits']]
        plot_func(ax, time=time, qubit_idxs=idxs, tex=op['gate_info']['tex'])

    ax.set_xlim(-.2, t_constr['abs_time']+1)

    return f, ax
