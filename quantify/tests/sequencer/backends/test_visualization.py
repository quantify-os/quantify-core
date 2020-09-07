from quantify.sequencer.backends.visualization import pulse_diagram_plotly, circuit_diagram_matplotlib
from quantify.sequencer import Schedule
from quantify.sequencer.gate_library import Reset, Measure, CZ, CNOT, Rxy
from quantify.sequencer.compilation import _determine_absolute_timing, qcompile
import matplotlib.pyplot as plt
import json

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from ... import test_data  # relative-import the *package* containing the templates


DEVICE_TEST_CFG = json.loads(pkg_resources.read_text(test_data, 'transmon_test_config.json'))


def test_circuit_diagram_matplotlib():
    sched = Schedule('Test experiment')

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ('q0', 'q1')
    ref_label_1 = 'my_label'

    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(operation=CNOT(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')

    sched = _determine_absolute_timing(sched, clock_unit='ideal')
    f, ax = circuit_diagram_matplotlib(sched)


def test_pulse_diagram_plotly():
    sched = Schedule('Test schedule')

    # define the resources
    q0, q1 = ('q0', 'q1')
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    # sched.add(operation=CZ(qC=q0, qT=q1)) # not implemented in config
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')
    # pulse information is added
    sched = qcompile(sched, DEVICE_TEST_CFG, None)

    # It should be possible to generate this visualization after compilation
    fig = pulse_diagram_plotly(sched, ch_list=["qcm0.s0", "qrm0.s0", "qrm0.r0", "qrm0.s1", "qrm0.r1"])
    # and with auto labels
    fig = pulse_diagram_plotly(sched)
