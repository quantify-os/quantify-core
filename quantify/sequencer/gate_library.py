"""
Library containing a standard gateset for use with the quantify sequencer.
"""

import numpy as np
from .types import Operation


class Rxy(Operation):
    """
    A single qubit rotation around an axis in the equator of the Bloch sphere.


    This operation can be represented by the following unitary:

    .. math::

        \\mathsf {R}_{xy} \\left(\\theta, \\varphi\\right) = \\begin{bmatrix}
        \\textrm {cos}(\\theta /2) & -ie^{-i\\varphi }\\textrm {sin}(\\theta /2)
        \\\\ -ie^{i\\varphi }\\textrm {sin}(\\theta /2) & \\textrm {cos}(\\theta /2) \\end{bmatrix}

    """

    def __init__(self, theta: float, phi: float, qubit: str):
        """
        A single qubit rotation around an axis in the equator of the Bloch sphere.

        Args:
            theta (float) : rotation angle in degrees
            phi (float)   : phase of the rotation axis
            qubit (str)   : the target qubit

        N.B. we now expect floats, no parametrized operations yet...
        """

        # data =

        name = ('Rxy({:.2f}, {:.2f}) {}'.format(theta, phi, qubit))

        theta_r = np.deg2rad(theta)
        phi_r = np.deg2rad(phi)

        # not all operations have a valid unitary description
        # (e.g., measure and init)
        unitary = np.array([
            [np.cos(theta_r/2), -1j*np.exp(-1j*phi_r)*np.sin(theta_r/2)],
            [-1j*np.exp(-1j*phi_r)*np.sin(theta_r/2), np.cos(theta_r/2)]])

        tex = r'$R_{xy}'+'({:.1f}, {:.1f})$'.format(theta, phi)
        data = {}
        data['name'] = name
        data['gate_info'] = {'unitary': unitary,
                             'tex': tex,
                             'qubits': [qubit],
                             'theta': theta,
                             'phi': phi}

        super().__init__(name, data=data)


class X(Rxy):
    """
    A single qubit rotation of 180 degrees around the X-axis.


    This operation can be represented by the following unitary:

    .. math::

        X = \\sigma_x = \\begin{bmatrix}
             0 & 1 \\\\
             1 & 0 \\ \\end{bmatrix}

    """

    def __init__(self, qubit: str):
        """
        Args:
            qubit (str): the target qubit
        """
        super().__init__(theta=180, phi=0, qubit=qubit)
        self.data['name'] = 'X {}'.format(qubit)
        self.data['gate_info']['tex'] = r'$X_{\pi}$'


class X90(Rxy):
    """
    A single qubit rotation of 90 degrees around the X-axis.
    """

    def __init__(self, qubit: str):
        """
        Args:
            qubit (str): the target qubit
        """
        super().__init__(theta=90, phi=0, qubit=qubit)
        self.data['name'] = 'X {}'.format(qubit)
        self.data['gate_info']['tex'] = r'$X_{\pi/2}$'


class Y(Rxy):
    """
    A single qubit rotation of 180 degrees around the Y-axis.


    .. math::

        \\mathsf Y = \\sigma_y = \\begin{bmatrix}
             0 & -i \\\\
             i & 0 \\end{bmatrix}

    """

    def __init__(self, qubit: str):
        """
        Args:
            qubit (str): the target qubit
        """
        super().__init__(theta=180, phi=90, qubit=qubit)
        self.data['name'] = 'Y_{90} {}'.format(qubit)
        self.data['gate_info']['tex'] = r'$Y_{\pi/2}$'


class Y90(Rxy):
    """
    A single qubit rotation of 90 degrees around the Y-axis.
    """

    def __init__(self, qubit: str):
        """
        Args:
            qubit (str): the target qubit
        """
        super().__init__(theta=90, phi=90, qubit=qubit)
        self.data['name'] = 'Y_{90} {}'.format(qubit)
        self.data['gate_info']['tex'] = r'$Y_{\pi/2}$'


class CNOT(Operation):
    """
    Conditional-NOT gate, a common entangling gate.

    Performs an X gate on the target qubit qT conditional on the state
    of the control qubit qC.

    This operation can be represented by the following unitary:

    .. math::

        \\mathrm{CNOT}  = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 0 & 1 & 0 \\ \\end{bmatrix}

    """

    def __init__(self, qC, qT):
        data = {}
        data['gate_info'] = {'unitary': np.array([1, 0, 0, 0],
                                                 [0, 1, 0, 0]
                                                 [0, 0, 0, 1]
                                                 [0, 0, 1, 0]),
                             'tex': r'CNOT',
                             'qubits': [qC, qT]}
        super().__init__('CNOT ({}, {})'.format(qC, qT), data=data)


class CZ(Operation):
    """
    Conditional-phase gate, a common entangling gate.

    Performs a Z gate on the target qubit qT conditional on the state
    of the control qubit qC.

    This operation can be represented by the following unitary:

    .. math::

        \\mathrm{CNOT}  = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & -1 \\ \\end{bmatrix}

    """

    def __init__(self, qC, qT):
        data = {}
        data['gate_info'] = {'unitary': np.array([1, 0, 0, 0],
                                                 [0, 1, 0, 0]
                                                 [0, 0, 1, 0]
                                                 [0, 0, 0, -1]),
                             'tex': r'CZ',
                             'qubits': [qC, qT]}
        super().__init__('CNOT ({}, {})'.format(qC, qT), data=data)


class Reset(Operation):
    """
    Reset a qubit to the :math:`|0\\rangle` state.

    .. note::
        strictly speaking this is not a gate as it can not
        be described by a unitary.

    """

    def __init__(self, *qubits):
        data = {}
        data['gate_info'] = {'unitary': None,
                             'tex': r'$|0\rangle$',
                             'qubits': qubits}

        super().__init__('Reset {}'.format(qubits), data=data)


class Measure(Operation):
    """
    A projective measurement in the Z-basis.

    .. note::
        strictly speaking this is not a gate as it can not
        be described by a unitary.
    """

    def __init__(self, *qubits):
        data = {}
        data['gate_info'] = {'unitary': None,
                             'tex': r'$\langle0|$',
                             'qubits': qubits}

        super().__init__('Measure {}'.format(qubits), data=None)
