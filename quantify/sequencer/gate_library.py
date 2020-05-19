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

        \\mathsf {R}_{xy} \\left(\\theta, \\varphi\\right) = \\begin{bmatrix}\\textrm {cos}(\\theta /2) & -ie^{-i\\varphi }\\textrm {sin}(\\theta /2) \\\\ -ie^{i\\varphi }\\textrm {sin}(\\theta /2) & \\textrm {cos}(\\theta /2) \\end{bmatrix}

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
        unitary = np.matrix([
            [np.cos(theta_r/2), -1j*np.exp(-1j*phi_r)*np.sin(theta_r/2)],
            [-1j*np.exp(-1j*phi_r)*np.sin(theta_r/2), np.cos(theta_r/2)]])

        tex = r'$R_{xy}'+'({:.1f}, {:.1f})$'.format(theta, phi)
        data = {}
        data['name'] = name
        data['gate_info'] = {'unitary': unitary,
                             'tex': tex,
                             'qubits': [qubit]}

        super().__init__(name, data=data)


class X(Rxy):
    """
    A single qubit rotation of 180 degrees around the X-axis.
    """
    def __init__(self, qubit: str):
        """
        Args:
            qubit (str): the target qubit
        """
        super().__init__(theta=180, phi=0, qubit=qubit)
        self.data['name'] = 'X {}'.format(qubit)
        self.data['gate_info']['tex'] = '$X_{\pi}$'


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
        self.data['gate_info']['tex'] = '$X_{\pi/2}$'


class Y(Rxy):
    """
    A single qubit rotation of 180 degrees around the Y-axis.
    """
    def __init__(self, qubit: str):
        """
        Args:
            qubit (str): the target qubit
        """
        super().__init__(theta=180, phi=90, qubit=qubit)
        self.data['name'] = 'Y_{90} {}'.format(qubit)
        self.data['gate_info']['tex'] = '$Y_{\pi/2}$'


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
        self.data['gate_info']['tex'] = '$Y_{\pi/2}$'


class CNOT(Operation):
    """
    CNOT
    """

    def __init__(self, qC, qT):
        super().__init__('CNOT ({}, {})'.format(qC, qT), data=None)


class Reset(Operation):
    """

    """

    def __init__(self, *qubits):
        super().__init__('Reset {}'.format(qubits), data=None)


class Measure(Operation):
    """
    A projective measurement in the Z-basis.

    N.B. strictly speaking this is not a gate type operation.
    """

    def __init__(self, *qubits):
        super().__init__('Measure {}'.format(qubits), data=None)

