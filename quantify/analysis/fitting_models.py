# -----------------------------------------------------------------------------
# Description:    Module containing fitting models.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
"""Models and fit functions to be used with the lmfit fitting framework."""
from __future__ import annotations

import numpy as np
import lmfit


def get_model_common_doc() -> str:
    """
    Returns a common docstring to be used with custom fitting
    :class:`~lmfit.model.Model` s.

    .. admonition:: Usage example for a custom fitting model
        :class: dropdown, tip

        See the usage example at the end of the :class:`~ResonatorModel` source-code:

        .. literalinclude:: ../quantify/analysis/fitting_models.py
            :pyobject: ResonatorModel
    """
    return (
        lmfit.models.COMMON_DOC.replace(":class:`Model`", ":class:`~lmfit.model.Model`")
        .replace("\n    ", "\n")
        .replace(" : str", " : :obj:`str`")
        .replace("['x']", ":code:`['x']`")
        .replace(", optional", "")
        .replace(" optional", "")
        .replace("{'raise', 'propagate', 'omit'}", "")
    )


def get_guess_common_doc() -> str:
    """
    Returns a common docstring to be used for the :meth:`~lmfit.model.Model.guess`
    method of custom fitting :class:`~lmfit.model.Model` s.

    .. admonition:: Usage example for a custom fitting model
        :class: dropdown, tip

        See the usage example at the end of the :class:`~ResonatorModel` source-code:

        .. literalinclude:: ../quantify/analysis/fitting_models.py
            :pyobject: ResonatorModel
    """
    return (
        lmfit.models.COMMON_GUESS_DOC.replace(
            " : Parameters", " : :class:`~lmfit.parameter.Parameters`"
        )
        .replace("\n    ", "\n")
        .replace(" optional", "")
        .replace(" : array_like", " : :class:`~numpy.ndarray`")
    )


def mk_seealso(function_name: str, role: str = "func", prefix: str = "\n\n") -> str:
    """
    Returns a sphinx `seealso` pointing to a function.

    Intended to be used for building custom fitting model docstrings.

    .. admonition:: Usage example for a custom fitting model
        :class: dropdown, tip

        See the usage example at the end of the :class:`~ResonatorModel` source-code:

        .. literalinclude:: ../quantify/analysis/fitting_models.py
            :pyobject: ResonatorModel

    Parameters
    ----------
    function_name
        name of the function to point to
    role
        a sphinx role, e.g. :code:`"func"`
    prefix
        string preceding the `seealso`

    Returns
    -------
    :
        resulting string
    """
    return f"{prefix}.. seealso:: :{role}:`~.{function_name}`\n"


def hanger_func_complex_SI(
    f: float,
    fr: float,
    Ql: float,
    Qe: float,
    A: float,
    theta: float,
    phi_v: float,
    phi_0: float,
    alpha: float = 1,
) -> complex:
    r"""
    This is the complex function for a hanger (lambda/4 resonator).

    Parameters
    ----------
    f:
        frequency
    fr:
        resonance frequency
    A:
        background transmission amplitude
    Ql:
        loaded quality factor of the resonator
    Qe:
        magnitude of extrinsic quality factor :code:`Qe = |Q_extrinsic|`
    theta:
        phase of extrinsic quality factor (in rad)
    phi_v:
        phase to account for propagation delay to sample
    phi_0:
        phase to account for propagation delay from sample
    alpha:
        slope of signal around the resonance

    Returns
    -------
    :
        complex valued transmission


    See eq. S4 from Bruno et al. (2015) `ArXiv:1502.04082 <https://arxiv.org/abs/1502.04082>`_.

    .. math::

        S_{21} = A \left(1+\alpha \frac{f-f_r}{f_r} \right)
        \left(1- \frac{\frac{Q_l}{|Q_e|}e^{i\theta} }{1+2iQ_l \frac{f-f_r}{f_r}} \right)
        e^{i (\phi_v f + \phi_0)}

    The loaded and extrinsic quality factors are related to the internal and coupled Q according to:

    .. math::

        \frac{1}{Q_l} = \frac{1}{Q_c}+\frac{1}{Q_i}

    and

    .. math::

        \frac{1}{Q_c} = \mathrm{Re}\left(\frac{1}{|Q_e|e^{-i\theta}}\right)

    """
    slope_corr = 1 + alpha * (f - fr) / fr

    hanger_contribution = 1 - Ql / Qe * np.exp(1j * theta) / (
        1 + 2.0j * Ql * (f - fr) / fr
    )

    propagation_delay_corr = np.exp(1j * (phi_v * f + phi_0))
    S21 = A * slope_corr * hanger_contribution * propagation_delay_corr

    return S21


class ResonatorModel(lmfit.model.Model):
    """
    Resonator model

    Implementation and design patterns inspired by the
    `complex resonator model example <https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html>`_
    (lmfit documentation).

    """  # pylint: disable=line-too-long

    # pylint: disable=empty-docstring
    # pylint: disable=abstract-method
    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(hanger_func_complex_SI, *args, **kwargs)
        self.set_param_hint("Ql", min=0)  # Enforce Q is positive
        self.set_param_hint("Qe", min=0)  # Enforce Q is positive

        # Internal and coupled quality factor can be derived from fitted params
        self.set_param_hint("Qi", expr="1./(1./Ql-1./Qe*cos(theta))", vary=False)
        self.set_param_hint("Qc", expr="Qe/cos(theta)", vary=False)

    def guess(self, data, **kwargs):
        params = self.make_params()

        if kwargs.get("f", None) is None:
            return None

        f = kwargs["f"]
        argmin_s21 = np.abs(data).argmin()
        fmin = f.min()
        fmax = f.max()
        # guess that the resonance is the lowest point
        fr_guess = f[argmin_s21]
        # assume the user isn't trying to fit just a small part of a resonance curve.
        Q_min = 0.1 * (fr_guess / (fmax - fmin))
        delta_f = np.diff(f)  # assume f is sorted
        min_delta_f = delta_f[delta_f > 0].min()
        Q_max = (
            fr_guess / min_delta_f
        )  # assume data actually samples the resonance reasonably
        Q_guess = np.sqrt(Q_min * Q_max)  # geometric mean, why not?
        (phi_0_guess, phi_v_guess) = phase_guess(
            data, f
        )  # Come up with a guess for phase velocity

        self.set_param_hint("fr", value=fr_guess, min=fmin, max=fmax)
        self.set_param_hint("Ql", value=Q_guess, min=Q_min, max=Q_max)
        self.set_param_hint("Qe", value=Q_guess, min=0)
        self.set_param_hint("A", value=np.mean(abs(data)), min=0)

        # The parameters below need a proper guess.
        self.set_param_hint("theta", value=0, min=-np.pi / 2, max=np.pi / 2)
        self.set_param_hint("phi_0", value=phi_0_guess)
        self.set_param_hint("phi_v", value=phi_v_guess)
        self.set_param_hint("alpha", value=0, min=-1, max=1)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

    # Same design patter is used in lmfit.models
    __init__.__doc__ = get_model_common_doc() + mk_seealso("hanger_func_complex_SI")
    guess.__doc__ = get_guess_common_doc()


# Guesses the phase velocity based on the median of all the differences between
# consecutive phases
def phase_guess(S21, freq):
    phase = np.angle(S21)

    med_diff = np.median(np.diff(phase))
    freq_step = np.median(np.diff(freq))

    phi_v = med_diff / freq_step

    phi_0 = phase[0] - phi_v * freq[0]

    return phi_0, phi_v
