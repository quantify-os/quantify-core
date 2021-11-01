# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Models and fit functions to be used with the lmfit fitting framework."""
from __future__ import annotations

from typing import Tuple

import lmfit
import numpy as np


def get_model_common_doc() -> str:
    """
    Returns a common docstring to be used with custom fitting
    :class:`~lmfit.model.Model` s.

    .. admonition:: Usage example for a custom fitting model
        :class: dropdown, tip

        See the usage example at the end of the :class:`~ResonatorModel` source-code:

        .. literalinclude:: ../quantify_core/analysis/fitting_models.py
            :pyobject: ResonatorModel
    """
    return (
        lmfit.models.COMMON_INIT_DOC.replace(
            ":class:`Model`", ":class:`~lmfit.model.Model`"
        )
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

        .. literalinclude:: ../quantify_core/analysis/fitting_models.py
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


def mk_seealso(
    function_name: str,
    role: str = "func",
    prefix: str = "\n\n",
    module_location: str = ".",
) -> str:
    """
    Returns a sphinx `seealso` pointing to a function.

    Intended to be used for building custom fitting model docstrings.

    .. admonition:: Usage example for a custom fitting model
        :class: dropdown, tip

        See the usage example at the end of the :class:`~ResonatorModel` source-code:

        .. literalinclude:: ../quantify_core/analysis/fitting_models.py
            :pyobject: ResonatorModel

    Parameters
    ----------
    function_name
        name of the function to point to
    role
        a sphinx role, e.g. :code:`"func"`
    prefix
        string preceding the `seealso`
    module_location
        can be used to indicate a function outside this module, e.g.,
        :code:`my_module.submodule` which contains the function.

    Returns
    -------
    :
        resulting string
    """
    return f"{prefix}.. seealso:: :{role}:`~{module_location}{function_name}`\n"


# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
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


    See eq. S4 from Bruno et al. (2015)
    `ArXiv:1502.04082 <https://arxiv.org/abs/1502.04082>`_.

    .. math::

        S_{21} = A \left(1+\alpha \frac{f-f_r}{f_r} \right)
        \left(1- \frac{\frac{Q_l}{|Q_e|}e^{i\theta} }{1+2iQ_l \frac{f-f_r}{f_r}} \right)
        e^{i (\phi_v f + \phi_0)}

    The loaded and extrinsic quality factors are related to the internal and coupled Q
    according to:

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


def cos_func(
    x: float,
    frequency: float,
    amplitude: float,
    offset: float,
    phase: float = 0,
) -> float:
    r"""
    An oscillating cosine function:

    :math:`y = \mathrm{amplitude} \times \cos(2 \pi \times \mathrm{frequency} \times x + \mathrm{phase}) +  \mathrm{offset}`

    Parameters
    ----------
    x:
        The independent variable (time, for example)
    frequency:
        A generalized frequency (in units of inverse x)
    amplitude:
        Amplitude of the oscillation
    offset:
        Output signal vertical offset
    phase:
        Phase offset / rad

    Returns
    -------
    :
        Output signal magnitude
    """  # pylint: disable=line-too-long

    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset


def exp_decay_func(
    t: float,
    tau: float,
    amplitude: float,
    offset: float,
    n_factor: float,
) -> float:
    r"""
    This is a general exponential decay function:

    :math:`y = \mathrm{amplitude} \times \exp\left(-(t/\tau)^\mathrm{n\_factor}\right) + \mathrm{offset}`

    Parameters
    ----------
    t:
        time
    tau:
        decay time
    amplitude:
        amplitude of the exponential decay
    offset:
        asymptote of the exponential decay, the value at t=infinity
    n_factor:
        exponential decay factor

    Returns
    -------
    :
        Output of exponential function as a float
    """  # pylint: disable=line-too-long
    return amplitude * np.exp(-((t / tau) ** n_factor)) + offset


def exp_damp_osc_func(
    t: float,
    tau: float,
    n_factor: float,
    frequency: float,
    phase: float,
    amplitude: float,
    offset: float,
):
    r"""
    A sinusoidal oscillation with an exponentially decaying envelope function:

    :math:`y = \mathrm{amplitude} \times \exp\left(-(t/\tau)^\mathrm{n\_factor}\right)(\cos(2\pi\mathrm{frequency}\times t + \mathrm{phase}) + \mathrm{oscillation_offset}) + \mathrm{exponential_offset}`

    Parameters
    ----------
    t:
        time
    tau:
        decay time
    n_factor:
        exponential decay factor
    frequency:
        frequency of the oscillation
    phase:
        phase of the oscillation
    amplitude:
        initial amplitude of the oscillation
    oscillation_offset:
        vertical offset of cosine oscillation relative to exponential asymptote
    exponential_offset:
        offset of exponential asymptote

    Returns
    -------
    :
        Output of decaying cosine function as a float
    """  # pylint: disable=line-too-long

    oscillation = amplitude * (np.cos(2 * np.pi * frequency * t + phase))
    exp_decay = np.exp(-((t / tau) ** n_factor))
    osc_decay = oscillation * exp_decay + offset
    return osc_decay


# This class is used a literal include in the docs so the pylint options are here
# pylint: disable=empty-docstring
# pylint: disable=abstract-method
# pylint: disable=too-few-public-methods
class CosineModel(lmfit.model.Model):
    """
    Exemplary lmfit model with a guess for a cosine.

    .. note::

        The :mod:`lmfit.models` module provides several fitting models that might fit
        your needs out of the box.
    """

    def __init__(self, *args, **kwargs):
        # pass in the model's equation
        super().__init__(cos_func, *args, **kwargs)

        # configure constraints that are independent from the data to be fitted
        self.set_param_hint("frequency", min=0, vary=True)  # enforce positive frequency
        self.set_param_hint("amplitude", min=0, vary=True)  # enforce positive amplitude
        self.set_param_hint("offset", vary=True)
        self.set_param_hint(
            "phase", vary=True, min=-np.pi, max=np.pi
        )  # enforce phase range

    # pylint: disable=missing-function-docstring
    def guess(self, data, **kws) -> lmfit.parameter.Parameters:

        # guess parameters based on the data

        self.set_param_hint("offset", value=np.average(data))
        self.set_param_hint("amplitude", value=(np.max(data) - np.min(data)) / 2)
        # a simple educated guess based on experiment type
        # a more elaborate but general approach is to use a Fourier transform
        self.set_param_hint("frequency", value=1.2)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    # Same design patter is used in lmfit.models to inherit common docstrings.
    # We adjust these common docstrings to our docs build pipeline
    __init__.__doc__ = get_model_common_doc() + mk_seealso("cos_func")
    guess.__doc__ = get_guess_common_doc()


class ResonatorModel(lmfit.model.Model):
    """
    Resonator model

    Implementation and design patterns inspired by the
    `complex resonator model example <https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html>`_
    (`lmfit` documentation).

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

    # pylint: disable=too-many-locals
    # pylint: disable=missing-function-docstring
    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        f = kws.get("f", None)
        if f is None:
            return None

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
        (phi_0_guess, phi_v_guess) = resonator_phase_guess(
            data, f
        )  # Come up with a guess for phase velocity

        self.set_param_hint("fr", value=fr_guess, min=fmin, max=fmax)
        self.set_param_hint("Ql", value=Q_guess * 1.01, min=Q_min, max=Q_max)
        self.set_param_hint("Qe", value=Q_guess * 0.99, min=0)
        self.set_param_hint("A", value=np.mean(abs(data)), min=0)

        # The parameters below need a proper guess.
        self.set_param_hint("theta", value=0, min=-np.pi / 2, max=np.pi / 2)
        self.set_param_hint("phi_0", value=phi_0_guess)
        self.set_param_hint("phi_v", value=phi_v_guess)
        self.set_param_hint("alpha", value=0, min=-1, max=1)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    # Same design patter is used in lmfit.models
    __init__.__doc__ = get_model_common_doc() + mk_seealso("hanger_func_complex_SI")
    guess.__doc__ = get_guess_common_doc()


class ExpDecayModel(lmfit.model.Model):
    """
    Model for an exponential decay, such as a qubit T1 measurement.
    """

    # pylint: disable=empty-docstring
    # pylint: disable=abstract-method
    # pylint: disable=too-few-public-methods

    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later.

        super().__init__(exp_decay_func, *args, **kwargs)
        self.set_param_hint("tau", min=0)  # Enforce T1 is positive

        self.set_param_hint("amplitude", vary=True)
        self.set_param_hint("offset", vary=True)
        self.set_param_hint("n_factor", expr="1", vary=False)

    # pylint: disable=missing-function-docstring
    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        delay = kws.get("delay", None)

        if delay is None:
            return None

        # To guess the upper amplitude and offset,
        # use the first and last values of the data
        self.set_param_hint("offset", value=data[-1])
        self.set_param_hint("amplitude", value=data[0] - data[-1])

        # The guess for tau is somewhere in the middle of the time range
        tau = np.median(delay)

        self.set_param_hint("tau", value=tau, min=0)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    # Same design patter is used in lmfit.models
    __init__.__doc__ = get_model_common_doc() + mk_seealso("exp_decay_func")
    guess.__doc__ = get_guess_common_doc()


class RabiModel(lmfit.model.Model):
    r"""
    Model for a Rabi oscillation as a function of the microwave drive amplitude.
    Phase of oscillation is fixed at :math:`\pi` in order to ensure that the oscillation
    is at a minimum when the drive amplitude is 0.
    """

    # pylint: disable=empty-docstring
    # pylint: disable=abstract-method
    # pylint: disable=too-few-public-methods

    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(cos_func, *args, **kwargs)

        # Enforce oscillation frequency is positive
        self.set_param_hint("frequency", min=0)

        # Fix the phase at pi so that the ouput is at a minimum when x=0
        self.set_param_hint("phase", expr="3.141592653589793", vary=False)

        # Pi-pulse amplitude can be derived from the oscillation frequency
        self.set_param_hint("amp180", expr="1/(2*frequency)", vary=False)

    # pylint: disable=missing-function-docstring
    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        drive_amp = kws.get("drive_amp", None)
        if drive_amp is None:
            return None

        amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
        offs_guess = np.mean(data)

        (freq_guess, _) = fft_freq_phase_guess(data, drive_amp)

        self.set_param_hint("frequency", value=freq_guess, min=0)
        self.set_param_hint("amplitude", value=amp_guess, min=0)
        self.set_param_hint("offset", value=offs_guess)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    # Same design patter is used in lmfit.models
    __init__.__doc__ = get_model_common_doc() + mk_seealso("cos_func")
    guess.__doc__ = get_guess_common_doc()


class DecayOscillationModel(lmfit.model.Model):
    r"""
    Model for a decaying oscillation which decays to a point with 0 offset from
    the centre of the of the oscillation (as in a Ramsey experiment, for example).
    """

    # pylint: disable=empty-docstring
    # pylint: disable=abstract-method
    # pylint: disable=too-few-public-methods

    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(exp_damp_osc_func, *args, **kwargs)

        # Enforce oscillation frequency is positive
        self.set_param_hint("frequency", min=0)
        # Enforce amplitude is positive
        self.set_param_hint("amplitude", min=0)
        # Enforce decay time is positive
        self.set_param_hint("tau", min=0)

        # Fix the n_factor at 1
        self.set_param_hint("n_factor", expr="1", vary=False)

    # pylint: disable=missing-function-docstring
    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        t = kws.get("t", None)
        if t is None:
            raise ValueError(
                'Time variable "t" must be specified in order to guess parameters'
            )
            return None

        amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
        exp_offs_guess = np.mean(data)
        tau_guess = 2 / 3 * np.max(t)

        (freq_guess, phase_guess) = fft_freq_phase_guess(data, t)

        self.set_param_hint("frequency", value=freq_guess, min=0)
        self.set_param_hint("amplitude", value=amp_guess, min=0)
        self.set_param_hint("offset", value=exp_offs_guess)
        self.set_param_hint("phase", value=phase_guess)
        self.set_param_hint("tau", value=tau_guess, min=0)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    # Same design patter is used in lmfit.models
    __init__.__doc__ = get_model_common_doc() + mk_seealso("exp_damp_osc_func")
    guess.__doc__ = get_guess_common_doc()


def resonator_phase_guess(s21: np.ndarray, freq: np.ndarray) -> Tuple[float, float]:
    """
    Guesses the phase velocity in resonator spectroscopy,
    based on the median of all the differences between consecutive phases.

    Parameters
    ----------
    s21:
        Resonator S21 data
    freq:
        Frequency of the spectroscopy pulse

    Returns
    -------
    phi_0:
        Guess for the phase offset
    phi_v:
        Guess for the phase velocity
    """
    phase = np.angle(s21)

    med_diff = np.median(np.diff(phase))
    freq_step = np.median(np.diff(freq))

    phi_v = med_diff / freq_step

    phi_0 = phase[0] - phi_v * freq[0]

    return phi_0, phi_v


def fft_freq_phase_guess(data: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    """
    Guess for a cosine fit using FFT, only works for evenly spaced points.

    Parameters
    ----------
    data:
        Input data to FFT
    t:
        Independent variable (e.g. time)

    Returns
    -------
    freq_guess:
        Guess for the frequency of the cosine function
    ph_guess:
        Guess for the phase of the cosine function
    """

    # Only first half of array is used, because the second half contains the
    # negative frequecy components, and we want a positive frequency.
    power = np.fft.fft(data)[: len(data) // 2]
    freq = np.fft.fftfreq(len(data), t[1] - t[0])[: len(power)]
    power[0] = 0  # Removes DC component from fourier transform

    # Use absolute value of complex valued spectrum
    abs_power = np.abs(power)
    freq_guess = abs(freq[abs_power == max(abs_power)][0])
    # the condition data == max(data) can have several solutions
    #               (for example when discretization is visible)
    # to prevent errors we pick the first solution
    ph_guess = 2 * np.pi - (2 * np.pi * t[data == max(data)] * freq_guess)[0]

    return freq_guess, ph_guess
