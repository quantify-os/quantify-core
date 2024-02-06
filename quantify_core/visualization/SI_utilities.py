# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
# pylint: disable=all
"""Utilities for managing SI units with plotting systems."""
from __future__ import annotations

import math
import re
import string
import warnings

import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uncertainties

golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
single_col_figsize = (3.39, golden_mean * 3.39)
double_col_figsize = (6.9, golden_mean * 6.9)
thesis_col_figsize = (12.2 / 2.54, golden_mean * 12.2 / 2.54)


def _get_scale_factor_and_offset_and_prefix(
    ticks: list[float], unit: str | None = None, precision: int = 4
) -> tuple[float, float, str]:
    """Return a convenient scale factor, offset and SI prefix based on the tick values.

    This function uses the :func:`~.SI_prefix_and_scale_factor` function to determine a
    scale factor such that the distance between ticks is in the range [0.1, 100.0), plus
    the corresponding scaled SI unit (e.g. 'mT', 'kV'), deduced from the input unit, to
    represent the tick values in those scaled units. In addition, an offset is
    calculated such that the maximum absolute tick value is less than 10^precision.

    Parameters
    ----------
    ticks
        A list of axis tick values.
    unit
        The unit of the tick values.
    precision
        The maximum amount of digits to display as tick labels.

    Returns
    -------
    scale_factor
        The scale factor to multiply the tick values with.
    offset
        The offset to subtract from the tick values.
    unit
        The unit including the SI prefix.

    Examples
    --------
    >>> _get_scale_factor_and_offset_and_prefix(
    ...     ticks=[2100000, 2100100, 2100200],
    ...     unit="Hz",
    ...     precision=4,
    ... )
    (1.0, 2100000, 'Hz')
    """
    max_v, min_v = max(ticks), min(ticks)
    resolution = (max_v - min_v) / len(ticks)
    scale_factor, unit = SI_prefix_and_scale_factor(val=resolution * 10, unit=unit)
    signed_max = max_v if abs(max_v) > abs(min_v) else min_v
    factor = pow(10, precision - 1)
    offset = int(signed_max * scale_factor / factor) * factor
    return scale_factor, offset, unit


def _set_offset_string(
    formatter: matplotlib.ticker.Formatter, offset: float, unit: str
) -> None:
    """Set the offset string of the Formatter to a conveniently scaled offset.

    This function scales the given offset and unit using
    :func:`~.SI_prefix_and_scale_factor`, and sets the offset string of the Formatter to
    the scaled offset value.

    Parameters
    ----------
    formatter
        The matplotlib Formatter.
    offset
        The value to scale and display.
    unit
        The unit of the value.
    """
    offset_scale, offset_unit = SI_prefix_and_scale_factor(offset, unit)
    disp_offset = offset * offset_scale
    formatter.set_offset_string(f"{disp_offset:+g} {offset_unit}")


def set_xlabel(
    label: str | plt.Axes,
    unit: str | None = None,
    axis: plt.Axes | None = None,
    auto_scale: bool = True,
    **kw,
) -> plt.Axes:
    """
    Add a unit aware x-label to an axis object.

    Parameters
    ----------
    label
        the desired label
    unit
        the unit
    auto_scale
        If True, then automatically scale the units
    axis
        matplotlib axis object to set label on
    **kw
        keyword argument to be passed to matplotlib.set_xlabel
    """
    if isinstance(label, plt.Axes):
        warnings.warn(
            "Passing axis as a first argument is deprecated and will be removed "
            "in quantify-core >= 0.10.0. Please use the new syntax "
            "set_xlabel(label, unit = None, axis = None)",
            FutureWarning,
            stacklevel=2,
        )
        axis, label, unit = label, unit, axis

    if axis is None:
        axis = plt.gca()

    if unit and auto_scale:
        xticks = axis.get_xticks()
        precision = 4
        scale_factor, offset, unit = _get_scale_factor_and_offset_and_prefix(
            xticks, unit, precision
        )

        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, _pos: f"{x * scale_factor - offset:.{precision}g}"
        )

        if offset != 0:
            _set_offset_string(formatter, offset, unit)

        axis.xaxis.set_major_formatter(formatter)
        axis.set_xlabel(label + f" [{unit}]", **kw)
    elif unit:
        axis.set_xlabel(label + f" [{unit}]", **kw)
    else:
        axis.set_xlabel(label, **kw)
    return axis


def set_ylabel(
    label: str | plt.Axes,
    unit: str | None = None,
    axis: plt.Axes | None = None,
    auto_scale: bool = True,
    **kw,
) -> plt.Axes | None:
    """
    Add a unit aware y-label to an axis object.

    Parameters
    ----------
    label
        the desired label
    unit
        the unit
    axis
        matplotlib axis object to set label on
    auto_scale
        If True, then automatically scale the units
    **kw
        keyword argument to be passed to matplotlib.set_ylabel
    """
    if isinstance(label, plt.Axes):
        warnings.warn(
            "Passing axis as a first argument is deprecated and will be removed"
            " in quantify-core >= 0.10.0. Please use the new syntax"
            " set_ylabel(label, unit = None, axis = None)",
            FutureWarning,
            stacklevel=2,
        )
        axis, label, unit = label, unit, axis  # type: ignore

    if axis is None:
        axis = plt.gca()

    if unit and auto_scale:
        yticks = axis.get_yticks()
        precision = 6
        scale_factor, offset, unit = _get_scale_factor_and_offset_and_prefix(
            yticks, unit, precision=precision
        )

        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, _pos: f"{x * scale_factor - offset:.{precision}g}"
        )

        if offset != 0:
            _set_offset_string(formatter, offset, unit)

        axis.yaxis.set_major_formatter(formatter)

        axis.set_ylabel(label + f" [{unit}]", **kw)
    elif unit:
        axis.set_ylabel(label + f" [{unit}]", **kw)
    else:
        axis.set_ylabel(label, **kw)
    return axis


def set_cbarlabel(
    cbar: matplotlib.colorbar.Colorbar, label: str, unit: str | None = None, **kw
):
    """
    Add a unit aware z-label to a colorbar object.

    Parameters
    ----------
    cbar
        colorbar object to set label on
    label
        the desired label
    unit
        the unit
    **kw
        keyword argument to be passed to cbar.set_label
    """
    if unit:
        zticks = cbar.get_ticks()
        precision = 6
        scale_factor, offset, unit = _get_scale_factor_and_offset_and_prefix(
            zticks, unit, precision=precision
        )

        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, _pos: f"{x * scale_factor - offset:.{precision}g}"
        )

        if offset != 0:
            _set_offset_string(formatter, offset, unit)

        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.set_label(label + f" [{unit}]")

    else:
        cbar.set_label(label, **kw)
    return cbar


def adjust_axeslabels_SI(ax) -> None:
    """Auto adjust the labels of a plot generated by xarray to SI-unit aware labels."""
    xlabel = ax.get_xlabel()
    idxl = xlabel.find("[")
    idxr = xlabel.find("]")

    # only update the label if brackets are present
    if idxl != -1 and idxr != -1:
        # extract unit
        xunit = xlabel[idxl + 1 : idxr]
        xlabel = xlabel[: -(len(xunit) + 3)]
        # replace by a unit aware label formatter
        set_xlabel(xlabel, xunit, ax)

    ylabel = ax.get_ylabel()
    idxl = ylabel.find("[")
    idxr = ylabel.find("]")
    # only update the label if brackets are present
    if idxl != -1 and idxr != -1:
        yunit = ylabel[idxl + 1 : idxr]
        ylabel = ylabel[: -(len(yunit) + 3)]
        # replace by a unit aware label formatter
        set_ylabel(ylabel, yunit, ax)


SI_PREFIXES = dict(zip(range(-24, 25, 3), "yzafpnμm kMGTPEZY"))
SI_PREFIXES[0] = ""

# N.B. not all of these are SI units, however, all of these support SI prefixes
SI_UNITS = (
    "SI_PREFIX_ONLY,m,s,g,W,J,V,A,F,T,Hz,Ohm,S,N,C,px,b,B,K,Bar,"
    r"Vpeak,Vpp,Vp,Vrms,A/s,$\Phi_0$".split(",")
)  # noqa: W605

_SI_PREFIX_TO_FACTOR_MAPPING = {v: 10**key for key, v in SI_PREFIXES.items()}
_SI_PREFIX_TO_FACTOR_MAPPING["u"] = 10**-6

_prefix_regexp = "(" + "|".join(list("yzafpnμmkMGTPEZY")) + ")"
_si_regex = "(" + "|".join(map(re.escape, SI_UNITS)) + ")"
_prefixed_si_regex = re.compile(f"{_prefix_regexp}{_si_regex}$")


def SI_prefix_and_scale_factor(
    val: float, unit: str | None = None
) -> tuple[float, str]:
    """
    Takes in a value and unit, returns a scale factor and scaled unit.
    It returns a scale factor to convert the input value to a value in the
    range [1.0, 1000.0), plus the corresponding scaled SI unit (e.g. 'mT', 'kV'),
    deduced from the input unit, to represent the input value in those scaled units.

    The scaling is only applied if the unit is an unscaled or scaled unit present in
    the variable :data::`SI_UNITS`.

    If the unit is None, no scaling is done.
    If the unit is "SI_PREFIX_ONLY", the value is scaled and an SI prefix is applied
    without a base unit.

    Parameters
    ----------
    val
        the value
    unit
        the unit of the value

    Returns
    -------
    scale_factor
        scale_factor needed to convert value
    scaled_unit
        unit including the prefix
    """
    if unit and val is not None and (match := _prefixed_si_regex.match(unit)):
        scale_part = match.group(1)
        unit_part = match.group(2)
        plus_scale = _SI_PREFIX_TO_FACTOR_MAPPING[scale_part]
        scale_factor, scaled_unit = SI_prefix_and_scale_factor(
            plus_scale * val, unit_part
        )
        return plus_scale * scale_factor, scaled_unit

    if unit in SI_UNITS:
        try:
            with np.errstate(all="ignore"):
                prefix_power = np.log10(abs(val)) // 3 * 3
                prefix = SI_PREFIXES[prefix_power]
                # Greek symbols not supported in tex
                if plt.rcParams["text.usetex"] and prefix == "μ":
                    prefix = r"$\mu$"
            if unit == "SI_PREFIX_ONLY":
                scale_factor, scaled_unit = 10**-prefix_power, prefix
            elif unit == "s" and val > 2 * 60:
                if val > 2 * 3600:  # Convert to hours if larger than 2 hours
                    scale_factor, scaled_unit = 1 / 3600, "hrs"
                else:  # Convert to minutes if between 2 minutes and 2 hours
                    scale_factor, scaled_unit = 1 / 60, "min"
            else:
                scale_factor, scaled_unit = 10**-prefix_power, prefix + unit
        # this exception can be triggered in the pyqtgraph multi processing
        except (KeyError, TypeError):
            scale_factor, scaled_unit = 1, unit

    elif unit is None:
        scale_factor, scaled_unit = 1, ""
    else:
        scale_factor, scaled_unit = 1, unit
    return scale_factor, scaled_unit


def SI_val_to_msg_str(val: float | int, unit: str = None, return_type=str):
    """
    Takes in a value  with optional unit and returns a string tuple consisting of
    (value_str, unit) where the value and unit are rescaled according to SI prefixes,
    IF the unit is an SI unit (according to the comprehensive list of
    SI units in this file ;).

    the value_str is of the type specified in return_type (str) by default.
    """
    sc, new_unit = SI_prefix_and_scale_factor(val, unit)
    try:
        new_val = sc * val
    except TypeError:
        return return_type(val), unit

    # Convert floats to int if possible and if inital value was int
    if isinstance(val, int) and isinstance(new_val, float) and new_val.is_integer():
        new_val = int(new_val)

    return return_type(new_val), new_unit


class SafeFormatter(string.Formatter):
    """
    A formatter that replaces "missing" values and "bad_fmt" to prevent unexpected
    Exceptions being raised.

    Parameters
    ----------
    missing
        Replaces missing values with specified string.
    bad_fmt
        Replaces values that cannot be formatted with specified string.

    Notes
    -----
    Based on https://stackoverflow.com/questions/20248355/how-to-get-python-to-gracefully-format-none-and-non-existing-fields
    """

    def __init__(self, missing: str = "~~", bad_fmt: str = "!!"):
        self.missing, self.bad_fmt = missing, bad_fmt

    def get_field(self, field_name, args, kwargs):  # noqa: D102
        # Handle a key not found
        try:
            val = super().get_field(field_name, args, kwargs)
            # Python 3, 'super().get_field(field_name, args, kwargs)' works
        except (KeyError, AttributeError):
            val = None, field_name
        return val

    def format_field(self, value, format_spec):  # noqa: D102
        # handle an invalid format
        if value is None:
            return self.missing
        try:
            return super().format_field(value, format_spec)
        except ValueError as e:
            if self.bad_fmt is not None:
                return self.bad_fmt
            raise e


def format_value_string(
    par_name: str,
    parameter: (
        lmfit.Parameter
        | uncertainties.core.Variable
        | uncertainties.core.AffineScalarFunc
        | float
    ),
    end_char="",
    unit=None,
) -> str:
    """
    Format an lmfit parameter or uncertainties ufloat to a string of value with
    uncertainty.

    If there is no stderr, use 5 significant figures.
    If there is a standard error use a precision one order of magnitude more precise
    than the size of the error and display the stderr itself to two significant figures
    in standard index notation in the same units as the value.

    Parameters
    ----------
    par_name:
        A name of the parameter to use in the string
    parameter : :class:`lmfit.parameter.Parameter`,
        :class:`!uncertainties.core.Variable` or float.
        A :class:`~lmfit.parameter.Parameter` object or an object e.g.,
        returned by :func:`!uncertainties.ufloat`. The value and stderr of this
        parameter will be used. If a float is given, the stderr is taken to be None.
    end_char:
        A character that will be put at the end of the line.
    unit:
        A unit. If this is an SI unit it will be used in automatically
        determining a prefix for the unit and rescaling accordingly.

    Returns
    -------
    :
        The parameter and its error formatted as a string
    """
    if isinstance(
        parameter, (uncertainties.core.Variable, uncertainties.core.AffineScalarFunc)
    ):
        value = parameter.nominal_value
        stderr = parameter.std_dev
        if np.isnan(stderr):
            stderr = None
    elif isinstance(parameter, lmfit.Parameter):
        value = parameter.value
        stderr = parameter.stderr
    else:
        value = parameter
        stderr = None

    scale_factor, unit = SI_prefix_and_scale_factor(value, unit)
    val = value * scale_factor
    stderr = stderr * scale_factor if stderr is not None else None

    (val_format_specifier, err_format_specifier) = value_precision(val, stderr)

    fmt = SafeFormatter(missing="NaN")
    if stderr is not None:
        val_string = rf": {val_format_specifier}$\pm${err_format_specifier} {{}}{{}}"
        # par name is excluded from the format command to allow latex {} characters.
        val_string = par_name + fmt.format(val_string, val, stderr, unit, end_char)
    else:
        val_string = f": {val_format_specifier} {{}}{{}}"
        # par name is excluded from the format command to allow latex {} characters.
        val_string = par_name + fmt.format(val_string, val, unit, end_char)
    return val_string


def value_precision(val: float, stderr=None) -> tuple[str, str]:
    """
    Calculate the precision to which a parameter is to be specified, according to
    its standard error. Returns the appropriate format specifier string.

    If there is no stderr, use 5 significant figures.
    If there is a standard error use a precision one order of magnitude more precise
    than the size of the error and display the stderr itself to two significant figures
    in standard index notation in the same units as the value.

    Parameters
    ----------
    val
        the nominal value of the parameter
    stderr
        the standard error on the parameter

    Returns
    -------
    val_format_specifier
        python format specifier which sets the precision of the parameter value
    err_format_specifier
        python format specifier which set the precision of the error
    """
    if stderr is None or stderr == 0 or math.isnan(stderr):
        return "{:.5g}", "{:.1f}"

    # if statement to catch edge case of log10(0) being undefined.
    value_mag = 1 if val == 0 else np.floor(np.log10(abs(val))) + 1
    err_mag = 1 if stderr == 0 else np.floor(np.log10(abs(stderr))) + 1
    if err_mag == 2:
        return "{:.0f}", "{:.0f}"
    elif err_mag == 1:
        return "{:.1f}", "{:.1f}"
    else:
        sig_figs = int(
            max(value_mag - err_mag + 2, 2)
        )  # If the error is the same size as the value or larger, use 2 sig figs
        return "{:#." + f"{sig_figs:d}" + "g}", "{:#.2g}"
