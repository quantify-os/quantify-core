# -----------------------------------------------------------------------------
# Description:    Utilities for managing SI units with plotting systems.
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

golden_mean = (np.sqrt(5)-1.0)/2.0  # Aesthetic ratio
single_col_figsize = (3.39, golden_mean*3.39)
double_col_figsize = (6.9, golden_mean*6.9)
thesis_col_figsize = (12.2/2.54, golden_mean*12.2/2.54)


def set_xlabel(axis, label, unit=None, **kw):
    """
    Add a unit aware x-label to an axis object.

    Parameters
    ----------
    axis
        matplotlib axis object to set label on
    label
        the desired label
    unit
        the unit
    **kw
        keyword argument to be passed to matplotlib.set_xlabel
    """
    if unit is not None and unit != '':
        xticks = axis.get_xticks()
        scale_factor, unit = SI_prefix_and_scale_factor(val=max(abs(xticks)), unit=unit)
        formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.4g}'.format(x*scale_factor))

        axis.xaxis.set_major_formatter(formatter)
        axis.set_xlabel(label+' ({})'.format(unit), **kw)
    else:
        axis.set_xlabel(label, **kw)
    return axis


def set_ylabel(axis, label, unit=None, **kw):
    """
    Add a unit aware y-label to an axis object.

    Parameters
    ----------
    axis
        matplotlib axis object to set label on
    label
        the desired label
    unit
        the unit
    **kw
        keyword argument to be passed to matplotlib.set_ylabel
    """
    if unit is not None and unit != '':
        yticks = axis.get_yticks()
        scale_factor, unit = SI_prefix_and_scale_factor(
            val=max(abs(yticks)), unit=unit)
        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: '{:.6g}'.format(x*scale_factor))

        axis.yaxis.set_major_formatter(formatter)

        axis.set_ylabel(label+' ({})'.format(unit), **kw)
    else:
        axis.set_ylabel(label, **kw)
    return axis


def set_cbarlabel(cbar, label, unit=None, **kw):
    """
    Add a unit aware z-label to a colorbar object

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
    if unit is not None and unit != '':
        zticks = cbar.get_ticks()
        scale_factor, unit = SI_prefix_and_scale_factor(
            val=max(abs(zticks)), unit=unit)
        cbar.set_ticks(zticks)
        cbar.set_ticklabels(zticks*scale_factor)
        cbar.set_label(label + ' ({})'.format(unit))

    else:
        cbar.set_label(label, **kw)
    return cbar


SI_PREFIXES = dict(zip(range(-24, 25, 3), 'yzafpnμm kMGTPEZY'))
SI_PREFIXES[0] = ""

# N.B. not all of these are SI units, however, all of these support SI prefixes
SI_UNITS = 'm,s,g,W,J,V,A,F,T,Hz,Ohm,S,N,C,px,b,B,K,Bar,' \
           'Vpeak,Vpp,Vp,Vrms,$\Phi_0$,A/s'.split(',')  # noqa: W605


def SI_prefix_and_scale_factor(val, unit=None):
    """
    Takes in a value and unit and if applicable returns the proper scale factor and SI prefix.

    Parameters
    ----------
    val : float
        the value
    unit : str
        the unit of the value
    Returns
    -------
    scale_factor : float
        scale_factor needed to convert value
    unit : str
        unit including the prefix
    """
    if unit in SI_UNITS:
        try:
            with np.errstate(all="ignore"):
                prefix_power = np.log10(abs(val))//3 * 3
                prefix = SI_PREFIXES[prefix_power]
                # Greek symbols not supported in tex
                if plt.rcParams['text.usetex'] and prefix == 'μ':
                    prefix = r'$\mu$'

            return 10 ** -prefix_power,  prefix + unit
        except (KeyError, TypeError):
            pass

    return 1, unit if unit is not None else ""


def SI_val_to_msg_str(val: float, unit: str = None, return_type=str):
    """
    Takes in a value  with optional unit and returns a string tuple consisting of (value_str, unit) where the value
    and unit are rescaled according to SI prefixes, IF the unit is an SI unit (according to the comprehensive list of
    SI units in this file ;).

    the value_str is of the type specified in return_type (str) by default.
    """
    sc, new_unit = SI_prefix_and_scale_factor(val, unit)
    try:
        new_val = sc*val
    except TypeError:
        return return_type(val), unit

    return return_type(new_val), new_unit
