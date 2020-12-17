# -----------------------------------------------------------------------------
# Description:    Wrapper of appnope to avoid App Nap (macOS) of Quantify process
# Repository:     https://gitlab.com/quantify-os/quantify-core
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------

# Necessary issues with the power-saving features on macOS
# See also:
# https://github.com/pyqtgraph/pyqtgraph/pull/1092
# https://forum.libcinder.org/topic/os-x-how-to-keep-rendering-in-background
import sys
import platform
from distutils.version import LooseVersion as V
import appnope


def requires_appnope():
    """
    Used to check if `appnope` is necessary
    """
    # extracted from `appnope` source
    return (sys.platform == "darwin") and V(platform.mac_ver()[0]) > V("10.9")


def refresh_nope():
    """
    Communicate to the macOS that the process needs to stay awake
    and must not be sent to App Nap.

    This refresh needs to be triggered recurrently (e.g. QtCore.QTimer)
    """
    appnope.nope()
