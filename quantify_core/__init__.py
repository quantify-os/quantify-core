# pylint: disable=django-not-configured
import warnings

from . import utilities
from ._version import __version__

warnings.warn(
    "This package has reached its end of life. "
    "It is no longer maintained and will not receive any further updates or support. "
    "For further developments, please refer to the new Quantify repository: https://gitlab.com/quantify-os/quantify."
    "All existing functionalities can be accessed via the new Quantify repository.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["utilities"]
