"""Top-level package for quantify."""

__author__ = """Qblox"""
__email__ = 'hello@qblox.com'
__version__ = '0.1.0'


from quantify.data.handling import get_datadir, set_datadir, \
    snapshot

__all__ = ['get_datadir', 'set_datadir', 'snapshot']
