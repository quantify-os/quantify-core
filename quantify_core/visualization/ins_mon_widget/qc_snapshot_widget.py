# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
"""Module containing the pyqtgraph based plotting monitor."""
import json
from enum import Enum
import pprint
from typing import Any, Optional, Tuple
from collections import OrderedDict

from pyqtgraph.Qt import QtCore, QtWidgets
from qcodes.utils.helpers import NumpyJSONEncoder

from quantify_core.utilities import deprecated
from quantify_core.visualization import _appnope
from quantify_core.visualization.SI_utilities import SI_val_to_msg_str


class QcSnapshotWidget(QtWidgets.QTreeWidget):
    """
    Widget for displaying QcoDes instrument snapshots.
    Heavily inspired by the DataTreeWidget.
    """

    def __init__(self, parent=None, data=None):
        QtWidgets.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(4)
        self.setHeaderLabels(["Name/Label", "Value", "Unit", "Last update"])
        self.nodes = {}
        self.timer_appnope = None

        if _appnope.requires_appnope():
            # Start a timer to ensure the App Nap of macOS does not idle this process.
            # The process is sent to App Nap after a window is minimized or not
            # visible for a few seconds, this ensure we avoid that.
            # If this is not performed very long and cryptic errors will rise
            # (which is fatal for a running measurement)

            self.timer_appnope = QtCore.QTimer(self)
            self.timer_appnope.timeout.connect(_appnope.refresh_nope)
            self.timer_appnope.start(30)  # milliseconds

    def setData(self, data):
        """
        data should be a snapshot dict: See :class: `~quantify_core.data.handling.snapshot`
        """
        self.buildTreeSnapshot(snapshot=data)
        self.resizeColumnToContents(0)

    def buildTreeSnapshot(self, snapshot):
        # exists so that function can be called with no data in construction
        if snapshot is None:
            return

        parent = self.invisibleRootItem()
        for instrument in sorted(snapshot.keys()):
            sub_snap = snapshot[instrument]
            # Name of the node in the self.nodes dictionary
            instrument_name = sub_snap["name"]

            # Default to instrument_name for backwards compatibility.
            # "label" is not present in qcodes<0.36.
            instrument_label = sub_snap.get("label", instrument_name)

            node = self._add_node(parent, instrument_label, instrument_name)
            self._fill_node_recursively(sub_snap, node, instrument_name)

    def _add_node(self, parent, display_string, node_key):
        if node_key in self.nodes:
            # if node exists, update its string
            self.nodes[node_key].setData(0, 0, display_string)
        else:
            # if node doesn't exist create a new one
            self.nodes[node_key] = QtWidgets.QTreeWidgetItem([display_string, "", ""])
            parent.addChild(self.nodes[node_key])
        return self.nodes[node_key]

    def _fill_node_recursively(self, snapshot, node, node_key):
        """Takes an existing ``node``. Fills it with new nodes based on ``snapshot``.

        Parameters
        ----------
        snapshot
            Snapshot with information about content of ``node``
        node
            Node to be filled with information
        node_key
            Key of the existing ``node``
        """
        sub_snaps = {}
        for key in ["submodules", "channels"]:
            sub_snaps.update(snapshot.get(key, {}))

        for sub_snapshot_key in sorted(sub_snaps.keys()):
            sub_snap = sub_snaps[sub_snapshot_key]
            # Some names contain higher nodes, remove them (with underscore) for brevity
            for node_key_part in node_key.split("."):
                sub_snapshot_key = QcSnapshotWidget._remove_left(
                    sub_snapshot_key, node_key_part
                )
            sub_node_key = f"{node_key}.{sub_snapshot_key}"

            # Default to sub_snapshot_key for backwards compatibility.
            # "label" is not present in qcodes<0.36.
            instrument_label = sub_snap.get("label", sub_snapshot_key)

            sub_node = self._add_node(node, instrument_label, sub_node_key)
            self._fill_node_recursively(sub_snap, sub_node, sub_node_key)

        # Don't sort keys if we encounter an OrderedDict
        param_snaps = snapshot.get("parameters", {})
        param_snaps_keys = param_snaps.keys()
        if not isinstance(param_snaps, OrderedDict):
            param_snaps_keys = sorted(param_snaps.keys())

        for param_name in param_snaps_keys:
            param_snap = param_snaps[param_name]
            # Depending on the type of data stored in value do different things,
            # currently only blocks non-dicts
            if not "value" in param_snap.keys():
                # Some parameters do not have a value, these are not shown
                # in the instrument monitor.
                continue
            if isinstance(param_snap["value"], dict):
                # Treat dict as submodule and all entries of dict as parameters.
                # If the dict keys are not str, they are converted to str. Use
                # OrderedDict to sort numbers properly.
                pars = OrderedDict()
                for key in sorted(param_snap["value"].keys()):
                    val = param_snap["value"][key]
                    pars[str(key)] = {
                        "value": val,
                        "name": str(key),
                        "ts": param_snap["ts"],
                        "unit": "",
                        "label": "",
                    }
                sub_snap = {"submodules": {param_name: {"parameters": pars}}}
                self._fill_node_recursively(sub_snap, node, node_key)
            else:
                self._add_single_parameter(param_snap, param_name, node, node_key)

    @staticmethod
    def _convert_to_str(value: Any, unit: Optional[str]) -> Tuple[str, str]:
        """If no unit is given, convert to string and apply nice formatting.
        Otherwise make sure to interpret SI unit appropriately.

        Parameters
        ----------
        value:
            Value of parameter
        unit:
            Unit of parameter

        Returns
        -------
        :
            new value and new unit
        """
        if not unit:
            if isinstance(value, Enum):
                # For Enum, don't show class name
                return value.name, ""
            return str(value), ""
        return SI_val_to_msg_str(value, unit)

    def _add_single_parameter(self, param_snap, param_name, node, node_key):
        value_str, unit = self._convert_to_str(param_snap["value"], param_snap["unit"])
        # Omits printing of the date to make it more readable
        if param_snap["ts"] is not None:
            latest_str = param_snap["ts"][11:]
        else:
            latest_str = ""
        param_node_key = f"{node_key}.{param_name}"
        # If node does not yet exist, create a node
        if param_node_key not in self.nodes:
            param_node = QtWidgets.QTreeWidgetItem(
                [param_name, value_str, unit, latest_str]
            )
            node.addChild(param_node)
            self.nodes[param_node_key] = param_node
        else:  # else update existing node
            param_node = self.nodes[param_node_key]
            param_node.setData(1, 0, value_str)
            param_node.setData(2, 0, unit)
            param_node.setData(3, 0, latest_str)

    @staticmethod
    def _remove_left(
        in_string, to_be_removed
    ):  # todo: replace this method with str.removeprefix when at Python 3.9
        # Do not remove if to_be_removed matches the whole in_string
        if in_string != to_be_removed:
            try:
                _, out_string = in_string.split(f"{to_be_removed}_", 1)
                return out_string
            except ValueError:
                pass

        return in_string

    @deprecated(
        "0.10.0",
        "The function _get_entries_json provides similar functionality.",
    )
    def getNodes(self):
        return pprint.pformat(self.nodes)

    def _get_entries_json(self):
        """Get json encoding of entries of instrument monitor.

        This method is only used for testing. ``self.node`` cannot be returned
        to the test directly, because it lives in a different process. So this
        method provides a way to check the content of the available nodes. Not
        intended to be used for actual functionality.

        The dictionary returns one key-value pair for every entry in the table.
        The key has the format ``instrument.submodule.parameter``. Submodules
        and parameters are optional. Submodules can be nested (but only the key
        is nested, not the dictionary). The values are dictionaries with keys
        ``text0``, ``text1``, etc. for each column of the monitor.
        """

        # pylint: disable-next=too-few-public-methods
        class _QTreeWidgetEncoder(NumpyJSONEncoder):
            def default(self, obj: Any) -> Any:
                """Dedicated encoding method for QTreeWidgetItem"""
                if isinstance(obj, QtWidgets.QTreeWidgetItem):
                    return {
                        f"{key}{col}": getattr(obj, key)(col)
                        for key in ["text"]
                        for col in range(obj.columnCount())
                    }

                return super().default(obj)

        return json.dumps(self.nodes, cls=_QTreeWidgetEncoder)
