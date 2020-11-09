from pyqtgraph.Qt import QtGui
from quantify.visualization.SI_utilities import SI_val_to_msg_str


class QcSnaphotWidget(QtGui.QTreeWidget):

    """
    Widget for displaying QcoDes instrument snapshots.
    Heavily inspired by the DataTreeWidget.
    """

    def __init__(self, parent=None, data=None):
        QtGui.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(4)
        self.setHeaderLabels(['Name', 'Value', 'Unit', 'Last update'])
        self.nodes = {}

    def setData(self, data):
        """
        data should be a QCoDes snapshot of a station.
        """
        self.buildTreeSnapshot(snapshot=data)
        self.resizeColumnToContents(0)


    def buildTreeSnapshot(self, snapshot):
        # exists so that function can be called with no data in construction
        if snapshot is None:
            return

        parent = self.invisibleRootItem()

        instruments_in_snapshot = sorted(snapshot.keys())

        for ins in instruments_in_snapshot:
            current_instrument = snapshot[ins]
            # Name of the node in the self.nodes dictionary
            ins_name = current_instrument['name']
            if ins_name not in self.nodes:
                self.nodes[ins_name] = QtGui.QTreeWidgetItem([ins_name, "", ""])
                parent.addChild(self.nodes[ins_name])

            node = self.nodes[ins_name]

            for par_name in sorted(current_instrument['parameters'].keys()):
                par_snap = current_instrument['parameters'][par_name]
                # Depending on the type of data stored in value do different
                # things, currently only blocks non-dicts
                if 'value' in par_snap.keys():
                    # Some parameters do not have a value, these are not shown
                    # in the instrument monitor.
                    if not isinstance(par_snap['value'], dict):
                        value_str, unit = SI_val_to_msg_str(par_snap['value'],
                                                            par_snap['unit'])

                        # Omits printing of the date to make it more readable
                        if par_snap['ts'] is not None:
                            latest_str = par_snap['ts'][11:]
                        else:
                            latest_str = ''

                        param_node_name = '{}.{}'.format(ins_name, par_name)
                        # If node does not yet exist, create a node
                        if param_node_name not in self.nodes:
                            param_node = QtGui.QTreeWidgetItem(
                                [par_name, value_str, unit, latest_str])
                            node.addChild(param_node)
                            self.nodes[param_node_name] = param_node
                        else:  # else update existing node
                            param_node = self.nodes[param_node_name]
                            param_node.setData(1, 0, value_str)
                            param_node.setData(2, 0, unit)
                            param_node.setData(3, 0, latest_str)