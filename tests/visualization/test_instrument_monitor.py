from quantify.visualization.instrument_monitor import InstrumentMonitor
from quantify.visualization.ins_mon_widget.qc_snapshot_widget import QcSnaphotWidget
import pyqtgraph.multiprocess as pgmp


class TestInstrumentMonitor:
    @classmethod
    def setup_class(cls):
        cls.inst_mon = InstrumentMonitor(name="ins_mon_test")

    @classmethod
    def teardown_class(cls):
        cls.inst_mon.close()

    def test_attributes_created_during_init(self):
        hasattr(self.inst_mon, "update_interval")

    def test_update(self):
        self.inst_mon.update()

    def test_setGeometry(self):
        xywh = (300, 300, 600, 800)
        self.inst_mon.setGeometry(*xywh)
        widget = self.inst_mon.widget
        # N.B. x an y are absolute, OS docs or menu bars might prevent certain positions
        assert xywh[-2:] == (widget.width(), widget.height())


class TestQcSnapshotWidget:
    @classmethod
    def setup_class(cls):
        proc = pgmp.QtProcess(processRequests=False)  # pyqtgraph multiprocessing
        qc_widget = "quantify.visualization.ins_mon_widget.qc_snapshot_widget"
        r_qc_widget = proc._import(qc_widget)
        cls.widget = r_qc_widget.QcSnaphotWidget()

    @classmethod
    def teardown_class(cls):
        cls.widget.close()

    def test_buildTreeSnapshot(self):
        test_snapshot = {
            "test_snapshot": {
                "name": "test_snapshot",
                "parameters": {
                    "snapshot": {
                        "ts": "latest",
                        "label": "",
                        "unit": "",
                        "name": "string_representation",
                        "value": 1,
                    }
                },
            }
        }
        self.widget.buildTreeSnapshot(test_snapshot)
        nodes_str = self.widget.getNodes()
        assert "test_snapshot" in nodes_str and "QTreeWidgetItem" in nodes_str
