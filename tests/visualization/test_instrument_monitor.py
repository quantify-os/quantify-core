from enum import Enum
import time
from unittest.mock import Mock

import pyqtgraph.multiprocess as pgmp
import pytest
from pyqtgraph.multiprocess.remoteproxy import ClosedError

from quantify_core.visualization.instrument_monitor import (
    InstrumentMonitor,
    RepeatTimer,
)
from quantify_core.visualization.ins_mon_widget.qc_snapshot_widget import (
    QcSnapshotWidget,
)


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
        self.inst_mon._update()

    def test_setGeometry(self):
        xywh = (400, 400, 300, 400)
        widget = self.inst_mon.widget
        wh_init = (widget.width(), widget.height())
        self.inst_mon.setGeometry(*xywh)
        # N.B. x an y are absolute, OS docs or menu bars might prevent certain positions
        assert wh_init != xywh[-2:]

    def test_change_update_interval(self):
        # first test default
        assert self.inst_mon.update_interval() == 5

        # then change it
        self.inst_mon.update_interval(10)
        assert self.inst_mon.update_interval() == 10


class TestRepeatTimer:
    def setup(self):
        self.mock_function = Mock()
        self.repeat_timer = RepeatTimer(
            0.1, self.mock_function, args=(1, 2), kwargs={"a": 3}
        )

    def test_attributes_created_during_init(self):
        hasattr(self.repeat_timer, "interval")

    def test_function_is_called_with_args_and_kwargs(self):
        self.repeat_timer.start()
        time.sleep(0.2)
        self.repeat_timer.cancel()

        self.mock_function.assert_called_with(1, 2, a=3)

    def test_timer_can_be_paused_and_not_call_the_function(self):
        self.repeat_timer.start()
        self.repeat_timer.pause()
        time.sleep(0.2)
        self.repeat_timer.unpause()
        self.repeat_timer.cancel()

        self.mock_function.assert_not_called()

    def test_timer_cannot_be_restarted_after_being_cancelled(self):
        # use pause/unpause instead
        self.repeat_timer.start()
        self.repeat_timer.cancel()
        with pytest.raises(RuntimeError):
            self.repeat_timer.start()


class TestQcSnapshotWidget:
    @classmethod
    def setup_class(cls):
        proc = pgmp.QtProcess(processRequests=False)  # pyqtgraph multiprocessing
        qc_widget = "quantify_core.visualization.ins_mon_widget.qc_snapshot_widget"
        r_qc_widget = proc._import(qc_widget, timeout=60)

        for i in range(10):
            try:
                cls.widget = r_qc_widget.QcSnapshotWidget()
            except (ClosedError, ConnectionResetError) as e:
                # the remote process might crash
                if i >= 9:
                    raise e
                time.sleep(0.2)
                r_qc_widget = proc._import(qc_widget, timeout=60)
            else:
                break

    @classmethod
    def teardown_class(cls):
        cls.widget.close()

    def test_buildTreeSnapshot(self):
        param = {
            "ts": "latest",
            "label": "",
            "unit": "",
            "name": "string_representation",
            "value": 1,
        }
        test_snapshot = {
            "test_snapshot": {
                "name": "test_snapshot",
                "parameters": {"snapshot": param},
                "submodules": {"sub1": {"parameters": {"param1": param}}},
                "channels": {"ch1": {"parameters": {"param1": param}}},
                "others": {"other1": {"parameters": {"param1": param}}},
            }
        }
        self.widget.buildTreeSnapshot(test_snapshot)
        nodes_str = self.widget.getNodes()
        assert (
            "test_snapshot" in nodes_str
            and "QTreeWidgetItem" in nodes_str
            and "test_snapshot.sub1.param1" in nodes_str
            and "test_snapshot.ch1.param1" in nodes_str
            and "test_snapshot.other1.param1" not in nodes_str
        )


def test_parameter_conversion():
    """Test conversion of snapshot parameters to displayed values."""
    convert = QcSnapshotWidget._convert_to_str

    # Test SI unit conversion
    assert convert(1.1, "m") == ("1.1", "m")
    assert convert(1, "m") == ("1", "m")
    assert convert(1.0, "m") == ("1.0", "m")
    assert convert(0.9, "m") == ("900.0", "mm")
    assert convert(0.00004, "m") == ("40.0", "μm")
    assert convert(0.0000004321, "m") == ("432.1", "nm")
    assert convert(1050, "m") == ("1.05", "km")
    assert convert(2000, "m") == ("2", "km")
    assert convert(100, "mm") == ("100", "mm")
    assert convert(0.08, "mm") == ("80.0", "μm")
    assert convert(1234, "nm") == ("1.234", "μm")
    assert convert(None, "nm") == ("None", "nm")

    # Test value formatting without unit
    assert convert(None, "") == ("None", "")
    assert convert(None, None) == ("None", "")
    assert convert(1, "") == ("1", "")
    assert convert(1.0, "") == ("1.0", "")
    assert convert(True, "") == ("True", "")

    class TestEnum(Enum):
        """Dummy enum to test conversion"""

        VAL1 = 0
        VAL2 = 1

    assert convert(TestEnum.VAL1, "") == ("VAL1", "")
    assert convert(TestEnum.VAL2, "") == ("VAL2", "")
