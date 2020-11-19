import numpy as np

from quantify.visualization.instrument_monitor import InstrumentMonitor
from quantify.visualization.ins_mon_widget.qc_snapshot_widget import QcSnaphotWidget

from tests.helpers import get_test_data_dir
from quantify.data.handling import set_datadir

from quantify.data.handling import snapshot


test_datadir = get_test_data_dir()


class TestInstrumentMonitor:

    @classmethod
    def setup_class(cls):
        cls.inst_mon = InstrumentMonitor(name='ins_mon_test')
        # ensures the default datadir is used which is excluded from git
        set_datadir(test_datadir)

    @classmethod
    def teardown_class(cls):
        cls.inst_mon.close()
        set_datadir(None)

    def test_attributes_created_during_init(self):
        hasattr(self.inst_mon, 'update_interval')

    def test_update_function(self):
        self.inst_mon.update()


class TestQcSnapshotWidget:

    @classmethod
    def setup_class(cls):
        cls.widget = QcSnaphotWidget()
        # ensures the default datadir is used which is excluded from git
        set_datadir(test_datadir)

    @classmethod
    def teardown_class(cls):
        cls.widget.close()
        set_datadir(None)

    def test_buildTreeSnapshot(self):
        test_snapshot = {'test_snapshot':
                            {'name': 'test_snapshot',
                             'parameters':
                                    {'snapshot':
                                            {
                                                'ts': 'latest',
                                                'label': "",
                                                'unit': '',
                                                'name': 'string_representation',
                                                'value': 1
                                            }
                                    }
                            }
                        }
        self.widget.buildTreeSnapshot(test_snapshot)
        assert self.widget.nodes
