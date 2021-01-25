import quantify.data.handling as dh
from quantify.analysis import base_analysis as ba
from tests.helpers import get_test_data_dir


def test_load_dataset():
    dh.set_datadir(get_test_data_dir())

    tuid = '20200430-170837-001-315f36'
    a = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert list(a.figs.keys()) == ['x0-y0']

    tuid = '20210118-202044-211-58ddb0'
    a = ba.Basic1DAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert list(a.figs.keys()) == ['x0-y0', 'x0-y1']
