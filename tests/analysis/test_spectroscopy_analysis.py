import quantify.data.handling as dh
from quantify.analysis import spectroscopy_analysis as sa
from tests.helpers import get_test_data_dir


def test_load_dataset():
    dh.set_datadir(get_test_data_dir())

    tuid = '20210118-202044-211-58ddb0'
    a = sa.ResonatorSpectroscopyAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert set(a.figs_mpl.keys()) == {'S21'}
