import xarray as xr
import quantify.measurement.data_handling as dh
from datetime import datetime


def test_is_valid_dset():

    test_dset = xr.Dataset()
    assert dh.is_valid_dset(test_dset)


def test_gen_tuid():

    ts = datetime.now()

    tuid = dh.gen_tuid(ts)

    readable_ts = ts.strftime('%Y%m%d-%H%M%S')

    assert tuid[:15] == readable_ts
    assert len(tuid) == 20  # 4 random characters added at the end of tuid
