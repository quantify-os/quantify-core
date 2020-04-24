import xarray as xr
import quantify.measurement.data_handling as dh


def test_is_valid_dset():


    test_dset = xr.Dataset()
    assert dh.is_valid_dset(test_dset) == True