import xarray as xr
import numpy as np
import quantify.measurement.data_handling as dh
from datetime import datetime
from qcodes import ManualParameter

def test_is_valid_dset():

    test_dset = xr.Dataset()
    assert dh.is_valid_dset(test_dset)


def test_gen_tuid():

    ts = datetime.now()

    tuid = dh.gen_tuid(ts)

    readable_ts = ts.strftime('%Y%m%d-%H%M%S')

    assert tuid[:15] == readable_ts
    assert len(tuid) == 22  # 6 random characters added at the end of tuid


def test_initialize_dataset():

    setpar = ManualParameter('x', unit='m', label='X position')
    getpar = ManualParameter('y', unit='V', label='Signal amplitude')
    setable_pars = [setpar]
    setpoints = np.arange(0, 100, 32)
    setpoints = setpoints.reshape((len(setpoints), 1))

    getable_pars = [getpar]
    dataset = dh.initialize_dataset(setable_pars, setpoints, getable_pars)

    assert isinstance(dataset, xr.Dataset)
    assert len(dataset.data_vars) == 2
    assert dataset.attrs.keys() == {'tuid'}
    assert dataset.variables.keys() == {'x0', 'y0'}

    x0 = dataset['x0']
    assert isinstance(x0, xr.DataArray)
    assert x0.attrs['unit'] == 'm'
    assert x0.attrs['name'] == 'x'
    assert x0.attrs['long_name'] == 'X position'

    y0 = dataset['y0']
    assert isinstance(y0, xr.DataArray)
    assert y0.attrs['unit'] == 'V'
    assert y0.attrs['name'] == 'y'
    assert y0.attrs['long_name'] == 'Signal amplitude'