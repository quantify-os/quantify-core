import os
import pytest
import xarray as xr
import numpy as np
import quantify.measurement.data_handling as dh
from quantify.measurement.measurement_control import MeasurementControl
from datetime import datetime
from qcodes import ManualParameter
import quantify

test_datadir = os.path.join(os.path.split(
    quantify.__file__)[0], '..', 'tests', 'data')


def test_is_valid_dset():

    test_dset = xr.Dataset()
    assert dh.is_valid_dset(test_dset)


def test_gen_tuid():

    ts = datetime.now()

    tuid = dh.gen_tuid(ts)

    readable_ts = ts.strftime('%Y%m%d-%H%M%S-%f')[:-3]

    assert tuid[:19] == readable_ts
    assert len(tuid) == 26  # 6 random characters added at the end of tuid


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

def test_initialize_dataset_2D():
    xpar = ManualParameter('x', unit='m', label='X position')
    ypar = ManualParameter('y', unit='m', label='Y position')
    getpar = ManualParameter('z', unit='V', label='Signal amplitude')
    setable_pars = [xpar, ypar]
    setpoints = np.arange(0, 100, 32)
    setpoints = setpoints.reshape((len(setpoints)//2, 2))
    getable_pars = [getpar]

    dataset = dh.initialize_dataset(setable_pars, setpoints, getable_pars)


    assert isinstance(dataset, xr.Dataset)
    assert len(dataset.data_vars) == 3
    assert dataset.attrs.keys() == {'tuid'}
    assert set(dataset.variables.keys()) == {'x0', 'x1', 'y0'}




def test_getset_datadir():
    # here to ensure we always start with default datadir
    dh.set_datadir(None)

    default_datadir = dh.get_datadir()
    dd = os.path.split(default_datadir)
    assert dd[-1] == 'data'
    assert os.path.split(dd[-2])[-1] == 'quantify'

    dh.set_datadir('my_ddir')
    assert dh.get_datadir() == 'my_ddir'

    # Test resetting to default
    dh.set_datadir(None)
    assert dh.get_datadir() == default_datadir


def test_load_dataset():
    dh.set_datadir(test_datadir)
    tuid = '20200430-170837-315f36'
    dataset = dh.load_dataset(tuid=tuid)
    assert dataset.attrs['tuid'] == tuid

    tuid_short = '20200430-170837'
    dataset = dh.load_dataset(tuid=tuid_short)
    assert dataset.attrs['tuid'] == tuid

    with pytest.raises(FileNotFoundError):
        tuid = '20200430-170837-3b5f36'
        dh.load_dataset(tuid=tuid)

    with pytest.raises(FileNotFoundError):
        tuid = '20200230-170837'
        dh.load_dataset(tuid=tuid)


def test_get_latest_tuid_invalid_datadir():
    dh.set_datadir('some_invalid_datadir')
    with pytest.raises(FileNotFoundError):
        dh.get_latest_tuid()


def test_get_latest_tuid_empty_datadir():
    valid_dir_but_no_data = os.path.join(os.path.split(
        quantify.__file__)[0], '..', 'tests', 'measurement')
    dh.set_datadir(valid_dir_but_no_data)
    with pytest.raises(FileNotFoundError) as excinfo:
        dh.get_latest_tuid()
    assert "There are no valid day directories" in str(excinfo.value)


def test_get_latest_tuid_no_match():
    dh.set_datadir(test_datadir)
    with pytest.raises(FileNotFoundError) as excinfo:
        dh.get_latest_tuid(contains='nonexisting_label')
    assert "No experiment found containing" in str(excinfo.value)


def test_get_latest_tuid_correct_tuid():
    dh.set_datadir(test_datadir)
    tuid = dh.get_latest_tuid(contains='36-Cosine')
    exp_tuid = '20200430-170837-315f36'
    assert tuid == exp_tuid


def test_snapshot():
    empty_snap = dh.snapshot()
    assert empty_snap == {'instruments': {}, 'parameters': {}}
    test_MC = MeasurementControl(name='MC')

    test_MC.soft_avg(5)
    snap = dh.snapshot()
    assert snap['instruments'].keys() == {'MC'}
    assert snap['instruments']['MC']['parameters']['soft_avg']['value'] == 5

    test_MC.close()
