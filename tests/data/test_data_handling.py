# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import gc
import json
import os
import bz2
import gzip
import lzma
from pathlib import Path
import pytest
from quantify_core.data.handling import load_snapshot
import shutil
import tempfile
from datetime import datetime

import dateutil
import numpy as np
import uncertainties
import xarray as xr
from qcodes import Instrument, ManualParameter, InstrumentChannel
from qcodes.utils.helpers import NumpyJSONEncoder

import quantify_core.data.handling as dh
from quantify_core.analysis.base_analysis import BasicAnalysis
from quantify_core.data.types import TUID
from quantify_core.measurement.control import MeasurementControl

TUID_1D_1PLOT = "20200430-170837-001-315f36"


def test_gen_tuid():
    ts = datetime.now()
    tuid = dh.gen_tuid(ts)

    assert TUID.is_valid(tuid)
    assert isinstance(tuid, str)


def test_initialize_dataset():
    setpar = ManualParameter("x", unit="m", label="X position")
    getpar = ManualParameter("y", unit="V", label="Signal amplitude")
    setable_pars = [setpar]
    setpoints = np.arange(0, 100, 32)
    setpoints = setpoints.reshape((len(setpoints), 1))

    getable_pars = [getpar]
    dataset = dh.initialize_dataset(setable_pars, setpoints, getable_pars)

    assert isinstance(dataset, xr.Dataset)
    assert len(dataset.data_vars) == 1
    assert dataset.attrs.keys() == {"tuid"}
    assert dataset.variables.keys() == {"x0", "y0"}

    x0 = dataset["x0"]
    assert isinstance(x0, xr.DataArray)
    assert x0.attrs["units"] == "m"
    assert x0.attrs["name"] == "x"
    assert x0.attrs["long_name"] == "X position"

    y0 = dataset["y0"]
    assert isinstance(y0, xr.DataArray)
    assert y0.attrs["units"] == "V"
    assert y0.attrs["name"] == "y"
    assert y0.attrs["long_name"] == "Signal amplitude"

    assert set(dataset.coords.keys()) == {"x0"}
    assert set(dataset.dims) == {"dim_0"}


def test_initialize_dataset_2d():
    xpar = ManualParameter("x", unit="m", label="X position")
    ypar = ManualParameter("y", unit="m", label="Y position")
    getpar = ManualParameter("z", unit="V", label="Signal amplitude")
    setable_pars = [xpar, ypar]
    setpoints = np.arange(0, 100, 32)
    setpoints = setpoints.reshape((len(setpoints) // 2, 2))
    getable_pars = [getpar]

    dataset = dh.initialize_dataset(setable_pars, setpoints, getable_pars)

    assert isinstance(dataset, xr.Dataset)
    assert len(dataset.data_vars) == 1
    assert dataset.attrs.keys() == {"tuid"}
    assert set(dataset.variables.keys()) == {"x0", "x1", "y0"}
    assert set(dataset.coords.keys()) == {"x0", "x1"}


def test_set_datadir(tmp_test_data_dir):
    # Test valid directory creation
    new_dir_path = os.path.join(tmp_test_data_dir, "test_datadir")
    dh.set_datadir(new_dir_path)
    assert os.access(new_dir_path, os.W_OK)

    # Test setting to None
    dh.set_datadir(None)
    assert dh.get_datadir() == dh.default_datadir()

    # Test setting invalid type (not a directory)
    with pytest.raises(TypeError):
        dh.set_datadir(5)

    # Test setting to empty str
    with pytest.raises(FileNotFoundError):
        dh.set_datadir("")


def test_get_datadir(tmp_test_data_dir):
    dh._datadir = None

    with pytest.raises(NotADirectoryError):
        # Ensure users are forced to pick a datadir in order to avoid
        # potential dataloss
        dh.get_datadir()

    new_dir_path = os.path.join(tmp_test_data_dir, "test_datadir2")
    os.mkdir(new_dir_path)
    dh.set_datadir(new_dir_path)
    assert os.path.split(dh.get_datadir())[-1] == "test_datadir2"


def test_load_dataset(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = "20200430-170837-001-315f36"
    dataset = dh.load_dataset(tuid=tuid)
    assert dataset.attrs["tuid"] == tuid

    tuid_short = "20200430-170837"
    dataset = dh.load_dataset(tuid=tuid_short)
    assert dataset.attrs["tuid"] == tuid

    with pytest.raises(FileNotFoundError):
        tuid = "20200430-170837-001-3b5f36"
        dh.load_dataset(tuid=tuid)

    with pytest.raises(FileNotFoundError):
        tuid = "20200230-001-170837"
        dh.load_dataset(tuid=tuid)


def test_get_latest_tuid_empty_datadir(tmp_test_data_dir):
    valid_dir_but_no_data = tmp_test_data_dir / "empty"
    dh.set_datadir(valid_dir_but_no_data)
    with pytest.raises(FileNotFoundError) as excinfo:
        dh.get_latest_tuid()
    assert "There are no valid day directories" in str(excinfo.value)


def test_get_latest_tuid_no_match(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    with pytest.raises(FileNotFoundError) as excinfo:
        dh.get_latest_tuid(contains="nonexisting_label")
    assert "No experiment found containing" in str(excinfo.value)


def test_get_latest_tuid_correct_tuid(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = dh.get_latest_tuid(contains="36-Cosine")
    exp_tuid = "20200430-170837-001-315f36"
    assert tuid == exp_tuid


def test_get_tuids_containing(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuids = dh.get_tuids_containing("Cosine test")
    assert len(tuids) == 2
    assert tuids[0] == "20200430-170837-001-315f36"
    assert tuids[1] == "20200504-191556-002-4209ee"


def test_get_tuids_containing_between_strings(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)

    t_start = "20201124"
    t_stop = "20201125"

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop)
    assert tuids == [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
        "20201124-184722-988-0463d4",
        "20201124-184729-618-85970f",
        "20201124-184736-341-3628d4",
    ]

    t_start = "20201124-180000"
    t_stop = "20201124-190000"

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop)
    assert tuids == [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
        "20201124-184722-988-0463d4",
        "20201124-184729-618-85970f",
        "20201124-184736-341-3628d4",
    ]

    t_start = "20201124-18:00:00"
    t_stop = "20201124-18:47:30"

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop)
    assert tuids == [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
        "20201124-184722-988-0463d4",
        "20201124-184729-618-85970f",
    ]

    t_start = "2020/11/24 18:00:00"
    t_stop = "2020/11/24 18:47:30"

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop)
    assert tuids == [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
        "20201124-184722-988-0463d4",
        "20201124-184729-618-85970f",
    ]

    t_start = "20201124-180000"
    t_stop = "20201124-184725"

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop)
    assert tuids == [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
        "20201124-184722-988-0463d4",
    ]


def test_get_tuids_containing_between_exclusive_t_stop(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    t_start = "20201124-180000"
    t_stop = "20201124-184722"  # test if t_stop is inclusive

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop)
    assert tuids == [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
    ]


def test_get_tuids_containing_between_inclusive_t_start(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    t_start = "20201124-184709"  # test if t_stop is inclusive
    t_stop = "20201124-184723"

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop)
    assert tuids == [
        "20201124-184709-137-8a5112",
        "20201124-184716-237-918bee",
        "20201124-184722-988-0463d4",
    ]


def test_get_tuids_containing_reverse(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    t_start = "20200814"
    t_stop = "20201124-190000"

    tuids = dh.get_tuids_containing("", t_start=t_start, t_stop=t_stop, reverse=True)
    assert tuids == [
        "20201124-184736-341-3628d4",
        "20201124-184729-618-85970f",
        "20201124-184722-988-0463d4",
        "20201124-184716-237-918bee",
        "20201124-184709-137-8a5112",
        "20200814-134652-492-fbf254",
    ]


def test_get_tuids_containing_between_datetimes(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    t_start = "20200430-170836"
    t_stop = "20200504-200000"
    t_start = dateutil.parser.parse(t_start)
    t_stop = dateutil.parser.parse(t_stop)

    tuids = dh.get_tuids_containing("Cosine test", t_start=t_start, t_stop=t_stop)
    assert tuids[0] == "20200430-170837-001-315f36"
    assert tuids[1] == "20200504-191556-002-4209ee"


def test_get_tuids_containing_options(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)

    tuids = dh.get_tuids_containing("Cosine test", t_start="20200501")
    assert len(tuids) == 1
    assert tuids[0] == "20200504-191556-002-4209ee"

    tuids = dh.get_tuids_containing("Cosine test", t_stop="20200501")
    assert len(tuids) == 1
    assert tuids[0] == "20200430-170837-001-315f36"

    tuids = dh.get_tuids_containing("Cosine test", t_start="20200430")
    assert tuids == ["20200430-170837-001-315f36", "20200504-191556-002-4209ee"]

    tuids = dh.get_tuids_containing(
        "Cosine test", t_start="20200430", t_stop="20200504"
    )
    assert len(tuids) == 1
    assert tuids[0] == "20200430-170837-001-315f36"


def test_get_tuids_containing_max_results(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuids = dh.get_tuids_containing(
        "Cosine test",
        t_start="20200430",
        t_stop="20200505",
        max_results=1,
        reverse=True,
    )
    assert len(tuids) == 1
    assert tuids == ["20200504-191556-002-4209ee"]


def test_get_tuids_containing_none_arg(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    for empties in [
        ("20200505", None),
        (None, "20200430"),
        ("20200410", "20200415"),
        ("20200510", "20200520"),
    ]:
        with pytest.raises(FileNotFoundError):
            dh.get_tuids_containing(
                "Cosine test", t_start=empties[0], t_stop=empties[1]
            )


def test_misplaced_exp_container(tmp_test_data_dir):
    """
    Ensures user is warned if a dataset was misplaced
    """
    tmp_data_path = os.path.join(
        tmp_test_data_dir,
        "misplaced_exp_container",
    )
    date = "20201006"
    container = "20201008-191556-002-4209eg-Experiment from my colleague"
    os.makedirs(os.path.join(tmp_data_path, date, container), exist_ok=True)
    dh.set_datadir(tmp_data_path)
    with pytest.raises(FileNotFoundError):
        dh.get_tuids_containing(contains="colleague")

    # cleanup
    shutil.rmtree(tmp_data_path)


def test_locate_experiment_container(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = "20200430-170837-001-315f36"
    experiment_container = dh.locate_experiment_container(tuid=tuid)
    path_parts = Path(experiment_container).parts

    assert tuid in path_parts[-1]
    assert "Cosine test" in path_parts[-1]
    assert path_parts[-2] == "20200430"
    assert path_parts[-3] == os.path.split(tmp_test_data_dir)[-1]


def test_load_dataset_from_path(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)

    tuid = "20200430-170837-001-315f36"
    path = Path(dh.locate_experiment_container(tuid=tuid)) / dh.DATASET_NAME
    dataset = dh.load_dataset_from_path(path)
    assert dataset.attrs["tuid"] == tuid


def test_dh_load_dataset_complex_numbers():
    complex_float = 1.0 + 5.0j
    complex_int = 1 + 4j
    dataset = mk_dataset_complex_array(
        complex_float=complex_float, complex_int=complex_int
    )
    tmp_dir = tempfile.TemporaryDirectory()
    path = Path(tmp_dir.name) / dh.DATASET_NAME
    dataset.to_netcdf(path, engine="h5netcdf", invalid_netcdf=True)

    load_dataset = dh.load_dataset_from_path(path)
    assert load_dataset.y0.values[0] == complex_int
    assert load_dataset.y1.values[0] == complex_float
    tmp_dir.cleanup()


def test_write_quantify_dataset():
    complex_float = 1.0 + 6.0j
    complex_int = 1 + 3j
    dataset = mk_dataset_complex_array(
        complex_float=complex_float, complex_int=complex_int
    )

    tmp_dir = tempfile.TemporaryDirectory()
    path = Path(tmp_dir.name) / dh.DATASET_NAME
    dh.write_dataset(path, dataset)

    load_dataset = xr.load_dataset(path, engine="h5netcdf")
    assert load_dataset.y0.values[0] == complex_int
    assert load_dataset.y1.values[0] == complex_float
    tmp_dir.cleanup()


def test_write_quantify_dataset_np_bool():
    dataset = mk_dataset_complex_array(complex_float=1.0j, complex_int=2.0j)

    dataset.attrs["test_attr"] = np.bool_(True)
    dataset.x0.attrs["test_attr"] = np.bool_(True)
    dataset.y0.attrs["test_attr"] = np.bool_(False)

    tmp_dir = tempfile.TemporaryDirectory()
    path = Path(tmp_dir.name) / dh.DATASET_NAME
    dh.write_dataset(path, dataset)

    load_dataset = xr.load_dataset(path, engine="h5netcdf")

    assert bool(load_dataset.attrs["test_attr"]) == True
    assert bool(load_dataset.x0.attrs["test_attr"]) == True
    assert bool(load_dataset.y0.attrs["test_attr"]) == False

    tmp_dir.cleanup()


def test_snapshot():
    empty_snap = dh.snapshot()
    assert empty_snap == {"instruments": {}, "parameters": {}}
    test_MC = MeasurementControl(name="MC")

    test_MC.update_interval(0.77)

    snap = dh.snapshot()

    assert snap["instruments"].keys() == {"MC"}
    assert snap["instruments"]["MC"]["parameters"]["update_interval"]["value"] == 0.77

    test_MC.close()


def test_snapshot_list(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = "20200430-170837-001-315f36"

    experiment_directory = dh.locate_experiment_container(tuid=tuid)
    snap = dh.snapshot()

    snap["instruments"]["test_instrument"] = {"parameters": {"test_list": [1, 2, 3]}}

    full_path_to_file = f"{experiment_directory}/test_snapshot_list.json"
    with open(full_path_to_file, "w", encoding="utf-8") as file:
        json.dump(snap, file, cls=NumpyJSONEncoder, indent=4)
    decoded_snapshot = dh.load_snapshot(
        tuid=tuid,
        list_to_ndarray=True,
        file="test_snapshot_list.json",
    )
    assert isinstance(
        decoded_snapshot["instruments"]["test_instrument"]["parameters"]["test_list"],
        np.ndarray,
    )


def test_snapshot_dead_instruments():
    """Ensure that the snapshot does not attempt to access dead instruments."""
    instrument_a = Instrument("a")
    instrument_b = Instrument("b")
    instrument_a = 123
    gc.collect()

    snap = dh.snapshot()

    assert "a" not in dh.snapshot()["instruments"]
    assert dh.snapshot()["instruments"].keys() == {"b"}


def test_dynamic_dataset():
    x = ManualParameter("x", unit="m", label="X position")
    y = ManualParameter("y", unit="m", label="Y position")
    z = ManualParameter("z", unit="V", label="Signal amplitude")
    settables = [x, y]
    gettables = [z]
    dset = dh.initialize_dataset(settables, np.empty((len(settables), 8)), gettables)

    x0_vals = np.random.random(8)
    x1_vals = np.random.random(8)
    y0_vals = np.random.random(8)
    dset["x0"].values[:] = x0_vals
    dset["x1"].values[:] = x1_vals
    dset["y0"].values[:] = y0_vals

    dset_grown = dh.grow_dataset(dset)
    assert len(dset_grown["x0"]) == len(dset_grown["x1"]) == len(dset_grown["y0"]) == 16
    assert np.isnan(dset_grown["x0"][8:]).all()
    assert np.isnan(dset_grown["x1"][8:]).all()
    assert np.isnan(dset_grown["y0"][8:]).all()
    assert dset_grown.attrs == dset.attrs
    assert dset_grown.x0.attrs == dset.x0.attrs
    assert dset_grown.x1.attrs == dset.x1.attrs
    assert dset_grown.y0.attrs == dset.y0.attrs

    x0_vals_ext = np.random.random(3)
    x1_vals_ext = np.random.random(3)
    y0_vals_ext = np.random.random(3)

    dset_grown["x0"].values[8:11] = x0_vals_ext
    dset_grown["x1"].values[8:11] = x1_vals_ext
    dset_grown["y0"].values[8:11] = y0_vals_ext

    dset_trimmed = dh.trim_dataset(dset_grown)
    assert (
        len(dset_trimmed["x0"])
        == len(dset_trimmed["x1"])
        == len(dset_trimmed["y0"])
        == 11
    )
    np.array_equal(dset_trimmed["x0"], np.concatenate([x0_vals, x0_vals_ext]))
    np.array_equal(dset_trimmed["x1"], np.concatenate([x1_vals, x1_vals_ext]))
    np.array_equal(dset_trimmed["y0"], np.concatenate([y0_vals, y0_vals_ext]))

    assert not np.isnan(dset_trimmed["x0"]).any()
    assert not np.isnan(dset_trimmed["x1"]).any()
    assert not np.isnan(dset_trimmed["y0"]).any()
    assert dset_trimmed.attrs == dset.attrs
    assert dset_trimmed.x0.attrs == dset.x0.attrs
    assert dset_trimmed.x1.attrs == dset.x1.attrs
    assert dset_trimmed.y0.attrs == dset.y0.attrs

    assert "tuid" in dset_trimmed.attrs


def test_to_gridded_dataset(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = "20200504-191556-002-4209ee"
    dset_orig = dh.load_dataset(tuid)
    dset_gridded = dh.to_gridded_dataset(dset_orig)

    assert dset_gridded.attrs["tuid"] == tuid
    assert tuple(dset_gridded.dims) == ("x0", "x1")
    assert tuple(dset_gridded.coords.keys()) == ("x0", "x1")
    assert tuple(dset_gridded.sizes.values()) == (50, 11)
    assert dset_gridded["y0"].dims == ("x0", "x1")
    assert len(dset_gridded["x0"].values) == len(np.unique(dset_orig["x0"]))

    for var in ("x0", "x1", "y0"):
        assert dset_orig[var].attrs == dset_gridded[var].attrs

    y0 = dset_gridded["y0"].values

    indices = [[10, 9], [22, 6], [7, 3], [10, 1], [18, 1], [0, 3]]
    expected = [
        -0.7983563142002691,
        0.1436698700195456,
        0.2493959207434933,
        0.7983563142002691,
        -0.6970549632987115,
        -0.3999999999999999,
    ]

    assert [y0[tuple(idxs)] for idxs in indices] == expected


def mk_dataset_complex_array(complex_float=1.0 + 5.0j, complex_int=1 + 4j):
    dataset = xr.Dataset(
        data_vars={
            "y0": ("dim_0", np.array([complex_int] * 5)),
            "y1": ("dim_0", np.array([complex_float] * 5)),
        },
        coords={"x0": np.linspace(0, 5, 5)},
    )
    return dataset


def test_qcodes_numpyjsonencoder():
    quantities_of_interest = {
        "python_tuple": (1, 2, 3, 4),
        "python_list": [1, 2, 3, 4],
        "numpy_array": np.array([1, 2, 3, 4]),
        "numpy_array_complex": np.array([1, 2], dtype=complex),
        "uncertainties_ufloat": uncertainties.ufloat(1.0, 2.0),
        "complex": complex(1.0, 2.0),
        "nan_value": float("nan"),
        "inf_value": float("inf"),
        "-inf_value": float("-inf"),
    }

    encoded = json.dumps(quantities_of_interest, cls=NumpyJSONEncoder, indent=4)
    decoded = json.loads(encoded, cls=dh.DecodeToNumpy)

    assert isinstance(decoded["python_tuple"], list)
    assert isinstance(decoded["python_list"], list)
    assert isinstance(decoded["numpy_array"], (list, np.ndarray))
    assert decoded["uncertainties_ufloat"] == {
        "__dtype__": "UFloat",
        "nominal_value": 1.0,
        "std_dev": 2.0,
    }
    assert decoded["complex"] == {"__dtype__": "complex", "re": 1.0, "im": 2.0}
    assert decoded["numpy_array_complex"] == [
        {"__dtype__": "complex", "re": 1.0, "im": 0.0},
        {"__dtype__": "complex", "re": 2.0, "im": 0.0},
    ]
    assert np.isnan(decoded["nan_value"])
    assert decoded["inf_value"] == float("inf")
    assert decoded["-inf_value"] == float("-inf")


def test_decode_list_to_numpy():
    quantities_of_interest = {
        "python_tuple": (1, 2, 3, 4),
        "python_list": [1, 2, 3, 4],
        "numpy_array": np.array([1, 2, 3, 4]),
        "numpy_array_complex": np.array([1, 2], dtype=complex),
        "uncertainties_ufloat": uncertainties.ufloat(1.0, 2.0),
        "complex": complex(1.0, 2.0),
        "nan_value": float("nan"),
        "inf_value": float("inf"),
        "-inf_value": float("-inf"),
    }

    encoded = json.dumps(quantities_of_interest, cls=NumpyJSONEncoder, indent=4)
    decoded = json.loads(encoded, cls=dh.DecodeToNumpy, list_to_ndarray=True)

    assert isinstance(decoded["python_tuple"], np.ndarray)
    assert isinstance(decoded["python_list"], np.ndarray)
    assert isinstance(decoded["numpy_array"], np.ndarray)
    assert decoded["uncertainties_ufloat"] == {
        "__dtype__": "UFloat",
        "nominal_value": 1.0,
        "std_dev": 2.0,
    }
    assert decoded["complex"] == {"__dtype__": "complex", "re": 1.0, "im": 2.0}
    assert (
        decoded["numpy_array_complex"]
        == np.array(
            [
                {"__dtype__": "complex", "re": 1.0, "im": 0.0},
                {"__dtype__": "complex", "re": 2.0, "im": 0.0},
            ]
        )
    ).all()
    assert np.isnan(decoded["nan_value"])
    assert decoded["inf_value"] == float("inf")
    assert decoded["-inf_value"] == float("-inf")


def test_load_analysis_output_files(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)

    # We need to run an analysis first, so the files to be loaded are generated
    BasicAnalysis(tuid=TUID_1D_1PLOT).run()

    assert isinstance(
        dh.load_quantities_of_interest(TUID_1D_1PLOT, BasicAnalysis.__name__), dict
    )
    assert isinstance(
        dh.load_processed_dataset(TUID_1D_1PLOT, BasicAnalysis.__name__), xr.Dataset
    )


def test_is_uniformly_spaced_array():
    x0 = [1, 1.1, 1.2, 1.3, 1.4]
    x0_ng = [1, 1.1, 1.2, 1.3, 1.401]

    x1 = [1e-9, 1.1e-9, 1.2e-9, 1.3e-9, 1.4e-9]
    x1_ng = [1e-9, 1.1e-9, 1.2e-9, 1.3e-9, 1.401e-9]

    x2 = [1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9]
    x2_ng = [1e9, 1.1e9, 1.2e9, 1.3e9, 1.401e9]

    cases = [
        (True, x0),
        (True, x1),
        (True, x2),
        (False, x0_ng),
        (False, x1_ng),
        (False, x2_ng),
    ]

    for expected, points in cases:
        assert dh._is_uniformly_spaced_array(points) == expected


def test_concat_dataset(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)

    correct_tuids = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2021-10-29", t_stop="2021-10-30"
    )
    tuid_string = "test"
    tuid_wrong_list = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2021-10-29", t_stop="2021-10-30"
    ) + [5]
    tuid_wrong_values = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2021-10-29", t_stop="2021-10-30"
    ) + ["test"]
    dim_non_existing = "main_dim"

    with pytest.raises(TypeError):
        dh.concat_dataset(tuid_string)

    with pytest.raises(TypeError):
        dh.concat_dataset(tuid_wrong_list)

    with pytest.raises(FileNotFoundError):
        dh.concat_dataset(tuid_wrong_values)

    with pytest.raises(KeyError):
        dh.concat_dataset(correct_tuids, dim=dim_non_existing)

    new_dataset = dh.concat_dataset(correct_tuids)
    assert isinstance(new_dataset, xr.Dataset)
    assert len(new_dataset.dim_0) == 720
    assert isinstance(new_dataset["ref_tuids"], xr.DataArray)
    assert len(new_dataset["ref_tuids"]) == 720
    assert new_dataset["ref_tuids"].is_dataset_ref
    assert new_dataset.name == "Pulsed spectroscopy q4"


def test_concat_dataset_different_names(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)

    tuids = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2022-09-30", t_stop="2022-10-01"
    )

    new_dataset = dh.concat_dataset(tuids, name="Pulsed spectroscopy concat")
    assert new_dataset.name == "Pulsed spectroscopy concat"

    new_dataset = dh.concat_dataset(tuids)
    assert new_dataset.name == "Pulsed spectroscopy 0"

    # Concatenate the processed datasets
    new_dataset = dh.concat_dataset(
        tuids, analysis_name="QubitSpectroscopyAnalysis", dim="x0"
    )
    assert isinstance(new_dataset, xr.Dataset)
    assert len(new_dataset.x0) == 720
    assert isinstance(new_dataset["ref_tuids"], xr.DataArray)
    assert len(new_dataset["ref_tuids"]) == 720
    assert new_dataset["ref_tuids"].is_dataset_ref


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="function", autouse=False)
def mock_instr_nested(request):
    """
    Set up an instrument with a sub module with the following structure

    instr
    -> a
    -> mod_a
        -> b
    -> mod_b
        -> mod_c
            -> c
    """

    instr = Instrument("DummyInstrument")

    instr.add_parameter("a", parameter_class=ManualParameter, unit="Hz")

    mod_a = InstrumentChannel(instr, "mod_a")
    mod_a.add_parameter("b", parameter_class=ManualParameter)
    instr.add_submodule("mod_a", mod_a)

    mod_b = InstrumentChannel(instr, "mod_b")
    mod_c = InstrumentChannel(instr, "mod_c")
    mod_b.add_submodule("mod_c", mod_c)
    mod_c.add_parameter("c", parameter_class=ManualParameter)

    instr.add_submodule("mod_b", mod_b)

    def cleanup_instruments():
        instr.close()

    request.addfinalizer(cleanup_instruments)

    return instr


# pylint: disable=invalid-name
def test_extract_parameter_from_snapshot(tmp_test_data_dir, mock_instr_nested):
    """
    Test that we can extract parameters from a snapshot, including those
    which are contained within submodules
    """
    # Always set datadir before instruments
    dh.set_datadir(tmp_test_data_dir)

    # set some random values
    mock_instr_nested.a(23)
    mock_instr_nested.mod_a.b(42)
    mock_instr_nested.mod_b.mod_c.c(23.1)

    # create snapshot
    snap = dh.snapshot()

    a = dh.extract_parameter_from_snapshot(snap, "DummyInstrument.a")["value"]
    a_unit = dh.extract_parameter_from_snapshot(snap, "DummyInstrument.a")["unit"]
    b = dh.extract_parameter_from_snapshot(snap, "DummyInstrument.mod_a.b")["value"]
    c = dh.extract_parameter_from_snapshot(snap, "DummyInstrument.mod_b.mod_c.c")[
        "value"
    ]

    assert a == 23
    assert a_unit == "Hz"
    assert b == 42
    assert c == 23.1


def test_get_varying_parameter(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    parameter = "fluxcurrent.FBL_4"
    correct_tuids = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2021-10-29", t_stop="2021-10-30"
    )
    values = np.array([-0.00100625, -0.000996875, -0.0009875])

    varying_parameter_values = dh.get_varying_parameter_values(correct_tuids, parameter)
    assert isinstance(varying_parameter_values, np.ndarray)
    assert len(varying_parameter_values) == len(correct_tuids)
    assert varying_parameter_values == pytest.approx(values)


def test_get_varying_parameter_error(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    parameter = "fluxcurrent.FBL_4"
    non_existing_parameter = "fluxcurrent.FBL_5"
    correct_tuids = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2021-10-29", t_stop="2021-10-30"
    )
    tuid_string = "test"
    tuid_wrong_list = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2021-10-29", t_stop="2021-10-30"
    ) + [5]
    tuid_wrong_values = dh.get_tuids_containing(
        "Pulsed spectroscopy", t_start="2021-10-29", t_stop="2021-10-30"
    ) + ["test"]

    with pytest.raises(ValueError):
        dh.get_varying_parameter_values(tuid_string, parameter)

    with pytest.raises(TypeError):
        dh.get_varying_parameter_values(tuid_wrong_list, parameter)

    with pytest.raises(ValueError):
        dh.get_varying_parameter_values(tuid_wrong_values, parameter)

    with pytest.raises(KeyError):
        dh.get_varying_parameter_values(correct_tuids, non_existing_parameter)


@pytest.mark.parametrize("new_name", [None, "concat"])
def test_multi_experiment_data_extractor(tmp_test_data_dir, new_name):
    dh.set_datadir(tmp_test_data_dir)
    t_start = "20211029"
    t_stop = "20211030"
    parameter = "fluxcurrent.FBL_4"
    experiment = "Pulsed spectroscopy"
    experiment_wrong_type = 5
    expected_varying_parameter_values = np.array(
        [-0.00100625, -0.000996875, -0.0009875]
    )

    with pytest.raises(TypeError):
        dh.multi_experiment_data_extractor(
            experiment_wrong_type,
            parameter,
            new_name=new_name,
            t_start=t_start,
            t_stop=t_stop,
        )

    # Test filling in all parameters and new_name=None
    new_dataset = dh.multi_experiment_data_extractor(
        experiment,
        parameter,
        new_name=new_name,
        t_start=t_start,
        t_stop=t_stop,
    )
    assert len(new_dataset.dim_0) == 720
    assert TUID.is_valid(new_dataset.attrs["tuid"])
    assert isinstance(new_dataset.attrs["tuid"], str)
    assert new_dataset.x1.values == pytest.approx(
        np.repeat(expected_varying_parameter_values, 240)
    )
    if new_name is None:
        assert new_dataset.name == f"{experiment} vs {parameter}"
    else:
        assert new_dataset.name == new_name

    # Try concatenating the processed datasets
    new_dataset = dh.multi_experiment_data_extractor(
        experiment,
        parameter,
        new_name=new_name,
        t_start=t_start,
        t_stop=t_stop,
        analysis_name="QubitSpectroscopyAnalysis",
        dimension="x0",
    )
    assert len(new_dataset.dim_0) == 720
    assert TUID.is_valid(new_dataset.attrs["tuid"])
    assert isinstance(new_dataset.attrs["tuid"], str)
    assert new_dataset.x1.values == pytest.approx(
        np.repeat(expected_varying_parameter_values, 240)
    )
    if new_name is None:
        assert new_dataset.name == f"{experiment} vs {parameter}"
    else:
        assert new_dataset.name == new_name


def test_instrument_submodule_settable(mock_instr_nested):
    # Add parameter to test for edge case where parameters with same name but different
    # submodules are found correctly
    mock_instr_nested.add_parameter("c", parameter_class=ManualParameter)

    a = mock_instr_nested.a
    lst = dh._instrument_submodules_settable(a)
    assert len(lst) == 2
    assert mock_instr_nested in lst
    assert a in lst

    b = mock_instr_nested.mod_a.b
    lst = dh._instrument_submodules_settable(b)
    assert len(lst) == 3
    assert mock_instr_nested in lst
    assert mock_instr_nested.mod_a in lst
    assert b in lst

    c = mock_instr_nested.mod_b.mod_c.c
    lst = dh._instrument_submodules_settable(c)
    assert len(lst) == 4
    assert mock_instr_nested in lst
    assert mock_instr_nested.mod_b in lst
    assert mock_instr_nested.mod_b.mod_c in lst
    assert c in lst
    assert mock_instr_nested.c not in lst

    p = ManualParameter("test_parameter", label="Test parameter", unit="Hz")
    lst = dh._instrument_submodules_settable(p)
    assert len(lst) == 1
    assert p in lst

    c = mock_instr_nested.c
    lst = dh._instrument_submodules_settable(c)
    assert len(lst) == 2
    assert mock_instr_nested in lst
    assert c in lst
    assert mock_instr_nested.mod_b not in lst
    assert mock_instr_nested.mod_b.mod_c not in lst
    assert mock_instr_nested.mod_b.mod_c.c not in lst


def test_generate_long_name(mock_instr_nested):
    a = mock_instr_nested.a
    name = dh._generate_long_name(a)
    assert name == "DummyInstrument a"

    b = mock_instr_nested.mod_a.b
    name = dh._generate_long_name(b)
    assert name == "DummyInstrument mod_a b"

    c = mock_instr_nested.mod_b.mod_c.c
    name = dh._generate_long_name(c)
    assert name == "DummyInstrument mod_b mod_c c"

    p = ManualParameter("test_parameter", label="Test parameter", unit="Hz")
    name = dh._generate_long_name(p)
    assert name == "Test parameter"


def test_generate_name(mock_instr_nested):
    a = mock_instr_nested.a
    name = dh._generate_name(a)
    assert name == "DummyInstrument.a"

    b = mock_instr_nested.mod_a.b
    name = dh._generate_name(b)
    assert name == "DummyInstrument.mod_a.b"

    c = mock_instr_nested.mod_b.mod_c.c
    name = dh._generate_name(c)
    assert name == "DummyInstrument.mod_b.mod_c.c"

    p = ManualParameter("test_parameter", label="Test parameter", unit="Hz")
    name = dh._generate_name(p)
    assert name == "test_parameter"


def test_load_snapshot_compressed_files(tmp_path):
    test_data = {"test": "data", "list": [1, 2, 3]}
    tuid = "20200430-170837-001-315f36"

    exp_path = tmp_path / "20200430" / f"{tuid}-TestExperiment"
    exp_path.mkdir(parents=True)

    for ext, opener in [(".bz2", bz2.open), (".gz", gzip.open), (".xz", lzma.open)]:
        snapshot_path = exp_path / f"snapshot.json{ext}"
        with opener(snapshot_path, "wt", encoding="utf-8") as f:
            json.dump(test_data, f)

        result = load_snapshot(tuid, datadir=tmp_path)
        assert result == test_data
        # snapshot_path.unlink()


def test_load_snapshot_compressed_with_array_conversion(tmp_path):
    test_data = {"array_data": [1, 2, 3, 4, 5]}
    tuid = "20200430-170837-001-315f36"
    exp_path = tmp_path / "20200430" / f"{tuid}-TestExperiment"
    exp_path.mkdir(parents=True)
    snapshot_path = exp_path / "snapshot.json.gz"

    with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
        json.dump(test_data, f)

    result = load_snapshot(tuid, datadir=tmp_path, list_to_ndarray=True)
    assert isinstance(result["array_data"], np.ndarray)
    np.testing.assert_array_equal(result["array_data"], np.array([1, 2, 3, 4, 5]))


def test_load_snapshot_invalid_json(tmp_path):
    tuid = "20200430-170837-001-315f36"

    exp_path = tmp_path / "20200430" / f"{tuid}-TestExperiment"
    exp_path.mkdir(parents=True)
    snapshot_path = exp_path / "snapshot.json"

    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write("invalid json content")

    with pytest.raises(ValueError, match="Invalid JSON file"):
        load_snapshot(tuid, datadir=tmp_path)


def test_load_snapshot_not_found(tmp_path):
    tuid = "20200430-170837-001-315f36"

    exp_path = tmp_path / "20200430" / f"{tuid}-TestExperiment"
    exp_path.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No snapshot file found"):
        load_snapshot(tuid, datadir=tmp_path)


def test_load_snapshot_container_not_found(tmp_path):
    tuid = "20200430-170837-001-315f36"
    with pytest.raises(
        FileNotFoundError,
        match=r"(No such file or directory|The system cannot find the path specified)",
    ):
        load_snapshot(tuid, datadir=tmp_path)
