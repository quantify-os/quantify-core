# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities used for creating examples for docs/tutorials/tests."""
from __future__ import annotations

from pathlib import Path
import xarray as xr
import quantify_core.data.handling as dh
import quantify_core.data.dataset_attrs as dd
import quantify_core.data.dataset_adapters as da


def mk_dataset_attrs(**kwargs) -> dict:
    tuid = dh.gen_tuid()
    software_versions = {
        "quantify_core": "921f1d4b6ebdbc7221f5fd55b17019283c6ee95e",
        "my_custom_repo": "<commit hash>",
        "my_intruments_drivers": "<version number>",
    }
    attrs = dd.QDatasetAttrs(tuid=tuid, software_versions=software_versions).to_dict()
    attrs.update(kwargs)

    return attrs


def mk_exp_coord_attrs(**kwargs) -> dict:
    attrs = dd.QExpCoordAttrs(batched=False, uniformly_spaced=True).to_dict()
    attrs.update(kwargs)
    return attrs


def mk_exp_var_attrs(**kwargs) -> dict:
    attrs = dd.QExpVarAttrs(grid=True, uniformly_spaced=True, batched=False).to_dict()
    attrs.update(kwargs)
    return attrs


def dataset_round_trip(ds: xr.Dataset) -> xr.Dataset:
    tuid = ds.tuid
    ds = da.AdapterH5NetCDF.adapt(ds)
    dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, ds)
    return da.AdapterH5NetCDF.recover(dh.load_dataset(tuid))


def par_to_attrs(par) -> dict:
    return dict(units=par.unit, long_name=par.label)
