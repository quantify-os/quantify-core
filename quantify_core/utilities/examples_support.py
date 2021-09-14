# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities used for creating examples for docs/tutorials/tests."""
from __future__ import annotations

import xarray as xr
from pathlib import Path
import quantify_core.data.handling as dh
import quantify_core.data.dataset as dd


def mk_dataset_attrs(**kwargs) -> dict:
    tuid = dh.gen_tuid()
    software_versions = [
        ("quantify_core", "921f1d4b6ebdbc7221f5fd55b17019283c6ee95e"),
        ("quantify_scheduler", "0.4.0"),
        ("qblox_instruments", "0.4.0"),
    ]
    attrs = dd.mk_default_dataset_attrs(tuid=tuid, software_versions=software_versions)
    attrs.update(kwargs)

    return attrs


def mk_exp_coord_attrs(**kwargs) -> dict:
    attrs = dd.mk_default_exp_coord_attrs(batched=False, uniformly_spaced=True)
    attrs.update(kwargs)
    return attrs


def mk_exp_var_attrs(**kwargs) -> dict:
    attrs = dd.mk_default_exp_var_attrs(grid=True, uniformly_spaced=True, batched=False)
    attrs.update(kwargs)
    return attrs


def dataset_round_trip(ds: xr.Dataset) -> xr.Dataset:
    tuid = ds.tuid
    dh.write_dataset(Path(dh.create_exp_folder(tuid)) / dh.DATASET_NAME, ds)
    return dh.load_dataset(tuid)


def par_to_attrs(par) -> dict:
    return dict(units=par.unit, long_name=par.label)
