# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities for managing experiment data."""

from typing import Dict, Any, Optional
from pathlib import Path

import xarray as xr

from quantify_core.data.types import TUID
from quantify_core.utilities.general import save_json, load_json
from quantify_core.data.handling import (
    locate_experiment_container,
    load_dataset,
    DATASET_NAME,
    write_dataset,
)
from quantify_core.data.handling import snapshot as create_snapshot


class QuantifyExperiment:
    """
    Class which represents all data related to an experiment. This allows the user to
    run experiments and store data without the
    `quantify_core.measurement.control.MeasurementControl`. The class serves as an
    initial interface for other data storage backends.
    """

    def __init__(self, tuid: Optional[str], dataset=None):
        """
        Creates an instance of the QuantifyExperiment.

        Parameters
        ----------
        tuid
            TUID to use
        dataset
            If the TUID is None, use the TUID from this dataset

        """
        if tuid is None:
            self.tuid = dataset.tuid
            self.dataset = dataset
        else:
            self.tuid = tuid
            self.dataset = None
        self.tuid = TUID(tuid)

    def __repr__(self) -> str:
        classname = ".".join([self.__module__, self.__class__.__qualname__])
        idx = "%x" % id(self)
        return f"<{classname} at %x{idx}>: TUID {self.tuid}"

    @property
    def experiment_directory(self) -> Path:
        """
        Returns a path to the experiment directory containing the TUID set within
        the class.

        Returns
        -------
        :

        """
        experiment_directory = locate_experiment_container(tuid=self.tuid)
        return Path(experiment_directory)

    def load_dataset(self) -> xr.Dataset:
        """
        Loads the quantify dataset associated with the TUID set within
        the class.

        Returns
        -------
        :

        """
        self.dataset = load_dataset(self.tuid)
        return self.dataset

    def write_dataset(self, dataset: xr.Dataset):
        """
        Writes the quantify dataset to the directory specified by
        `~.experiment_directory`.

        Parameters
        ----------
        dataset
            The dataset to be written to the directory

        """
        path = self.experiment_directory / DATASET_NAME
        write_dataset(path, dataset)

    def save_snapshot(self, snapshot: Optional[Dict[str, Any]] = None):
        """
        Writes the snapshot to disk as specified by
        `~.experiment_directory`.

        Parameters
        ----------
        snapshot
            The snapshot to be written to the directory

        """
        if snapshot is None:
            snapshot = create_snapshot()
        save_json(
            directory=self.experiment_directory, filename="snapshot.json", data=snapshot
        )

    def load_snapshot(self) -> Dict[str, Any]:
        """
        Loads the snapshot from the directory specified by
        `~.experiment_directory`.

        Returns
        -------
        :
            The loaded snapshot from disk

        """
        return load_json(full_path=self.experiment_directory / "snapshot.json")
