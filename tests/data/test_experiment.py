# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


import pytest

from quantify_core.data import handling as dh
from quantify_core.data.experiment import QuantifyExperiment


TUID_1D_1PLOT = "20200430-170837-001-315f36"


def test_quantify_experiment(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = TUID_1D_1PLOT
    folder = dh.create_exp_folder(tuid)
    _ = folder  # Unused directory name folder
    experiment = QuantifyExperiment(tuid)

    assert experiment.tuid == tuid

    assert tuid in experiment.__repr__()
    snap = {"snap": 1, "snap2": "str"}
    experiment.save_snapshot(snap)
    assert experiment.load_snapshot() == snap


def test_quantify_experiment_load_and_save_text(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = TUID_1D_1PLOT
    dh.create_exp_folder(tuid)
    experiment = QuantifyExperiment(tuid)

    text = "text to be saved"
    rel_path = "directory/file.txt"

    experiment.save_text(text, rel_path)

    assert experiment.load_text(rel_path) == text


def test_quantify_experiment_load_and_save_metadata(tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    tuid = TUID_1D_1PLOT
    dh.create_exp_folder(tuid)
    experiment = QuantifyExperiment(tuid)
    with pytest.raises(FileNotFoundError):
        _ = experiment.load_metadata()

    dictionary = {"key": "entry to be saved"}
    experiment.save_metadata(dictionary)
    assert experiment.load_metadata() == dictionary
