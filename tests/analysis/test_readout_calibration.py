# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
from pathlib import Path

import numpy as np
import pytest

import quantify_core.data.handling as dh
from quantify_core.analysis.readout_calibration_analysis import (
    ReadoutCalibrationAnalysis,
)


@pytest.fixture(
    scope="session",
    autouse=True,
    params=[
        (
            "20230509-135911-755-9471f2",
            0.0090165,
            -0.020234 + np.pi,
        ),
        (
            "20230509-135927-693-0977e0",
            0.0081609,
            -0.13452 + np.pi,
        ),
        (
            "20230509-152441-841-faef49",
            0.0079211,
            -0.11857 + np.pi,
        ),
        # dataset with blobs phase np.pi/3
        ("20250522-124805-670-268fee", -0.0006539222727661782, 7.412065607082965),
        # dataset with flipped blobs phase np.pi/3
        ("20250522-124811-167-62ba0c", 0.0006895225374590243, 4.057147283249042),
        # dataset with blobs phase 2*np.pi/3
        ("20250522-125147-088-126753", -0.000683036945318069, 6.288826679632406),
        # dataset with flipped blobs phase 2*np.pi/3
        ("20250522-125152-962-78e4bc", 0.0006960853616184543, 3.1061587727080457),
    ],
)
def analysis_and_ref(tmp_test_data_dir, request):
    dh.set_datadir(tmp_test_data_dir)

    tuid, threshold, angle = request.param
    analysis = ReadoutCalibrationAnalysis(
        tuid=tuid, dataset=dh.load_dataset(tuid)
    ).run()

    return analysis, (threshold, angle)


def test_load_fit_results(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref
    for fit_name, fit_result in analysis.fit_results.items():
        loaded_fit_result = ReadoutCalibrationAnalysis.load_fit_result(
            tuid=analysis.tuid, fit_name=fit_name
        )
        assert loaded_fit_result.params == fit_result.params


def test_processed_dataset(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref

    container = Path(dh.locate_experiment_container(analysis.tuid))
    file_path = (
        container / "analysis_ReadoutCalibrationAnalysis" / "dataset_processed.hdf5"
    )
    _ = dh.load_dataset_from_path(file_path)


# pylint: disable=invalid-name
def test_quantities_of_interest(analysis_and_ref):
    analysis, (threshold, angle) = analysis_and_ref

    fitted_threshold = analysis.quantities_of_interest["acq_threshold"]
    fitted_angle = analysis.quantities_of_interest["acq_rotation_rad"]

    # .1 mV allowance
    assert abs(threshold - fitted_threshold.nominal_value) < 1e-4

    # 1e-4 radian allowance
    assert abs(angle % (2 * np.pi) - fitted_angle.nominal_value % (2 * np.pi)) < 1e-4


def test_print_error_without_crash(analysis_and_ref, tmp_test_data_dir, capsys):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref

    # re-run analysis with nan values
    bad_ds = analysis.dataset
    bad_ds.y1.data = np.asarray([float("nan")] * bad_ds.x0.data.size)

    _ = ReadoutCalibrationAnalysis(dataset=bad_ds).run()

    # Capture the printed output
    captured = capsys.readouterr()

    assert "Error during fit:" in captured.out
