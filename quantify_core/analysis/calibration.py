# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Module containing analysis utilities for calibration procedures.

In particular, manipulation of data and calibration points for qubit readout
calibration.
"""

from __future__ import annotations

import numpy as np


def rotate_to_calibrated_axis(
    data: np.ndarray, ref_val_0: complex, ref_val_1: complex
) -> np.ndarray:
    """
    Rotates, normalizes and offsets complex valued data based on calibration points.

    Parameters
    ----------
    data
        An array of complex valued data points.
    ref_val_0
        The reference value corresponding to the 0 state.
    ref_val_1
        The reference value corresponding to the 1 state.

    Returns
    -------
    :
        Calibrated array of complex data points.
    """
    rotation_anle = np.angle(ref_val_1 - ref_val_0)
    norm = np.abs(ref_val_1 - ref_val_0)
    offset = ref_val_0 * np.exp(-1j * rotation_anle) / norm

    corrected_data = data * np.exp(-1j * rotation_anle) / norm - offset

    return corrected_data


# pylint: disable=too-many-locals
def has_calibration_points(
    s21: np.ndarray, indices_state_0: list = [-2], indices_state_1: list = [-1]
) -> bool:
    r"""
    Attempts to determine if the provided complex s21 data has calibration points for
    the ground and first excited states.

    In this ideal scenario, if the last 2
    datapoints do correspond to the calibration points, then these points will be
    located on the extremities of a "segment" on the IQ plane. The angle with
    respect to the average of the datapoints and the distance between the
    calibration points are used to infer the presence of calibration points.

    The detection is made robust by averaging 3 datapoints for each extremity of
    the "segment" described by the data on the IQ-plane.

    Parameters
    ----------
    s21
        Array of complex datapoints corresponding to the experiment on the IQ plane.
    """

    def _arg_min_n(array: np.ndarray, num: int):
        return np.argpartition(array, num)[:num]

    def _arg_max_n(array: np.ndarray, num: int):
        return np.argpartition(array, -num)[-num:]

    magnitude = np.abs(s21[:-2])  # last two points would be the calibration
    # Use the 3 points with maximum magnitude for resilience against noise and
    # outliers
    arg_max = list(_arg_max_n(magnitude, 3))
    # Move one side of the "segment" described by the data on the IQ-plane to the
    # center of the IQ plane. This is necessary for the arg_max and arg_min of the
    # magnitude to correspond to the "segment" extremities.
    s21_array: np.ndarray = s21 - s21[arg_max].mean()
    maybe_cal_pnts: np.ndarray = s21_array[-2:]
    s21_array: np.ndarray = s21_array[:-2]

    magnitude: float = np.abs(s21_array)
    arg_max: int = list(_arg_max_n(magnitude, 3))
    arg_min: int = list(_arg_min_n(magnitude, 3))
    center: complex = s21_array[arg_min + arg_max].mean()  # center of the "segment"

    angles: float = np.angle(maybe_cal_pnts - center, deg=True)
    angles_diff: float = angles.max() - angles.min()

    avg_max: complex = s21_array[arg_max].mean()
    avg_min: complex = s21_array[arg_min].mean()
    segment_len: float = np.abs(avg_max - avg_min)
    far_enough = np.abs(maybe_cal_pnts[0] - maybe_cal_pnts[1]) > 0.5 * segment_len

    has_cal_pnts = angles_diff > 90 and far_enough

    return has_cal_pnts
