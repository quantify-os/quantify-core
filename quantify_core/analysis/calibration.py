# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the main branch
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
    s21: np.ndarray, indices_state_0: tuple = (-2,), indices_state_1: tuple = (-1,)
) -> bool:
    r"""
    Attempts to determine if the provided complex S21 data has calibration points for
    the ground and first excited states of qubit.

    In this ideal scenario, if the datapoints indicated by the indices correspond to the
    calibration points, then these points will be located on the extremities of a
    "segment" on the IQ plane.

    Three pieces of information are used to infer the presence of calibration points:

    - The angle of the calibration points with respect to the average of the datapoints,
    - The distance between the calibration points, and
    - The average distance to the line defined be the calibration points.

    The detection is made robust by averaging 3 datapoints for each extremity of
    the "segment" described by the data on the IQ-plane.

    .. admonition:: Examples
        :class: dropdown

        In these examples this function is able to correctly predict the presence of
        the calibrations in both cases.

        .. include:: examples/analysis.calibration.has_calibration_points.rst.txt

    Parameters
    ----------
    s21
        Array of complex datapoints corresponding to the experiment on the IQ plane.
    indices_state_0
        Indices in the ``s21`` array that correspond to the ground state.
    indices_state_1
        Indices in the ``s21`` array that correspond to the first excited state.

    Returns
    -------
    :
        The inferred presence of calibration points.
    """
    indices_state_0 = np.asarray(indices_state_0)
    indices_state_1 = np.asarray(indices_state_1)

    def _arg_min_n(array: np.ndarray, num: int):
        return np.argpartition(array, num)[:num]

    def _arg_max_n(array: np.ndarray, num: int):
        return np.argpartition(array, -num)[-num:]

    not_cal: np.ndarray = np.ones(s21.shape, dtype=bool)
    not_cal[indices_state_0] = False
    not_cal[indices_state_1] = False

    # do not include the potential calibration points since that can significantly
    # affect if the most of the data is far away from one of the calibration points
    magnitude_no_cal: np.ndarray = np.abs(s21[not_cal])
    # Use the 3 points with maximum magnitude for resilience against noise and
    # outliers
    arg_max_no_cal: list = list(_arg_max_n(magnitude_no_cal, 3))
    # Move one side of the "segment" described by the data on the IQ-plane to the
    # center of the IQ plane. This is necessary for the arg_max and arg_min of the
    # magnitude to correspond to the "segment" extremities.
    s21_shifted: np.ndarray = s21 - s21[arg_max_no_cal].mean()
    maybe_cal_pnts_0: np.ndarray = s21_shifted[indices_state_0].mean()
    maybe_cal_pnts_1: np.ndarray = s21_shifted[indices_state_1].mean()

    magnitude: float = np.abs(s21_shifted)
    arg_max: list = list(_arg_max_n(magnitude, 3))
    arg_min: list = list(_arg_min_n(magnitude, 3))
    center: complex = s21_shifted[arg_min + arg_max].mean()  # center of the "segment"

    maybe_cal_pnts: np.ndarray = np.array((maybe_cal_pnts_0, maybe_cal_pnts_1))
    angles: np.ndarray = np.angle(maybe_cal_pnts - center, deg=True)
    angles_diff: float = angles.max() - angles.min()

    avg_max: complex = s21_shifted[arg_max].mean()
    avg_min: complex = s21_shifted[arg_min].mean()
    segment_len: float = np.abs(avg_max - avg_min)
    cal_dist: float = np.abs(maybe_cal_pnts_0 - maybe_cal_pnts_1)

    far_enough: bool = cal_dist > 0.5 * segment_len

    def _cross_prod_on_plane(num_a: complex, num_b: complex):
        return num_a.real * num_b.imag - num_b.real * num_a.imag

    def _dist_to_line(point_a: complex, point_b: complex, point_c: complex):
        vec_a = point_b - point_a
        vec_b = point_c - point_a
        return np.abs(_cross_prod_on_plane(vec_a, vec_b)) / np.abs(vec_a)

    # to exclude some false positives confirm that most of the data is withing a circle
    # with radius equal to half the distance between the calibration points
    dist_to_line: np.ndarray = _dist_to_line(
        maybe_cal_pnts_0, maybe_cal_pnts_1, s21_shifted
    )
    data_close_enough_to_line: bool = dist_to_line.mean() < cal_dist / 4

    good_angle: bool = angles_diff > 90
    has_cal_pnts: bool = far_enough and good_angle and data_close_enough_to_line

    return has_cal_pnts
