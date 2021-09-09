import numpy as np
from quantify_core.analysis.single_qubit_timedomain import (
    rotate_to_calibrated_axis,
)


def test_rotate_to_calibrated_axis():

    ref_val_0 = 0.24 + 324 * 1j
    ref_val_1 = 0.89 + 0.324 * 1j
    data = np.array([ref_val_0, ref_val_1])

    corrected_data = rotate_to_calibrated_axis(
        data, ref_val_0=ref_val_0, ref_val_1=ref_val_1
    )

    assert corrected_data == np.array([0.0, 1.0])
