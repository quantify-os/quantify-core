from quantify.sequencer.windows import Hann
import numpy as np
import pytest


@pytest.mark.skip('todo')
def test_hanning():
    wf = np.ones(20)
    hann_equal = Hann(1)
    hann_half = Hann(2)
    hann_quarted = Hann(4)

    wf_equal_filter = hann_equal.filter(wf)
    wf_half_filter = hann_half.filter(wf)
    wf_4x_filter = hann_quarted.filter(wf)
