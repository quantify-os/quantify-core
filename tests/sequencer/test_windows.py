from quantify.sequencer.windows import Hanning
import numpy as np
import matplotlib.pyplot as plt


def test_hanning():
    #  would be cool if this thing scaled adjusted the scale or warned if resolution is too low
    wf = np.ones(8)
    hann_equal = Hanning(1)
    hann_half = Hanning(2)
    hann_quarted = Hanning(4)

    wf_equal_filter = hann_equal.filter(wf)
    wf_half_filter = hann_half.filter(wf)
    wf_4x_filter = hann_quarted.filter(wf)

    fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    ax_orig.plot(wf_equal_filter)
    ax_orig.set_title('Equal')
    ax_orig.margins(0, 0.1)
    ax_win.plot(wf_half_filter)
    ax_win.set_title('Half')
    ax_win.margins(0, 0.1)
    ax_filt.plot(wf_4x_filter)
    ax_filt.set_title('4x')
    ax_filt.margins(0, 0.1)
    fig.tight_layout()
    fig.show()
    plt.show()
