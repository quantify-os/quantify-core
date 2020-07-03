from scipy import signal


class Hanning:

    def __init__(self, scale):
        """
        Args:
            scale (int) : size of the window compared to the signal (divided)
        """
        self.scale = scale

    def filter(self, pulse):
        window = signal.windows.hann(int(len(pulse) / self.scale))
        return signal.convolve(pulse, window, mode='same') / sum(window)
