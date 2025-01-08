"""Module which contains Peaks Method"""

from math import ceil

import numpy as np
from scipy.signal import find_peaks

from mpest.components_number.methods.abstract_estimator import AComponentsNumber
from mpest.types import Samples


class Peaks(AComponentsNumber):
    """Peaks Method with Emperical Density"""

    @property
    def name(self) -> str:
        return "Peaks"

    def estimate(self, samples: Samples) -> int:
        """
        Doanes fromula to determinate numbers of bins
        #  nbins = 1 + log2(n) + log2(1 + |skewness| / sg1)
        #  sg1 = √(6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0)))
        """

        n = samples.size
        sg1 = np.sqrt(6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0)))
        skew = np.mean(((samples - np.mean(samples)) / np.std(samples)) ** 3)

        nbins = ceil(1 + np.log2(n) + np.log2(1 + abs(skew) / sg1))

        #  Emperical Density
        hist = np.histogram(samples, bins=nbins, density=True)[0]
        hist = np.concatenate((np.zeros(1), hist, np.zeros(1)))

        #  Peaks counting
        peaks, _ = find_peaks(hist)
        peaks_count = len(peaks)
        return peaks_count
