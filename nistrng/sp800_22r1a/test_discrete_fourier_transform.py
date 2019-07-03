#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import numpy
import math
import sys

# Import required src

from nistrng import Test, Result


class DiscreteFourierTransformTest(Test):
    """
    Discrete Fourier transform (spectral) test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the peak heights in the Discrete Fourier Transform of the sequence.
    The purpose of this test is to detect periodic features (i.e., repetitive patterns that are near each other) in the
    tested sequence that would indicate a deviation from the assumption of randomness.
    The intention is to detect whether the number of peaks exceeding the 95% threshold is significantly different than 5%.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Generate base Test class
        super(DiscreteFourierTransformTest, self).__init__("Discrete Fourier Transform", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Make sure the sequence is even in length
        bits_copy: numpy.ndarray = bits.copy()
        if (bits_copy.size % 2) == 1:
            bits_copy = bits_copy[:-1]
        # Convert all the zeros in the array to -1
        bits_copy[bits_copy == 0] = -1
        # Compute DFT
        discrete_fourier_transform = numpy.fft.fft(bits_copy)
        # Compute magnitudes of first half of sequence depending on the system type
        if sys.version_info > (3, 0):
            magnitudes = abs(discrete_fourier_transform)[:bits_copy.size // 2]
        else:
            magnitudes = abs(discrete_fourier_transform)[:bits_copy.size / 2]
        # Compute upper threshold
        threshold: float = math.sqrt(math.log(1.0 / 0.05) * bits_copy.size)
        # Compute the expected number of peaks (N0)
        expected_peaks: float = 0.95 * bits_copy.size / 2.0
        # Count the peaks above the upper threshold (N1)
        counted_peaks: float = float(len(magnitudes[magnitudes < threshold]))
        # Compute the score (P-value) using the normalized difference
        normalized_difference: float = (counted_peaks - expected_peaks) / math.sqrt((bits_copy.size * 0.95 * 0.05) / 4)
        score: float = math.erfc(abs(normalized_difference) / math.sqrt(2))
        # Return result
        if score >= self.significance_value:
            return Result(self.name, True, numpy.array(score))
        return Result(self.name, False, numpy.array(score))

    def is_eligible(self,
                    bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # This test is always eligible for any sequence
        return True
