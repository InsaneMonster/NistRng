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

# Import required src

from nistrng import Test, Result


class RandomExcursionVariantTest(Test):
    """
    Random excursion variant test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the total number of times that a particular state is visited (i.e., occurs) in a cumulative sum random walk.
    The purpose of this test is to detect deviations from the expected number of visits to various states in the random walk.
    This test is actually a series of eighteen tests (and conclusions), one test and conclusion for each of the
    states: -9, -8, ..., -1 and +1, +2, ..., +9.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Generate base Test class
        super(RandomExcursionVariantTest, self).__init__("Random Excursion Variant", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Copy the bits to a new array
        bits_copy: numpy.ndarray = bits.copy()
        # Convert all the zeros in the array to -1
        bits_copy[bits_copy == 0] = -1
        # Generate the padded cumulative sum of the array of -1, 1
        sum_prime: numpy.ndarray = numpy.concatenate((numpy.array([0]), numpy.cumsum(bits_copy), numpy.array([0]))).astype(int)
        # Count the number of cycles in S' (sum_prime)
        cycles_size: int = numpy.count_nonzero(sum_prime[1:] == 0)
        # Generate the counts of offsets
        unique, counts = numpy.unique(sum_prime[abs(sum_prime) < 10], return_counts=True)
        # Compute the scores (P-values)
        scores: [] = []
        for key, value in zip(unique, counts):
            # Compute the P-value for this value (if not zero)
            if key != 0:
                scores.append(abs(value - cycles_size) / math.sqrt(2.0 * cycles_size * ((4.0 * abs(key)) - 2.0)))
        # Return result
        if all(score >= self.significance_value for score in scores):
            return Result(self.name, True, numpy.array(scores))
        return Result(self.name, False, numpy.array(scores))

    def is_eligible(self,
                    bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # This test is always eligible for any sequence
        return True
