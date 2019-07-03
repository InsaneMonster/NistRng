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


class RunsTest(Test):
    """
    Runs test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the total number of runs in the sequence, where a run is an uninterrupted sequence of identical bits.
    A run of length k consists of exactly k identical bits and is bounded before and after with a bit of the opposite value.
    The purpose of the runs test is to determine whether the number of runs of ones and zeros of various lengths is as expected
    for a random sequence. In particular, this test determines whether the oscillation between such zeros and ones is too fast or too slow.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Generate base Test class
        super(RunsTest, self).__init__("Runs", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        proportion: float = numpy.count_nonzero(bits) / bits.size
        # Count the observed runs (list of adjacent equal bits)
        observed_runs: float = 1.0
        for i in range(bits.size - 1):
            if bits[i] != bits[i + 1]:
                observed_runs += 1.0
        # Compute score (P-value)
        score: float = math.erfc(abs(observed_runs - (2.0 * bits.size * proportion * (1.0 - proportion))) / (2.0 * math.sqrt(2.0 * bits.size) * proportion * (1 - proportion)))
        # Return result
        if score >= self.significance_value:
            return Result(self.name, True, numpy.array(score))
        return Result(self.name, False, numpy.array(score))

    def is_eligible(self,
                    bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Check for eligibility
        proportion: float = numpy.count_nonzero(bits) / bits.size
        tau: float = 2.0 / math.sqrt(bits.size)
        if abs(proportion - 0.5) > tau:
            return False
        return True
