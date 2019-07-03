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


class MonobitTest(Test):
    """
    Monobit test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of the test is the proportion of zeroes and ones for the entire sequence. The purpose of this test is to determine
    whether the number of ones and zeros in a sequence are approximately the same as would be expected for a truly random sequence.
    The test assesses the closeness of the fraction of ones to 1/2, that is, the number of ones and zeroes in a sequence
    should be about the same.

    The significance value of the test is 0.01.
    """
    def __init__(self):
        # Generate base Test class
        super(MonobitTest, self).__init__("Monobit", 0.01)

    def _execute(self,
                 bits: numpy.ndarray):
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Compute ones and zeroes
        ones: int = numpy.count_nonzero(bits)
        zeroes: int = bits.size - ones
        # Compute difference
        difference: int = abs(ones - zeroes)
        # Compute score
        score: float = math.erfc(float(difference) / (math.sqrt(float(bits.size)) * math.sqrt(2.0)))
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



