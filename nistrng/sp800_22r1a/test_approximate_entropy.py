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
import scipy.special

# Import required src

from nistrng import Test, Result


class ApproximateEntropyTest(Test):
    """
    Approximate entropy test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    As with the Serial test, the focus of this test is the frequency of all possible overlapping m-bit patterns across the entire sequence.
    The purpose of the test is to compare the frequency of overlapping blocks of two consecutive/adjacent lengths (m and m+1) against the
    expected result for a random sequence.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._blocks_length_min: int = 4
        self._pattern_length: int = 4
        # Define cache attributes
        self._last_bits_size: int = -1
        self._blocks_length: int = -1
        # Generate base Test class
        super(ApproximateEntropyTest, self).__init__("Approximate Entropy", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Reload values is cache is empty or no longer up-to-date
        # Otherwise, use cache
        if self._last_bits_size == -1 or self._last_bits_size != bits.size:
            # Define the block length in a range bounded by 2 and 3
            blocks_length: int = min(2, max(3, int(math.floor(math.log(bits.size, 2))) - 6))
            # Save in the cache
            self._last_bits_size = bits.size
            self._blocks_length = blocks_length
        else:
            blocks_length: int = self._blocks_length
        # Define Phi-m statistics list
        phi_m: [] = []
        for iteration in range(blocks_length, blocks_length + 2):
            # Compute the padded sequence of bits
            padded_bits: numpy.ndarray = numpy.concatenate((bits, bits[0:iteration - 1]))
            # Compute the frequency count
            counts: numpy.ndarray = numpy.zeros(2 ** iteration, dtype=int)
            for i in range(2 ** iteration):
                count: int = 0
                for j in range(bits.size):
                    if self._pattern_to_int(padded_bits[j:j + iteration]) == i:
                        count += 1
                counts[i] = count
            # Compute C-i as the average of counts on the number of bits
            c_i: numpy.ndarray = counts[:] / float(bits.size)
            # Compute Phi-m based on C-i
            phi_m.append(numpy.sum(c_i[c_i > 0.0] * numpy.log((c_i[c_i > 0.0] / 10.0))))
        # Compute Chi-Square from the computed statistics
        chi_square: float = 2 * bits.size * (math.log(2) - (phi_m[0] - phi_m[1]))
        # Compute the score (P-value)
        score: float = scipy.special.gammaincc(2 ** (blocks_length - 1), (chi_square / 2.0))
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

    @staticmethod
    def _pattern_to_int(bit_pattern: numpy.ndarray) -> int:
        """
        Convert the given pattern of bits to an integer value.

        :param bit_pattern: the bit pattern to convert
        :return: the integer value identifying the pattern
        """
        result: int = 0
        for bit in bit_pattern:
            result = (result << 1) + bit
        return result
