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


class SerialTest(Test):
    """
    Serial test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the frequency of all possible overlapping m-bit patterns across the entire sequence.
    The purpose of this test is to determine whether the number of occurrences of the 2mm-bit overlapping patterns is
    approximately the same as would be expected for a random sequence. Random sequences have uniformity; that is, every m-bit
    pattern has the same chance of appearing as every other m-bit pattern.
    Note that for m = 1, the Serial test is equivalent to the Monobit test.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._blocks_length_min: int = 4
        self._pattern_length: int = 4
        # Generate base Test class
        super(SerialTest, self).__init__("Serial", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Pad the sequence
        padded_bits: numpy.ndarray = numpy.concatenate((bits, bits[0:self._pattern_length - 1]))
        # Compute Psi-Squared statistics
        psi_sq_m_0: float = self._psi_sq_mv1(self._pattern_length, bits.size, padded_bits)
        psi_sq_m_1: float = self._psi_sq_mv1(self._pattern_length - 1, bits.size, padded_bits)
        psi_sq_m_2: float = self._psi_sq_mv1(self._pattern_length - 2, bits.size, padded_bits)
        delta_1: float = psi_sq_m_0 - psi_sq_m_1
        delta_2: float = psi_sq_m_0 - (2 * psi_sq_m_1) + psi_sq_m_2
        # Compute the scores (P-values)
        score_1: float = scipy.special.gammaincc(2 ** (self._pattern_length - 2), delta_1 / 2.0)
        score_2: float = scipy.special.gammaincc(2 ** (self._pattern_length - 3), delta_2 / 2.0)
        # Return result
        if score_1 >= self.significance_value and score_2 >= self.significance_value:
            return Result(self.name, True, numpy.array([score_1, score_2]))
        return Result(self.name, False, numpy.array([score_1, score_2]))

    def is_eligible(self,
                    bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Check for eligibility
        if int(math.floor(math.log(bits.size, 2))) - 2 < self._blocks_length_min:
            return False
        return True

    @staticmethod
    def _count_pattern(pattern: numpy.ndarray, padded_sequence: numpy.ndarray, sequence_size: int) -> int:
        """
        Count the matches in the padded sequence of the given size with the given pattern.

        :param pattern: the pattern to match against
        :param padded_sequence: the sequence of bits once padded
        :param sequence_size: the size of the original sequence of bits
        :return: the integer value of the count
        """
        count: int = 0
        for i in range(sequence_size):
            match: bool = True
            for j in range(len(pattern)):
                if pattern[j] != padded_sequence[i + j]:
                    match = False
            if match:
                count += 1
        return count

    @staticmethod
    def _psi_sq_mv1(block_size: int, sequence_size: int, padded_sequence: numpy.ndarray) -> float:
        """
        Compute the Psi-Squared statistics from the NIST paper.

        :param block_size: the size of the block
        :param sequence_size: the size of the sequence of bits
        :param padded_sequence: the original sequence once padded
        :return: the float value of Psi-Squared statistics
        """
        # Count the patterns
        counts: numpy.ndarray = numpy.zeros(2 ** block_size, dtype=int)
        for i in range(2 ** block_size):
            pattern: numpy.ndarray = (i >> numpy.arange(block_size, dtype=int)) & 1
            counts[i] = SerialTest._count_pattern(pattern, padded_sequence, sequence_size)
        # Compute Psi-Squared statistics and return it
        psi_sq_m: float = numpy.sum(counts[:] ** 2)
        psi_sq_m *= (2 ** block_size) / sequence_size
        psi_sq_m -= sequence_size
        return psi_sq_m
