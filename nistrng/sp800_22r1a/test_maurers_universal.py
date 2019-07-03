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


class MaurersUniversalTest(Test):
    """
    Maurers universal test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the number of bits between matching patterns (a measure that is related to the length of a compressed sequence).
    The purpose of the test is to detect whether or not the sequence can be significantly compressed without loss of information.
    A significantly compressible sequence is considered to be non-random.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        # Note: tables from https://static.aminer.org/pdf/PDF/000/120/333/a_universal_statistical_test_for_random_bit_generators.pdf
        self._sequence_size_min: int = 387840
        self._default_pattern_size: int = 6
        self._freedom_degrees: int = 5
        self._substring_bits_length: int = 1062
        self._thresholds: [] = [904960, 2068480, 4654080, 10342400, 22753280, 49643520, 107560960, 231669760, 496435200, 1059061760]
        self._expected_value_table: [] = [0, 0.73264948, 1.5374383, 2.40160681, 3.31122472, 4.25342659, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243, 10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
        self._variance_table: [] = [0, 0.690, 1.338, 1.901, 2.358, 2.705, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
        # Define cache attributes
        self._last_bits_size: int = -1
        self._pattern_length: int = -1
        self._blocks_number: int = -1
        self._q_blocks: int = -1
        self._k_blocks: int = -1
        # Generate base Test class
        super(MaurersUniversalTest, self).__init__("Maurers Universal", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Reload values is cache is empty or no longer up-to-date
        # Otherwise, use cache
        if self._last_bits_size == -1 or self._last_bits_size != bits.size:
            # Compute the pattern size
            pattern_length: int = self._default_pattern_size
            for threshold in self._thresholds:
                if bits.size >= threshold:
                    pattern_length += 1
            # Split the data into Q and K blocks
            blocks_number: int = int(bits.size // pattern_length)
            q_blocks: int = 10 * (2 ** pattern_length)
            k_blocks: int = blocks_number - q_blocks
            # Save in the cache
            self._last_bits_size = bits.size
            self._pattern_length = pattern_length
            self._blocks_number = blocks_number
            self._q_blocks = q_blocks
            self._k_blocks = k_blocks
        else:
            pattern_length: int = self._pattern_length
            blocks_number: int = self._blocks_number
            q_blocks: int = self._q_blocks
            k_blocks: int = self._k_blocks
        # Construct table of symbols all zeroed out at the beginning
        table: numpy.ndarray = numpy.zeros(2 ** pattern_length, dtype=int)
        # Mark final position in Q-blocks
        for i in range(q_blocks):
            # Get the pattern in the Q-block
            pattern: numpy.ndarray = bits[i * pattern_length:(i + 1) * pattern_length]
            # +1 to number indexes 1... (2 ** L) + 1 instead of 0... 2 ** L
            table[self._pattern_to_int(pattern)] = i + 1
        # Mark final position in K-blocks and compute the sum
        computed_sum: float = 0.0
        for i in range(q_blocks, blocks_number):
            # Get the pattern in the K-block
            pattern: numpy.ndarray = bits[i * pattern_length:(i + 1) * pattern_length]
            # Compute difference with respect to the current value in the table
            difference: int = i + 1 - table[self._pattern_to_int(pattern)]
            # Update the current value in the table
            table[self._pattern_to_int(pattern)] = i + 1
            # Update the computed sum
            computed_sum += math.log(difference, 2)
        # Compute the test statistic
        fn: float = computed_sum / k_blocks
        # Compute magnitude
        magnitude: float = abs((fn - self._expected_value_table[pattern_length]) / ((math.sqrt(self._variance_table[pattern_length])) * math.sqrt(2)))
        # Compute the score (P-value)
        score: float = math.erfc(magnitude)
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
        if bits.size < self._sequence_size_min:
            return False
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
