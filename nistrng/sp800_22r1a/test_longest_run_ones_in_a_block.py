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
import scipy.special

# Import required src

from nistrng import Test, Result


class LongestRunOnesInABlockTest(Test):
    """
    Longest run ones in a block test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of the test is the longest run of ones within M-bit blocks. The purpose of this test is to determine whether
    the length of the longest run of ones within the tested sequence is consistent with the length of the longest run of
    ones that would be expected in a random sequence. Note that an irregularity in the expected length of the longest run
    of ones implies that there is also an irregularity in the expected length of the longest run of zeroes.
    Therefore, only a test for ones is necessary.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._sequence_size_min: int = 128
        # Define cache attributes
        self._last_bits_size: int = -1
        self._block_size: int = -1
        self._blocks_number: int = -1
        self._k: int = -1
        # Generate base Test class
        super(LongestRunOnesInABlockTest, self).__init__("Longest Run Ones In A Block", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Reload values is cache is empty or no longer up-to-date
        # Otherwise, use cache
        if self._last_bits_size == -1 or self._last_bits_size != bits.size:
            # Set the block size depending on the input sequence length
            block_size: int = 10000
            if bits.size < 6272:
                block_size: int = 8
            elif bits.size < 750000:
                block_size: int = 128
            # Set the block number and K depending on the block size
            k: int = 6
            blocks_number: int = 75
            if block_size == 8:
                k: int = 3
                blocks_number: int = 16
            elif block_size == 128:
                k: int = 5
                blocks_number: int = 49
            # Save in the cache
            self._last_bits_size = bits.size
            self._block_size = block_size
            self._blocks_number = blocks_number
            self._k = k
        else:
            block_size: int = self._block_size
            blocks_number: int = self._blocks_number
            k: int = self._k
        # Define the array of frequencies
        frequencies: numpy.ndarray = numpy.zeros(7, dtype=int)
        # Find longest run length in each block
        for i in range(blocks_number):
            block: numpy.ndarray = bits[i * block_size:((i + 1) * block_size)]
            run_length: int = 0
            longest_run_length: int = 0
            # Count the length of each adjacent bits group (runs) in the current block and update the max length of them
            for j in range(block_size):
                if block[j] == 1:
                    run_length += 1
                    if run_length > longest_run_length:
                        longest_run_length = run_length
                else:
                    run_length = 0
            # Update the list of frequencies
            if block_size == 8:
                frequencies[min(3, max(0, longest_run_length - 1))] += 1
            elif block_size == 128:
                frequencies[min(5, max(0, longest_run_length - 4))] += 1
            else:
                frequencies[min(6, max(0, longest_run_length - 10))] += 1
        # Compute Chi-square
        chi_square: float = 0.0
        for i in range(k + 1):
            chi_square += ((frequencies[i] - blocks_number * self._probabilities(block_size, i)) ** 2) / (blocks_number * self._probabilities(block_size, i))
        # Compute score (P-value)
        score: float = scipy.special.gammaincc(k / 2.0, chi_square / 2.0)
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
    def _probabilities(size_of_block: int, index: int) -> float:
        """
        Returns a probability at the given index in the array or probabilities defined for the block of the given size.

        :param size_of_block: can be 8, 128, 512, 1000 and in any other case will fallback on 10000
        :param index: the index of the probability
        :return: the probability at the given index
        """
        if size_of_block == 8:
            return [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124][index]
        elif size_of_block == 128:
            return [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124][index]
        elif size_of_block == 512:
            return [0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124][index]
        elif size_of_block == 1000:
            return [0.1307, 0.2437, 0.2452, 0.1714, 0.1002, 0.1088][index]
        else:
            return [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727][index]
