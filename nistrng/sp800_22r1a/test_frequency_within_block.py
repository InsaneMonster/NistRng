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


class FrequencyWithinBlockTest(Test):
    """
    Frequency within block test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of the test is the proportion of ones within M-bit blocks. The purpose of this test is to determine whether the frequency of
    ones in an M-bit block is approximately M/2, as would be expected under an assumption of randomness.
    For block size M=1, this test degenerates to the Frequency (Monobit) test.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._sequence_size_min: int = 100
        self._default_block_size: int = 20
        self._blocks_number_max: int = 100
        # Define cache attributes
        self._last_bits_size: int = -1
        self._block_size: int = -1
        self._blocks_number: int = -1
        # Generate base Test class
        super(FrequencyWithinBlockTest, self).__init__("Frequency Within Block", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Reload values is cache is empty or no longer up-to-date
        # Otherwise, use cache
        if self._last_bits_size == -1 or self._last_bits_size != bits.size:
            # Get the number of blocks (N) with the default minimum block size (M)
            block_size: int = self._default_block_size
            blocks_number: int = int(bits.size // block_size)
            # Get the block size (M) if the number of blocks (N) exceed the allowed max
            if blocks_number >= self._blocks_number_max:
                blocks_number = self._blocks_number_max - 1
                block_size = int(bits.size // blocks_number)
            # Save in the cache
            self._last_bits_size = bits.size
            self._block_size = block_size
            self._blocks_number = blocks_number
        else:
            block_size: int = self._block_size
            blocks_number: int = self._blocks_number
        # Initialize a list of fractions
        block_fractions: numpy.ndarray = numpy.zeros(blocks_number, dtype=float)
        for i in range(blocks_number):
            # Get the bits in the current block
            block: numpy.ndarray = bits[i * block_size:((i + 1) * block_size)]
            # Compute ones and save the fraction in the array
            block_fractions[i] = numpy.count_nonzero(block) / block_size
        # Compute Chi-square
        chi_square: float = numpy.sum(4.0 * block_size * ((block_fractions[:] - 0.5) ** 2))
        # Compute score (P-value) applying the lower incomplete gamma function
        score: float = scipy.special.gammaincc((blocks_number / 2.0), chi_square / 2.0)
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
