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


class LinearComplexityTest(Test):
    """
    Linear complexity test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the length of a linear feedback shift register (LFSR). The purpose of this test is to determine whether
    or not the sequence is complex enough to be considered random. Random sequences are characterized by longer LFSRs.
    An LFSR that is too short implies non-randomness.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._sequence_size_min: int = 1000000
        self._pattern_length: int = 512
        self._freedom_degrees: int = 6
        self._probabilities: numpy.ndarray = numpy.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
        # Compute mean
        self._mu: float = (self._pattern_length / 2.0) + (((-1) ** (self._pattern_length + 1)) + 9.0) / 36.0 - ((self._pattern_length / 3.0) + (2.0 / 9.0)) / (2 ** self._pattern_length)
        # Define cache attributes
        self._last_bits_size: int = -1
        self._blocks_number: int = -1
        # Generate base Test class
        super(LinearComplexityTest, self).__init__("Linear Complexity", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Reload values is cache is empty or no longer up-to-date
        # Otherwise, use cache
        if self._last_bits_size == -1 or self._last_bits_size != bits.size:
            # Set the block size
            blocks_number: int = int(bits.size // self._pattern_length)
            # Save in the cache
            self._last_bits_size = bits.size
            self._blocks_number = blocks_number
        else:
            blocks_number: int = self._blocks_number
        # Compute the linear complexity of the blocks
        blocks_linear_complexity: numpy.ndarray = numpy.zeros(blocks_number, dtype=int)
        for i in range(blocks_number):
            blocks_linear_complexity[i] = self._berlekamp_massey(bits[(i * self._pattern_length):((i + 1) * self._pattern_length)])
        # Count the distribution over tickets
        tickets: numpy.ndarray = ((-1.0) ** self._pattern_length) * (blocks_linear_complexity[:] - self._mu) + (2.0 / 9.0)
        # Compute frequencies depending on tickets
        frequencies: numpy.ndarray = numpy.zeros(self._freedom_degrees + 1, dtype=int)
        for ticket in tickets:
            frequencies[min(self._freedom_degrees, int(max(-2.5, ticket) + 2.5))] += 1
        # Compute Chi-square using pre-defined probabilities
        chi_square: float = float(numpy.sum(((frequencies[:] - (blocks_number * self._probabilities[:])) ** 2.0) / (blocks_number * self._probabilities[:])))
        # Compute the score (P-value)
        score: float = scipy.special.gammaincc((self._freedom_degrees / 2.0), (chi_square / 2.0))
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
    def _berlekamp_massey(sequence: numpy.ndarray) -> int:
        """
        Compute the linear complexity of a sequence of bits by the means of the Berlekamp Massey algorithm.

        :param sequence: the sequence of bits to compute the linear complexity for
        :return: the int value of the linear complexity
        """
        # Initialize b and c to all zeroes with first element one
        b: numpy.ndarray = numpy.zeros(sequence.size, dtype=int)
        c: numpy.ndarray = numpy.zeros(sequence.size, dtype=int)
        b[0] = 1
        c[0] = 1
        # Initialize the generator length
        generator_length: int = 0
        # Initialize variables
        m: int = -1
        n: int = 0
        while n < sequence.size:
            # Compute discrepancy
            discrepancy = sequence[n]
            for j in range(1, generator_length + 1):
                discrepancy: int = discrepancy ^ (c[j] & sequence[n - j])
            # If discrepancy is not zero, adjust polynomial
            if discrepancy != 0:
                t = c[:]
                for j in range(0, sequence.size - n + m):
                    c[n - m + j] = c[n - m + j] ^ b[j]
                if generator_length <= n / 2:
                    generator_length = n + 1 - generator_length
                    m = n
                    b = t
            n = n + 1
        # Return the length of generator
        return generator_length
