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
import random
import scipy.special

# Import required src

from nistrng import Test, Result


class NonOverlappingTemplateMatchingTest(Test):
    """
    Non overlapping template matching test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the number of occurrences of  pre-specified target strings.
    The purpose of this test is to detect generators that produce too many occurrences of a given non-periodic (aperiodic)
    pattern. For this test an m-bit window is used to search for a specific m-bit pattern. If the pattern is not found,
    the window slides one bit position. If the pattern is found, the window is reset to the bit after the found pattern,
    and the search resumes.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._blocks_number: int = 8
        self._templates: [] = [
                                [[0, 1], [1, 0]],
                                [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]],
                                [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                                [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]],
                                [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
                                [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1], [0, 0, 1, 1, 0, 1, 1], [0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]],
                                [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0, 1, 1],  [0, 0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1],  [0, 0, 1, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 1, 1], [0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 0], [1, 1, 1, 0, 0, 1, 1, 0], [1, 1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]]
                                ]
        # Define cache attributes
        self._last_bits_size: int = -1
        self._substring_bits_length: int = -1
        # Generate base Test class
        super(NonOverlappingTemplateMatchingTest, self).__init__("Non Overlapping Template Matching", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Choose the template B at random
        b_template: numpy.ndarray = numpy.array(random.choice(random.choice(self._templates)))
        # Reload values is cache is empty or no longer up-to-date
        # Otherwise, use cache
        if self._last_bits_size == -1 or self._last_bits_size != bits.size:
            # Split into N blocks of M bits
            substring_bits_length: int = int(bits.size // self._blocks_number)
            # Save in the cache
            self._last_bits_size = bits.size
            self._substring_bits_length = substring_bits_length
        else:
            substring_bits_length: int = self._substring_bits_length
        # Count the number of matches of the template in each block
        matches: numpy.ndarray = numpy.zeros(self._blocks_number, dtype=int)
        for i in range(self._blocks_number):
            # Define the block at the current index
            block: numpy.ndarray = bits[i * substring_bits_length:(i + 1) * substring_bits_length]
            # Define counting variables
            position: int = 0
            count: int = 0
            # Count the matches in the block with the chosen template
            while position < (substring_bits_length - b_template.size):
                if (block[position:position + b_template.size] == b_template).all():
                    position += b_template.size
                    count += 1
                else:
                    position += 1
            matches[i] = count
        # Compute mu and sigma
        mu: float = float(substring_bits_length - b_template.size + 1) / float(2 ** b_template.size)
        sigma: float = substring_bits_length * ((1.0 / float(2 ** b_template.size)) - (float((2 * b_template.size) - 1) / float(2 ** (2 * b_template.size))))
        # Compute Chi-square
        chi_square: float = float(numpy.sum(((matches[:] - mu) ** 2) / (sigma ** 2)))
        # If Chi-square is zero, fail the test
        if chi_square != 0:
            # Compute the score (P-value)
            score: float = scipy.special.gammaincc(self._blocks_number / 2.0, chi_square / 2.0)
            # Return result
            if score >= self.significance_value:
                return Result(self.name, True, numpy.array(score))
        return Result(self.name, False, numpy.array(0.0))

    def is_eligible(self,
                    bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # This test is always eligible for any sequence
        return True
