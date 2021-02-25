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


class OverlappingTemplateMatchingTest(Test):
    """
    Overlapping template matching test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of the Overlapping Template Matching test is the number of occurrences of pre-specified target strings.
    Both this test and the Non-overlapping Template Matching test use an m-bit window to search for a specific m-bit pattern.
    As with the other test, if the pattern is not found, the window slides one bit position.
    The difference between this test and the other is that when the pattern is found, the window slides only one bit before resuming the search.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._template_bits_length: int = 10
        self._blocks_number: int = 968
        self._freedom_degrees: int = 5
        self._substring_bits_length: int = 1062
        # Generate base Test class
        super(OverlappingTemplateMatchingTest, self).__init__("Overlapping Template Matching", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Build the template B as a fixed sized sequence of ones
        b_template: numpy.ndarray = numpy.ones(self._template_bits_length, dtype=int)
        # Count the distribution of matches of the template across blocks
        matches_distributions: numpy.ndarray = numpy.zeros(self._freedom_degrees + 1, dtype=int)
        for i in range(self._blocks_number):
            # Define the block at the current index
            block: numpy.ndarray = bits[i * self._substring_bits_length:(i + 1) * self._substring_bits_length]
            # Define counting variable
            count: int = 0
            # Count the matches in the block with respect to the given template
            for position in range(self._substring_bits_length - self._template_bits_length):
                if (block[position:position + self._template_bits_length] == b_template).all():
                    count += 1
            matches_distributions[min(count, self._freedom_degrees)] += 1
        # Define eta and default probabilities (from STS) of size freedom degrees + 1
        eta: float = (self._substring_bits_length - self._template_bits_length + 1.0) / (2.0 ** self._template_bits_length) / 2.0
        probabilities: numpy.ndarray = numpy.array([0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865])
        # Compute probabilities up to degrees of freedom and change the last based on the sum of all of them
        probabilities[:self._freedom_degrees] = self._get_probabilities(numpy.arange(self._freedom_degrees)[:], eta)
        probabilities[-1] = 1.0 - numpy.sum(probabilities)
        # Compute Chi-square
        chi_square: float = float(numpy.sum(((matches_distributions[:] - (self._blocks_number * probabilities[:])) ** 2) / (self._blocks_number * probabilities[:])))
        # If Chi-square is zero, fail the test
        if chi_square != 0:
            # Compute the score (P-value)
            score: float = scipy.special.gammaincc(5.0 / 2.0, chi_square / 2.0)
            # Return result
            if score >= self.significance_value:
                return Result(self.name, True, numpy.array(score))
        return Result(self.name, False, numpy.array(0.0))

    def is_eligible(self,
                    bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Check for eligibility
        # Note: at least 1,028,016 bits are required
        if bits.size < (self._substring_bits_length * self._blocks_number):
            return False
        return True

    @staticmethod
    def _log_gamma(x: []) -> []:
        """
        Compute log-gamma function on the given input list (of integers).

        :param x: input integer list of the log-gamma function
        :return: the log-gamma function computed using scipy and numpy
        """
        return numpy.log(scipy.special.gamma(x))

    @staticmethod
    def _get_probabilities(freedom_degree_values: [], eta_value: float) -> []:
        """
        Compute probabilities at the given freedom values with the given eta value.

        :param freedom_degree_values: the freedom degree values for which to compute probability
        :param eta_value: the eta value for which to compute probability
        :return: the probabilities list
        """
        probabilities: [] = []
        for freedom_degree_value in freedom_degree_values:
            if freedom_degree_value == 0:
                probability: float = numpy.exp(-eta_value)
            else:
                indexes: numpy.ndarray = numpy.arange(1, freedom_degree_value + 1)
                probability: float = float(numpy.sum(numpy.exp(-eta_value - freedom_degree_value * numpy.log(2) + indexes[:] * numpy.log(eta_value) - OverlappingTemplateMatchingTest._log_gamma(indexes[:] + 1) + OverlappingTemplateMatchingTest._log_gamma(freedom_degree_value) - OverlappingTemplateMatchingTest._log_gamma(indexes[:]) - OverlappingTemplateMatchingTest._log_gamma(freedom_degree_value - indexes[:] + 1))))
            probabilities.append(probability)
        return probabilities
