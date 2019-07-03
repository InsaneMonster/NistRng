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


class RandomExcursionTest(Test):
    """
    Random excursion test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the number of cycles having exactly K visits in a cumulative sum random walk.
    The cumulative sum random walk is derived from partial sums after the (0,1) sequence is transferred to the appropriate (-1, +1) sequence.
    A cycle of a random walk consists of a sequence of steps of unit length taken at random that begin at and return to the origin.
    The purpose of this test is to determine if the number of visits to a particular state within a cycle deviates from what one would expect
    for a random sequence. This test is actually a series of eight tests (and conclusions), one test and conclusion for each of the
    states: -4, -3, -2, -1 and +1, +2, +3, +4.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._probabilities_xk = [numpy.array([0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0312]),
                                  numpy.array([0.75, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791]),
                                  numpy.array([0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804]),
                                  numpy.array([0.875, 0.0156, 0.0137, 0.012, 0.0105, 0.0733]),
                                  numpy.array([0.9, 0.01, 0.009, 0.0081, 0.0073, 0.0656]),
                                  numpy.array([0.9167, 0.0069, 0.0064, 0.0058, 0.0053, 0.0588]),
                                  numpy.array([0.9286, 0.0051, 0.0047, 0.0044, 0.0041, 0.0531])]
        # Generate base Test class
        super(RandomExcursionTest, self).__init__("Random Excursion", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Copy the bits to a new array
        bits_copy: numpy.ndarray = bits.copy()
        # Convert all the zeros in the array to -1
        bits_copy[bits_copy == 0] = -1
        # Generate the padded cumulative sum of the array of -1, 1
        sum_prime: numpy.ndarray = numpy.concatenate((numpy.array([0]), numpy.cumsum(bits_copy), numpy.array([0]))).astype(int)
        # Compute the cycles iterating over each position of S' (sum_prime) and define the first cycle
        cycles: [] = []
        cycle: [] = [0]
        for index, _ in enumerate(sum_prime[1:]):
            # Once a zero crossing is found add all the non zero elements of S' to the cycle
            # Else wrap up the cycle and start a new cycle
            if sum_prime[index] != 0:
                cycle += [sum_prime[index]]
            else:
                cycle += [0]
                cycles.append(cycle)
                cycle: [] = [0]
        # Append the last cycle
        cycles.append(cycle)
        # Compute the size of the cycles list
        cycles_size: int = len(cycles)
        # Setup frequencies table (Vk(x))
        frequencies_table: dict = {
            -4: numpy.zeros(6, dtype=int),
            -3: numpy.zeros(6, dtype=int),
            -2: numpy.zeros(6, dtype=int),
            -1: numpy.zeros(6, dtype=int),
            1: numpy.zeros(6, dtype=int),
            2: numpy.zeros(6, dtype=int),
            3: numpy.zeros(6, dtype=int),
            4: numpy.zeros(6, dtype=int),
        }
        # Count occurrences
        for value in frequencies_table.keys():
            for k in range(frequencies_table[value].size):
                count: int = 0
                # Count how many cycles in which x occurs k times
                for cycle in cycles:
                    # Count how many times the value used as key of the table occurs in the current cycle
                    occurrences: int = numpy.count_nonzero(numpy.array(cycle) == value)
                    # If the value occurs k times, increment the cycle count
                    if 5 > k == occurrences:
                        count += 1
                    elif occurrences >= 5:
                        count += 1
                frequencies_table[value][k] = count
        # Compute the scores (P-values)
        scores: [] = []
        for value in frequencies_table.keys():
            # Compute Chi-Square for this value
            chi_square: float = numpy.sum(((frequencies_table[value][:] - (cycles_size * (self._probabilities_xk[abs(value) - 1][:]))) ** 2) / (cycles_size * self._probabilities_xk[abs(value) - 1][:]))
            # Compute the P-value for this value
            score: float = scipy.special.gammaincc(5.0 / 2.0, chi_square / 2.0)
            scores.append(score)
        # Return result
        if all(score >= self.significance_value for score in scores):
            return Result(self.name, True, numpy.array(scores))
        return Result(self.name, False, numpy.array(scores))

    def is_eligible(self,
                    bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # This test is always eligible for any sequence
        return True
