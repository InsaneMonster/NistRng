#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a
# Creative Commons Attribution-ShareAlike 3.0 Unported License.
#
# You should have received a copy of the license along with this
# work. If not, see <http://creativecommons.org/licenses/by-sa/3.0/>.

# Import packages

import numpy
import math

# Import required src

from nistrng import Test, Result


class BinaryMatrixRankTest(Test):
    """
    Binary matrix rank test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of the test is the rank of disjoint sub-matrices of the entire sequence.
    The purpose of this test is to check for linear dependence among fixed length substrings of the original sequence.
    Note that this test also appears in the DIEHARD battery of test

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Define specific test attributes
        self._rows_number: int = 32
        self._cols_number: int = 32
        self._block_size_min: int = 38
        # Compute the reference probabilities for full rank, full rank minus one and remained matrix rank (which is 1.0 minus the sum of the other probabilities)
        self._full_rank_probability: float = self._product(self._rows_number, self._cols_number) * (2.0 ** ((self._rows_number * (self._cols_number + self._rows_number - self._rows_number)) - (self._rows_number * self._cols_number)))
        self._minus_rank_probability: float = self._product(self._rows_number - 1, self._cols_number) * (2.0 ** ((self._rows_number * (self._cols_number + self._rows_number - self._rows_number)) - (self._rows_number * self._cols_number)))
        self._remained_rank_probability: float = 1.0 - (self._full_rank_probability + self._minus_rank_probability)
        # Define cache attributes
        self._last_bits_size: int = -1
        self._blocks_number: int = -1
        # Generate base Test class
        super(BinaryMatrixRankTest, self).__init__("Binary Matrix Rank", 0.01)

    def _execute(self,
                 bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Reload values is cache is empty or no longer up-to-date
        # Otherwise, use cache
        if self._last_bits_size == -1 or self._last_bits_size != bits.size:
            # Compute the number of blocks
            blocks_number: int = int(math.floor(bits.size / (self._rows_number * self._cols_number)))
            # Save in the cache
            self._last_bits_size = bits.size
            self._blocks_number = blocks_number
        else:
            blocks_number: int = self._blocks_number
        # Compute the number of full rank, minus rank and remained rank matrices
        full_rank_matrices: int = 0
        minus_rank_matrices: int = 0
        remainder: int = 0
        for i in range(blocks_number):
            # Get the bits in the block and reshape them in a 2D array (the matrix)
            block: numpy.ndarray = bits[i * (self._rows_number * self._cols_number):(i + 1) * (self._rows_number * self._cols_number)].reshape((self._rows_number, self._cols_number))
            # Compute rank of the block matrix
            rank: int = numpy.linalg.matrix_rank(block)
            # Count the result
            if rank == self._rows_number:
                full_rank_matrices += 1
            elif rank == self._rows_number - 1:
                minus_rank_matrices += 1
            else:
                remainder += 1
        # Compute Chi-square
        chi_square: float = (((full_rank_matrices - (self._full_rank_probability * blocks_number)) ** 2) / (self._full_rank_probability * blocks_number)) + (((minus_rank_matrices - (self._minus_rank_probability * blocks_number)) ** 2) / (self._minus_rank_probability * blocks_number)) + (((remainder - (self._remained_rank_probability * blocks_number)) ** 2) / (self._remained_rank_probability * blocks_number))
        # Compute the score (P-value)
        score: float = math.e ** (-chi_square / 2.0)
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
        blocks_number: int = int(math.floor(bits.size / (self._rows_number * self._cols_number)))
        if blocks_number < self._block_size_min:
            return False
        return True

    @staticmethod
    def _product(number_of_rows: int, number_of_cols: int) -> float:
        """
        Compute the matrix rank frequency product using numpy.

        :param number_of_rows: number of rows of the matrix
        :param number_of_cols:number of columns of the matrix
        :return: the float value of the matrix rank frequency product
        """
        # Compute the product used to compute the probabilities of each kind of matrix rank frequency
        indexes: numpy.ndarray = numpy.arange(number_of_rows)
        product: float = numpy.prod(((1.0 - (2.0 ** (indexes[:] - number_of_cols))) * (1.0 - (2.0 ** (indexes[:] - number_of_rows)))) / (1 - (2.0 ** (indexes[:] - number_of_rows))))
        return product
