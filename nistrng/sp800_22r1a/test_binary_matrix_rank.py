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
import copy

# Import required src

from nistrng import Test, Result


class BinaryMatrix:
    """
    Binary Matrix containing all the algorithm specified in the NIST suite for computing the **binary rank** of a matrix.
    """

    def __init__(self, block: numpy.ndarray, rows_number: int, columns_number: int):
        self._rows = rows_number
        self._columns = columns_number
        self._matrix = block
        self._base_rank = min(self._rows, self._columns)

    def _perform_row_operations(self, i: int, forward_elimination: bool):
        """
        Performs elementary row operations. This involves xor'ing up to two rows together depending on whether or not
        certain elements in the matrix contain 1 if the "current" element does not.
        :param i: the index of the "current" element of the matrix
        :param forward_elimination: boolean flag if we are performing forward elimination or not
        """
        if forward_elimination:
            # Process all the following rows
            j: int = i + 1
            while j < self._rows:
                if self._matrix[j][i] == 1:
                    self._matrix[j, :] = (self._matrix[j, :] + self._matrix[i, :]) % 2
                j += 1
        else:
            # Process all the previous rows
            j: int = i - 1
            while j >= 0:
                if self._matrix[j][i] == 1:
                    self._matrix[j, :] = (self._matrix[j, :] + self._matrix[i, :]) % 2
                j -= 1

    def _find_unit_element_swap(self, i: int, forward_elimination: bool) -> int:
        """
        Searches through the rows below/above the given index to see which rows contain 1, if they do then they are
        swapped. This is supposed to be called on the forward and backward elimination.
        :param i: the index of the "current" element of the matrix
        :param forward_elimination: boolean flag if we are performing forward elimination or not
        :return: an integer value 0 or 1 depending on row_swap_operation result
        """
        row_swap_operation: int = 0
        if forward_elimination:
            # Process the following rows
            index: int = i + 1
            while index < self._rows and self._matrix[index][i] == 0:
                index += 1
            if index < self._rows:
                row_swap_operation = self._swap_rows(i, index)
        else:
            # Process the previous rows
            index: int = i - 1
            while index >= 0 and self._matrix[index][i] == 0:
                index -= 1
            if index >= 0:
                row_swap_operation = self._swap_rows(i, index)
        return row_swap_operation

    def _swap_rows(self, source_row_index: int, target_row_index: int) -> int:
        """
        This method just swaps two rows in a matrix. We use copy package to ensure no memory leakage.
        :param source_row_index: the row we want to swap (source)
        :param target_row_index: the row we want to swap it with (target)
        :return: an integer value of 1
        """
        # Swap rows
        temp_matrix = copy.copy(self._matrix[source_row_index, :])
        self._matrix[source_row_index, :] = self._matrix[target_row_index, :]
        self._matrix[target_row_index, :] = temp_matrix
        # Always return 1
        return 1

    def _compute_rank(self) -> int:
        """
        Computes the rank of the transformed matrix. It must be called after the matrix is correctly prepared.
        :return: the rank of the transformed matrix
        """
        # Rank start from the minimum value of rows and columns
        rank: int = self._base_rank
        i: int = 0
        # Process all the rows
        while i < self._rows:
            all_zeros: bool = True
            # Process all the columns (check if there is at least a non-zero element)
            for j in range(self._columns):
                if self._matrix[i][j] == 1:
                    all_zeros = False
            # If a row is of all zeros, it's not counted towards the rank
            if all_zeros:
                rank -= 1
            i += 1
        return rank

    def compute_rank(self) -> int:
        """
        Computes the **binary rank** of the matrix.
        :return: an integer defining binary rank of the matrix.
        """
        # Perform row operations with forward elimination
        i: int = 0
        while i < self._base_rank - 1:
            if self._matrix[i][i] == 1:
                self._perform_row_operations(i, True)
            else:
                found = self._find_unit_element_swap(i, True)
                if found == 1:
                    self._perform_row_operations(i, True)
            i += 1
        # Perform row operations without forward elimination
        i = self._base_rank - 1
        while i > 0:
            if self._matrix[i][i] == 1:
                self._perform_row_operations(i, False)
            else:
                if self._find_unit_element_swap(i, False) == 1:
                    self._perform_row_operations(i, False)
            i -= 1
        # Compute the rank of the transformed matrix
        return self._compute_rank()


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
            matrix: BinaryMatrix = BinaryMatrix(block, self._rows_number, self._cols_number)
            rank: int = matrix.compute_rank()
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
        product: float = float(numpy.prod(((1.0 - (2.0 ** (indexes[:] - number_of_cols))) * (1.0 - (2.0 ** (indexes[:] - number_of_rows)))) / (1 - (2.0 ** (indexes[:] - number_of_rows)))))
        return product
