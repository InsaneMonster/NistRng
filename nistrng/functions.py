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

# Import required src

from nistrng import Test

from nistrng.sp800_22r1a import *

# Define default NIST battery constant

SP800_22R1A_BATTERY: dict = {
                                "monobit": MonobitTest(),
                                "frequency_within_block": FrequencyWithinBlockTest(),
                                "runs": RunsTest(),
                                "longest_run_ones_in_a_block": LongestRunOnesInABlockTest(),
                                "binary_matrix_rank": BinaryMatrixRankTest(),
                                "dft": DiscreteFourierTransformTest(),
                                "non_overlapping_template_matching": NonOverlappingTemplateMatchingTest(),
                                "overlapping_template_matching": OverlappingTemplateMatchingTest(),
                                "maurers_universal": MaurersUniversalTest(),
                                "linear_complexity": LinearComplexityTest(),
                                "serial": SerialTest(),
                                "approximate_entropy": ApproximateEntropyTest(),
                                "cumulative sums": CumulativeSumsTest(),
                                "random_excursion": RandomExcursionTest(),
                                "random_excursion_variant": RandomExcursionVariantTest()
                            }

# Define cache global variables
# Note: each test is defined by a tuple name and instance

_cached_tests: [] = []


# Define functions

def run_all_battery(bits: numpy.ndarray, battery: dict,
                    check_eligibility: bool = True) -> []:
    """
    Run all the given tests in the battery with the given bits as input.
    E.g. of a battery of test is the sp800-22r1a test battery.

    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of test (dict with keys the names and values the classes extending Test) to run on the sequence
    :param check_eligibility: whether to check or not for eligibility. If checked and failed, the associate test returns None
    :return: a list of Result objects zipped each one with its own elapsed time or Nones for each not eligible test (if check is required)
    """
    # Run all the tests in the battery by name
    results: [] = []
    for name in battery.keys():
        results.append(run_by_name_battery(name, bits, battery, check_eligibility))
    return results


def run_in_order_battery(bits: numpy.ndarray, battery: dict,
                         check_eligibility: bool = True) -> []:
    """
    Run all the given tests in the battery with the given bits as input and strictly following the dict order, i.e. not
    trying the following test is the one before is not passed.
    E.g. of a battery of test is the sp800-22r1a test battery.

    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of test (dict with keys the names and values the classes extending Test) to run on the sequence
    :param check_eligibility: whether to check or not for eligibility. If checked and failed, the associate test returns None
    :return: a list of Result objects zipped each one with its own elapsed time or Nones for each not eligible test (if check is required)
    """
    # Run all the tests in the battery by name
    results: [] = []
    for name in battery.keys():
        result, elapsed_time = run_by_name_battery(name, bits, battery, check_eligibility)
        results.append((result, elapsed_time))
        # Stop when a test is not passed
        if not result.passed:
            break
    return results


def run_by_name_battery(test_name: str,
                        bits: numpy.ndarray, battery: dict,
                        check_eligibility: bool = True) -> ():
    """
    Run the given test in the battery by name with the given bits as input.
    E.g. of a battery of test is the sp800-22r1a test battery.

    :param test_name: the name of the test to run
    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of test (dict with keys the names and values the classes extending Test) to run on the sequence
    :param check_eligibility: whether to check or not for eligibility. If checked and failed, return None
    :return: a Result object and the relative elapsed time if eligible, None otherwise (if check is required)
    """
    # Generate the test or fetch it from cache if possible
    test: Test or None = None
    for name, instance in _cached_tests:
        if name == test_name:
            test = instance
    if test is None:
        test = battery[test_name]
        _cached_tests.append((test_name, test))
    # Check for eligibility if required
    if check_eligibility:
        # If not eligible, return nothing
        if not test.is_eligible(bits):
            return None
    # Return test result and elapsed time
    return test.run(bits)


def check_eligibility_all_battery(bits: numpy.ndarray, battery: dict) -> dict:
    """
    Check the eligibility for  all the given tests in the battery with the given bits as input.
    E.g. of a battery of test is the sp800-22r1a test battery.

    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of test (dict with keys the names and values the classes extending Test) to run on the sequence
    :return: a dict with names and relative classes to use as a new battery to send into the run function
    """
    # Check eligibility all the tests in the battery by name and return the eligible test dictionary
    results: dict = {}
    for name in battery.keys():
        if check_eligibility_by_name_battery(name, bits, battery):
            results[name] = battery[name]
    return results


def check_eligibility_by_name_battery(test_name: str,
                                      bits: numpy.ndarray, battery: dict) -> bool:
    """
    Run the given test in the battery by name with the given bits as input.
    E.g. of a battery of test is the sp800-22r1a test battery.

    :param test_name: the name of the test to run
    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of test (dict with keys the names and values the classes extending Test) to run on the sequence
    :return: a boolean flag of True if eligible, false otherwise
    """
    # Generate the test or fetch it from cache if possible
    test: Test or None = None
    for name, instance in _cached_tests:
        if name == test_name:
            test = instance
    if test is None:
        test = battery[test_name]
        _cached_tests.append((test_name, test))
    # Check for eligibility
    return test.is_eligible(bits)


def pack_sequence(sequence: numpy.ndarray) -> numpy.ndarray:
    """
    Pack a sequence of signed integers to its binary 8-bit representation using numpy.

    :param sequence: the integer sequence to pack (in the form of a numpy array, ndarray)
    :return: the sequence packed in 8-bit integer in the form of a numpy array (ndarray)
    """
    return numpy.unpackbits(numpy.array(sequence, dtype=numpy.uint8)).astype(numpy.int8)


def unpack_sequence(sequence_binary_encoded: numpy.ndarray) -> numpy.ndarray:
    """
    Unpack a sequence of numbers represented with 8-bits to its signed integer representation using numpy.

    :param sequence_binary_encoded: the 8-bit numbers sequence to unpack (in the form of a numpy array, ndarray)
    :return: the sequence unpacked in signed integer in the form of a numpy array (ndarray)
    """
    return numpy.packbits(numpy.array(sequence_binary_encoded)).astype(numpy.int8)
