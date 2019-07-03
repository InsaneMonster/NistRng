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

# Import scripts

from .test_monobit import MonobitTest
from .test_frequency_within_block import FrequencyWithinBlockTest
from .test_runs import RunsTest
from .test_longest_run_ones_in_a_block import LongestRunOnesInABlockTest
from .test_binary_matrix_rank import BinaryMatrixRankTest
from .test_discrete_fourier_transform import DiscreteFourierTransformTest
from .test_non_overlapping_template_matching import NonOverlappingTemplateMatchingTest
from .test_overlapping_template_matching import OverlappingTemplateMatchingTest
from .test_maurers_universal import MaurersUniversalTest
from .test_linear_complexity import LinearComplexityTest
from .test_serial import SerialTest
from .test_approximate_entropy import ApproximateEntropyTest
from .test_cumulative_sums import CumulativeSumsTest
from .test_random_excursion import RandomExcursionTest
from .test_random_excursion_variant import RandomExcursionVariantTest