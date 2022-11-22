#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
# Created by  : Gabriel Pereira Alves, Jedrzej Kaniewski and Nicolas Gigena
# Created Date: April 14, 2022
# e-mail: gpereira@fuw.edu.pl
# ==================================================================================================
"""This file contains the a test script for the main functions of utils.py"""
# ==================================================================================================
# Imports: utils and numpy.
# ==================================================================================================


import utils as ut
import numpy as np

from numpy.random import random
from numpy.random import uniform


class colors:
    CYAN =  '\033[96m'
    GREEN = '\033[92m'
    RED =   '\033[91m'
    BOLD =  '\033[1m'
    FAINT = '\033[2m'
    END =   '\033[0m'


def printings(value1, value2, n = None, d = None, weight = None):

    """
    Printings
    ---------

    A function to print the expected and computed values for 'find_QRAC_value'. If the difference
    between these values is smaller than 'difference', then it is printed in green. If not, it is
    printed in red.

    Inputs
    ------
    value1: a float.
        The expected value for the QRAC.
    value2: a float.
        The computed value for the QRAC.
    n: an integer.
        n represents the number of distinct measurements in the QRAC.
    d: an integer.
        d represents the number of outcomes for each measurement. In general, d is also taken as the
        dimension of the measurement operators in the QRAC.
    weight: a float.
        'weight' represents the weight with which the QRAC is biased.
    """

    # Fixing the acceptable difference for expected and computed values to 1e-6.
    difference = 1e-6

    # This command is to allow printing superscripts in the prompt.
    superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    # If the QRAC is biased, so its is desirable to print the weight by which it is biased. So the
    # printing must be different if no weight is attributed.
    if weight == None:
        print(
            colors.CYAN
            + "For the "
            + str(n)
            + str(d).translate(superscript)
            + "-->1 QRAC:"
            + colors.END
            +"\nLiterature value: "
            + str("{:.7f}".format(value1))
            + "   Computed value: "
            + str("{:.7f}".format(value2))
            + "   Difference: ",
            colors.GREEN + colors.BOLD
            + str(float('%.1g' % (value1 - value2)))
            + colors.END
            if abs(value1 - value2) < difference
            else colors.RED + colors.BOLD
            + str(float('%.1g' % (value1 - value2)))
            + colors.END
            )
    else:
        print(
            colors.CYAN
            +"For weight = "
            + str("{:.3f}".format(weight))
            + colors.END
            +":\nExpected quantum value: "
            + str("{:.6f}".format(value1))
            + "   Computed value: "
            + str("{:.6f}".format(value2))
            + "   Difference: ",
            colors.GREEN + colors.BOLD
            + str(float('%.1g' % (value1 - value2)))
            + colors.END
            if abs(value1 - value2) < difference
            else colors.RED + colors.BOLD
            + str(float('%.1g' % (value1 - value2)))
            + colors.END
            )


def test_qubit_qracs(seeds):

    """
    Testing qubit QRACs
    -------------------

    This function tests QRACs of the form nˆ2-->1 in which n is a parameter ranging from 2 to 4. It
    makes use of the function 'find_QRAC_value'.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'find_QRAC_value'.

    Outputs
    -------
    It prints a list containing the literature quantum value, the computed quantum value and the
    difference between both values, for each value of n.

    Reference
    ---------
    The quantum values in 'literature_value' can be found in:
    1. A. Ambainis, D. Leung, L. Mancinska and M. Ozols, Quantum Random Access Codes with Shared
    Randomness, available in arXiv:0810.2937.
    """

    # These are the theoretical quantum values provided by the literature. For n = 2 and n = 3, the
    # quantum value is known exactly. For n > 4 these are just conjectured values. For n = 4, the
    # value below is calculated by using the conjectured measurements for this case, i.e., X, Y, Z
    # and X, where X, Y, and Z are the Pauli matrices.
    literature_value = {
                        2: (1 + 1 / np.sqrt(2)) / 2,
                        3: (1 + 1 / np.sqrt(3)) / 2,
                        4:  0.7414814565722667,
                        5:  0.713578,
                        6:  0.694046,
                        7:  0.678638,
                        8:  0.666633,
                        9:  0.656893,
                        10: 0.648200,
                        11: 0.641051,
                        12: 0.634871
                        }

    for j in range(2, 5):

        # find_QRAC_value returns a dictionary. Here, only the entry "optimal value" of this dic-
        # tionary is interesting.
        computation = ut.find_QRAC_value(j, 2, seeds, verbose = False)
        printings(literature_value[j], computation["optimal value"], j, 2)


def test_higher_dim_qracs(seeds):

    """
    Testing higher dimensional QRACs
    --------------------------------

    This function tests QRACs of the form 2ˆd-->1 in which d is a parameter ranging from 3 to 5. It
    makes use of the function 'find_QRAC_value'.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'find_QRAC_value'.

    Outputs
    -------
    It prints a list containing the literature quantum value, the computed quantum value and the
    difference between both values, for each value of d.

    References
    ----------
    The quantum values in 'literature_value' can be found in:
    1. E. A. Aguilar, J. J. Borkala, P. Mironowicz, M. Pawlowski, Connections between Mutually Un-
    biased Bases and Quantum Random Access Codes, Phys. Rev. Lett., 121, 050501, 2018.
    2. M. Farkas and J. Kaniewski, Self-testing mutually unbiased bases in the prepare-and-measure
    scenario. Phys. Rev. A, 99, 032316, 2019.
    """

    literature_value = {i: (1 + 1 / np.sqrt(i)) / 2 for i in range(3, 6)}

    for j in range(3, 6):

        computation = ut.find_QRAC_value(2, j, seeds, verbose = False)
        printings(literature_value[j], computation["optimal value"], 2, j)


def test_YPARAM(seeds):

    """
    Testing YPARAM bias
    -------------------

    This function tests biased 2ˆ2-->1 QRACs. We consider the 'YPARAM' bias in which the 1st entry
    of Alice is prefered with some weight. Then, we evaluate 'find_QRAC_value' for 3 distinct random
    weights.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'find_QRAC_value'.

    Outputs
    -------
    It prints a list containing the expected quantum value, the computed quantum value and the
    difference between both values, for each random weight of 'YPARAM' bias.

    Reference
    ---------
    The values in 'expected_quantum_value' are to be published soon.
    """

    # Defining the 3 random weights with numpy.random.random().
    random_bias = [random() for i in range(0, 3)]

    # Since this value is not in the literature yet, I'm naming it 'expected_quantum_value'.
    expected_quantum_value = [0.5 + 0.5 * np.sqrt(2 * i ** 2 - 2 * i + 1) for i in random_bias]

    for j in range(0, 3):

        computation = ut.find_QRAC_value(2, 2, seeds,
                                            verbose = False,
                                            bias = "YPARAM",
                                            weight = random_bias[j])

        printings(expected_quantum_value[j], computation["optimal value"], weight = random_bias[j])


def test_BPARAM(seeds):

    """
    Testing BPARAM bias
    -------------------

    This function tests biased 2ˆ2-->1 QRACs. We consider the 'BPARAM' bias in which the entry 0 is
    prefered to be retrieved with some weight. Then, we evaluate 'find_QRAC_value' for 3 distinct
    random weights.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'find_QRAC_value'.

    Outputs
    -------
    It prints a list containing the expected quantum value, the computed quantum value and the
    difference between both values, for each random weight of 'BPARAM' bias.

    Reference
    ---------
    The values in 'expected_quantum_value' are to be published soon.
    """

    # Not all values of 'weight' produce quantum-advantaged QRACs for the 'BPARAM' bias. The first
    # and the last elements of random_bias are produced in the intervals where there is no quantum
    # advantage.
    random_bias = [
                   uniform(0, (3 - np.sqrt(5)) / 4),
                   uniform((3 - np.sqrt(5)) / 4, (1 + np.sqrt(5)) / 4),
                   uniform((3 - np.sqrt(5)) / 4, (1 + np.sqrt(5)) / 4),
                   uniform((1 + np.sqrt(5)) / 4, 1)
                   ]

    # The function expected_BPARAM(i) returns the correct quantum value for the appropriate value of
    # 'weight'.
    expected_quantum_value = [expected_BPARAM(i) for i in random_bias]

    for j in range(0, 4):

        computation = ut.find_QRAC_value(2, 2, seeds,
                                            verbose = False,
                                            bias = "BPARAM",
                                            weight = random_bias[j])

        printings(expected_quantum_value[j], computation["optimal value"], weight = random_bias[j])


def expected_BPARAM(weight):

    """
    An auxiliary function for testing BPARAM. For bias in the retrived entry b, not all values of
    'weight' produce a QRAC with quantum advantage. In particular, the only range where quantum ad-
    vantage is expected is in the interval [(3 - sqrt(5)) / 4, (1 + sqrt(5)) / 4].

    Input
    -----
    weight: a float.
        'weight' represents the weight with which the QRAC is biased for the BPARAM case.

    Output
    The expected quantum value for the correct range.
    """

    if (3 - np.sqrt(5)) / 4 < weight < (1 + np.sqrt(5)) / 4:

        # Just an auxiliary variable for this interval.
        m = weight * (1 - weight)

        return 0.5 + 1 / np.sqrt(4 + 16 * m) * (1 / np.sqrt(16 * m) + np.sqrt(m))

    # There is just one interval inside where 'weight' produces quantum advantage. If 'weight' is
    # not contained inside the interval, the code returns the classical value.
    else:
        return 0.75 + 0.25 * abs(2 * weight - 1)


if __name__ == "__main__":

    """
    Main function
    -------------

    This function just select a few cases of QRACs whose quantum value is theoretically known, as
    follows.

    1. Qubit QRACs. QRACs whose local dimension is 2.
    2. 2-entries QRACs. Here referred in the function 'test_higher_dim_qracs'. QRACs whose number of
    entries of Alice is exactly 2.
    3. 2ˆ2-->1 QRAC with bias in the requested entry y. Here referred in the function 'test_YPARAM'.
    4. 2ˆ2-->1 QRAC with bias in the retrieved entry b. Here referred in the function 'test_BPARAM'.

    Outputs
    -------
    It prints a list containing the literature quantum value, the computed quantum value and the
    difference between both values, for each case.
    """

    # Fixing the seeds to 5.
    seeds = 5

    # Printing the header.
    print(
        "\n"
        + "=" * 80
        + "\n"
        + " " * 32
        + "QRAC-tools v1.0\n"
        + "=" * 80
        + "\n")

    # From now on, I just split the function in the cases I want to test.
    print(
        colors.FAINT
        + colors.BOLD
        + "Testing qubit QRACs"
        + colors.END
        )
    test_qubit_qracs(seeds)

    print(" ")

    print(
        colors.FAINT
        + colors.BOLD
        + "Testing higher dimensional QRACs"
        + colors.END
        )
    test_higher_dim_qracs(seeds)

    print(" ")

    print(
        colors.FAINT
        + colors.BOLD
        + "Testing the y-biased 2\u00b2-->1 QRAC for 3 random weights"
        + colors.END
        )
    test_YPARAM(seeds)

    print(" ")

    print(
        colors.FAINT
        + colors.BOLD
        + "Testing the b-biased 2\u00b2-->1 QRAC for 4 random weights"
        + colors.END
        )
    test_BPARAM(seeds)

    # Printing the footer.
    print(
        '\n'
        + '-' * 30
        + " End of computation "
        + '-' * 30
        )
