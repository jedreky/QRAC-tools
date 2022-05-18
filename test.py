#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
# Created by  : Gabriel Pereira Alves, Jedrzej Kaniewski and Nicolas Gigena
# Created Date: April 14, 2022
# e-mail: gpereira@fuw.edu.pl
# ==================================================================================================
"""This file contains the a test script for the main functions of utils.py"""
# ==================================================================================================
# Imports: utils, numpy and sigfig
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

    # If the QRAC is biased, so its is desirable to print the weight by which it is biased. So the
    # printing must be different if no weight is attributed.
    if weight == None:
        print(
            colors.CYAN
            + "For the "
            + str(n)
            + "ˆ"
            + str(d)
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
    It prints a list cointaining the literatute quantum value, the computed quantum value and the
    difference between both values, for each value of n.

    Reference
    ---------
    The quantum values in 'literature_value' can be found in:
    1. A. Ambainis, D. Leung, L. Mancinska and M. Ozols, Quantum Random Access Codes with Shared
    Randomness, available in arXiv:0810.2937.
    """

    # This is the theoretical quantum value provided by the literatute. For n = 2 and n = 3, the
    # quantum value is known exactly.
    literature_value = {
                        2: (1 + 1 / np.sqrt(2)) / 2,
                        3: (1 + 1 / np.sqrt(3)) / 2,
                        4:  0.741481,
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
        computed_value = ut.find_QRAC_value(j, 2, seeds, verbose = False)
        printings(literature_value[j], computed_value, j, 2)


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
    It prints a list cointaining the literatute quantum value, the computed quantum value and the
    difference between both values, for each value of d.

    Reference
    ---------
    The quantum values in 'literature_value' can be found in:
    1. M. Farkas and J. Kaniewski, Self-testing mutually unbiased bases in the prepare-and-measure
    scenario. Phys. Rev. A, 99, 032316, 2019.
    """

    literature_value = {i: (1 + 1 / np.sqrt(i)) / 2 for i in range(3, 6)}

    for j in range(3, 6):

        computed_value = ut.find_QRAC_value(2, j, seeds, verbose = False)
        printings(literature_value[j], computed_value, 2, j)


def test_YPARAM(seeds):

    """
    Testing YPARAM bias
    -------------------

    This function tests biased 2ˆ2-->1 QRACs. We consider the 'YPARAM' bias in which the 1st digit
    of Alice is prefered with some weight. Then, we evaluate 'find_QRAC_value' for 3 distinct random
    weights.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'find_QRAC_value'.

    Outputs
    -------
    It prints a list cointaining the expected quantum value, the computed quantum value and the
    difference between both values, for each random weight of 'YPARAM' bias.

    Reference
    ---------
    The values in 'expected_quantum_value' are to be published soon.
    """

    # Defining the random weights with numpy.random.random().
    random_bias = [random() for i in range(0, 3)]

    # Since this value is not in the literatute yet, I'm naming it 'expected_quantum_value'.
    expected_quantum_value = [0.5 + 0.5 * np.sqrt(2 * i ** 2 - 2 * i + 1) for i in random_bias]

    for j in range(0, 3):

        computed_value = ut.find_QRAC_value(2, 2, seeds,
                                            verbose = False,
                                            bias = "YPARAM",
                                            weight = random_bias[j])

        printings(expected_quantum_value[j], computed_value, weight = random_bias[j])


def test_BPARAM(seeds):

    """
    Testing BPARAM bias
    -------------------

    This function tests biased 2ˆ2-->1 QRACs. We consider the 'BPARAM' bias in which the digit 0 is
    prefered to be retrieved with some weight. Then, we evaluate 'find_QRAC_value' for 3 distinct
    random weights.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'find_QRAC_value'.

    Outputs
    -------
    It prints a list cointaining the expected quantum value, the computed quantum value and the
    difference between both values, for each random weight of 'BPARAM' bias.

    Reference
    ---------
    The values in 'expected_quantum_value' are to be published soon.
    """

    # Not all values of 'weight' produce quantum-advantaged QRACs for the 'BPARAM' bias. The first
    # and the last elements of random_bias are produced in the intervals where there is no quantum
    # advantage. This is why the formulas in 'expected_quantum_value' are different.
    random_bias = [
                   uniform(0, (3 - np.sqrt(5)) / 4),
                   uniform((3 - np.sqrt(5)) / 4, (1 + np.sqrt(5)) / 4),
                   uniform((3 - np.sqrt(5)) / 4, (1 + np.sqrt(5)) / 4),
                   uniform((1 + np.sqrt(5)) / 4, 1)
                   ]

    # Just an auxiliary variable for 'expected_quantum_value'.
    m = random_bias[1] * (1 - random_bias[1])
    n = random_bias[2] * (1 - random_bias[2])


    # 3/4 + (1/4) * abs(1 - 2 * weight) represents the classical value for the 'BPARAM' bias.
    expected_quantum_value = [
                            0.75 + 0.25 * abs(1 - 2 * random_bias[0]),
                            0.5 + 1 / np.sqrt(4 + 16 * m) * (1 / np.sqrt(16 * m) + np.sqrt(m)),
                            0.5 + 1 / np.sqrt(4 + 16 * n) * (1 / np.sqrt(16 * n) + np.sqrt(n)),
                            0.75 + 0.25 * abs(1 - 2 * random_bias[3])
                            ]

    for j in range(0, 4):

        computed_value = ut.find_QRAC_value(2, 2, seeds,
                                            verbose = False,
                                            bias = "BPARAM",
                                            weight = random_bias[j])

        printings(expected_quantum_value[j], computed_value, weight = random_bias[j] )


if __name__ == "__main__":

    """
    Main function
    -------------

    This function just select a few cases of QRACs whose quantum value is theoretically known, as
    follows.

    1. Qubit QRACs. QRACs whose local dimension is 2.
    2. 2-digit QRACs. Here referred in the function 'test_higher_dim_qracs'. QRACs whose number of
    digits of Alice is exactly 2.
    3. 2ˆ2-->1 QRAC with bias in the requested digit y. Here referred in the function 'test_YPARAM'.
    4. 2ˆ2-->1 QRAC with bias in the retrieved digit b. Here referred in the function 'test_BPARAM'.

    Outputs
    -------
    It prints a list cointaining the literatute quantum value, the computed quantum value and the
    difference between both values, for each case.
    """

    # Fixing the seeds to 5.
    seeds = 1

    # Printing the header.
    print(
        "\n"
        + "=" * 80
        + "\n"
        + " " * 32
        + "QRAC-tools 1.0 v\n"
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
        + "Testing the y-biased 2ˆ2-->1 QRAC for random weights"
        + colors.END
        )
    test_YPARAM(seeds)

    print(" ")

    print(
        colors.FAINT
        + colors.BOLD
        + "Testing the b-biased 2ˆ2-->1 QRAC for random weights"
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
