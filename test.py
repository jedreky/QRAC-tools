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
import constants as const

from numpy.random import random
from numpy.random import uniform


class colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    END = "\033[0m"


def printings(value1, value2, n=None, d=None, weight=None):

    """
    Printings
    ---------

    A function to print the expected and computed values for 'perform_seesaw'. If the difference
    between these values is smaller than 'const.BOUND', then it is printed in green. If not, it is
    printed in red.

    Inputs
    ------
    value1: a float.
        The expected value for the QRAC.
    value2: a float.
        The computed value for the QRAC.
    n: an integer.
        n represents the number of input characters for a given RAC.
    d: an integer.
        d represents the cardinality of the outputs of Bob. In general, d is also taken as the local
        dimension of the measurement operators in a QRAC.
    weight: a float.
        'weight' represents the weight with which the RAC is biased.
    """

    # This command is to allow printing superscripts in the prompt.
    superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    # If the QRAC is biased, it is desirable to print the weight by which it is biased. So the print
    # must be different if no weight is attributed.
    if weight is None:
        print(
            colors.CYAN
            + "For the "
            + str(n)
            + str(d).translate(superscript)
            + "-->1 QRAC:"
            + colors.END
            + "\nLiterature value: "
            + str("{:.7f}".format(value1))
            + "   Computed value: "
            + str("{:.7f}".format(value2))
            + "   Difference: ",
            colors.GREEN
            + colors.BOLD
            + str(float("%.1g" % (value1 - value2)))
            + colors.END
            if abs(value1 - value2) < const.BOUND
            else colors.RED
            + colors.BOLD
            + str(float("%.1g" % (value1 - value2)))
            + colors.END,
        )
    else:
        print(
            colors.CYAN
            + "For weight = "
            + str("{:.3f}".format(weight))
            + colors.END
            + ":\nExpected quantum value: "
            + str("{:.6f}".format(value1))
            + "   Computed value: "
            + str("{:.6f}".format(value2))
            + "   Difference: ",
            colors.GREEN
            + colors.BOLD
            + str(float("%.1g" % (value1 - value2)))
            + colors.END
            if abs(value1 - value2) < const.BOUND
            else colors.RED
            + colors.BOLD
            + str(float("%.1g" % (value1 - value2)))
            + colors.END,
        )


def test_qubit_qracs(seeds):

    """
    Testing qubit QRACs
    -------------------

    This function tests QRACs of the form nˆ2-->1 in which n is a parameter ranging from 2 to 4. It
    makes use of the function 'perform_seesaw'.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'perform_seesaw'.

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

    # These are known quantum values provided by the literature. For n = 2 and n = 3, the quantum
    # value is known exactly. For n > 5, these are just numerical estimations. For n = 4, the value
    # below is calculated by using the conjectured measurements for this case, i.e., X, Y, Z and X,
    # where X, Y, and Z are the Pauli matrices.
    literature_value = {
        2: (1 + 1 / np.sqrt(2)) / 2,
        3: (1 + 1 / np.sqrt(3)) / 2,
        4: 0.7414814565722667,
        5: 0.713578,
        6: 0.694046,
        7: 0.678638,
        8: 0.666633,
        9: 0.656893,
        10: 0.648200,
        11: 0.641051,
        12: 0.634871,
    }

    for j in range(2, 5):

        # perform_seesaw returns a dictionary. Here, only the entry "optimal value" of this dic-
        # tionary is used.
        computation = ut.perform_seesaw(j, 2, seeds, verbose=False)
        printings(literature_value[j], computation["optimal value"], j, 2)


def test_higher_dim_qracs(seeds):

    """
    Testing higher dimensional QRACs
    --------------------------------

    This function tests QRACs of the form 2ˆd-->1 in which d is a parameter ranging from 3 to 5. It
    makes use of the function 'perform_seesaw'.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'perform_seesaw'.

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

        computation = ut.perform_seesaw(2, j, seeds, verbose=False)
        printings(literature_value[j], computation["optimal value"], 2, j)


def test_Y_ONE(seeds):

    """
    Testing Y_ONE bias
    -------------------

    This function tests 2^2-->1 biased QRACs. We consider the case 'Y_ONE', in which the 1st char-
    acter of Alice is prefered with some weight. Then, we evaluate 'perform_seesaw' for 3 distinct
    random weights.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'perform_seesaw'.

    Outputs
    -------
    It prints a list containing the expected quantum value, the computed quantum value and the
    difference between both values, for each weight.

    Reference
    ---------
    The values in 'expected_quantum_value' can be found in the appendix C of:
    G. P. Alves, N. Gigena and J. Kaniewski, Biased Random Access Codes, to be published soon.
    """

    # Defining the 3 random weights with numpy.random.random().
    random_bias = [random() for i in range(0, 3)]

    # Since this value is not in the literature yet, I'm naming it 'expected_quantum_value'.
    expected_quantum_value = [
        0.5 + 0.5 * np.sqrt(2 * i ** 2 - 2 * i + 1) for i in random_bias
    ]

    for j in range(0, 3):

        computation = ut.perform_seesaw(
            2, 2, seeds, verbose=False, bias="Y_ONE", weight=random_bias[j]
        )

        printings(
            expected_quantum_value[j],
            computation["optimal value"],
            weight=random_bias[j],
        )


def test_B_ONE(seeds):

    """
    Testing B_ONE bias
    -------------------

    This function tests 2^2-->1 biased QRACs. We consider the 'B_ONE' case, in which the output 0 is
    prefered to be retrieved with some weight. Then, we evaluate 'perform_seesaw' for 4 distinct
    random weights.

    Input
    -----
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm im the
        function 'perform_seesaw'.

    Outputs
    -------
    It prints a list containing the expected quantum value, the computed quantum value and the dif-
    ference between both values, for each weight.

    Reference
    ---------
    The values in 'expected_quantum_value' can be found in the appendix C of:
    G. P. Alves, N. Gigena and J. Kaniewski, Biased Random Access Codes, to be published soon.
    """

    # Not all values of 'weight' produce QRACs with quantum advantaged for the 'B_ONE' case. The
    # first and the last elements of random_bias are produced in the intervals where there is no
    # quantum advantage.
    random_bias = [
        uniform(0, (3 - np.sqrt(5)) / 4),
        uniform((3 - np.sqrt(5)) / 4, (1 + np.sqrt(5)) / 4),
        uniform((3 - np.sqrt(5)) / 4, (1 + np.sqrt(5)) / 4),
        uniform((1 + np.sqrt(5)) / 4, 1),
    ]

    # The function expected_B_ONE(i) returns the correct quantum value for the appropriate value of
    # 'weight'.
    expected_quantum_value = [expected_B_ONE(i) for i in random_bias]

    for j in range(0, 4):

        computation = ut.perform_seesaw(
            2, 2, seeds, verbose=False, bias="B_ONE", weight=random_bias[j]
        )

        printings(
            expected_quantum_value[j],
            computation["optimal value"],
            weight=random_bias[j],
        )


def expected_B_ONE(weight):

    """
    An auxiliary function for testing B_ONE. For bias in the retrived output b, not all values of
    'weight' produce a QRAC with quantum advantage. In particular, the only range where quantum ad-
    vantage is expected is in the interval [(3 - sqrt(5)) / 4, (1 + sqrt(5)) / 4].

    Input
    -----
    weight: a float.
        'weight' represents the weight with which the QRAC is biased for the B_ONE case.

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

    This function just select a few cases of quantum RACs (or QRACs) whose quantum value is theore-
    tically known, as follows.

    1. Qubit QRACs. QRACs whose local dimension is 2.
    2. 2-qudits QRACs. Here referred in the function 'test_higher_dim_qracs'. These are QRACs whose
    number of input characters is 2.
    3. 2ˆ2-->1 QRAC with bias in the requested input y. Here referred in the function 'test_Y_ONE'.
    4. 2ˆ2-->1 QRAC with bias in the retrieved output b. Here referred in the function 'test_B_ONE'.

    Outputs
    -------
    It prints a list containing the literature quantum value, the computed quantum value and the
    difference between both values, for each case.
    """

    # Fixing the seeds to 5.
    seeds = 5

    # Printing the header.
    print("\n" + "=" * 80 + "\n" + " " * 33 + "RAC-tools v1.0\n" + "=" * 80 + "\n")

    # From now on, I just split the function in the cases I want to test.
    print(colors.FAINT + colors.BOLD + "Testing qubit QRACs" + colors.END)
    test_qubit_qracs(seeds)

    print(" ")

    print(colors.FAINT + colors.BOLD + "Testing higher dimensional QRACs" + colors.END)
    test_higher_dim_qracs(seeds)

    print(" ")

    print(
        colors.FAINT
        + colors.BOLD
        + "Testing the y-biased 2\u00b2-->1 QRAC for 3 random weights"
        + colors.END
    )
    test_Y_ONE(seeds)

    print(" ")

    print(
        colors.FAINT
        + colors.BOLD
        + "Testing the b-biased 2\u00b2-->1 QRAC for 4 random weights"
        + colors.END
    )
    test_B_ONE(seeds)

    # Printing the footer.
    print("\n" + "-" * 30 + " End of computation " + "-" * 30)
