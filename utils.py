#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
# Created by  : Gabriel Pereira Alves, Jedrzej Kaniewski and Nicolas Gigena
# Created Date: March 16, 2022
# e-mail: gpereira@fuw.edu.pl
# version = '1.0'
# ==================================================================================================
"""This file contains the main functions for optimizing an nˆd --> 1 QRAC. The main function can
be accessed by the command `find_QRAC_value`."""
# ==================================================================================================
# Imports: time, cvxpy, numpy, scipy and itertools
# ==================================================================================================

import cvxpy as cp
import numpy as np
import scipy as sp
import time as tp

import numpy.linalg as nalg

from numpy.random import rand
from scipy.linalg import sqrtm
from scipy.stats import unitary_group
from itertools import product


class colors:
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    END = "\033[0m"


# Thresholds for the optimization problem. The problem value converges much faster than the varia-
# bles, so we can allow a tighter tolerance for the problem value as low as PROB_BOUND.
PROB_BOUND = 1e-9

# BOUND is used as tolerance to check the projectivity, rankness, etc, of measurement operators. It
# is also used as the default value for the convergence criterion in meas_bound.
BOUND = 1e-7

# MUB_BOUND is used in the function check_if_MUBs. This function is less accurate than the other
# checks. Our analycal results allow us to trust in a higher tolerance for this particular check.
MUB_BOUND = 5e-6

# This is the solver's maximaum accuracy. By default, it must be lower than PROB_BOUND and BOUND.
# Decreasing this number might compromise the feasibility of the problems.
MOSEK_ACCURACY = 1e-13

# Maximal number of iterations for the optimization problem.
ITERATIONS = 100


def generate_random_measurements(n, d, m):

    """
    Generate random measurements
    ----------------------------

    This function generates n random measurements with m outcomes and dimension d. The first step is
    to generate m random complex operators named random_op. Then, it transforms these operators into
    a positive semidefinite matrix and stores them in the list partial_meas. By rescaling partial_
    sum using the sum of all random_herm operators, it can produce a random measurement.

    Input
    -----
    n: an integer.
        n represents the number of distinct measurements.
    d: an integer
        The dimension of the measurement operators.
    m: an integer
        The number of outcomes for the generated measurement.

    Output
    ------
    M: a list of n lists. Each sub-list of M contains m matrices of size d x d.
        M represents a list with n d-dimensional random measurements with m outcomes.
    """

    # Initializing an empty list of measurements.
    measurements_list = []
    for i in range(0, n):

        # Trying to generate one measurement for 10 times. If, in a certain iteration, a suitable
        # measurement is not produced, the code skips the iteration in the "CHECKING" points.
        attempts = 0
        while attempts < 10:
            attempts += 1

            # Initializing an empty list.
            partial_meas = []

            # Creating d random complex operators and appending them to the list partial_meas.
            for j in range(0, m):

                # Generating random complex operators. Each entry should have both the real and ima-
                # ginary parts between [-1, +1].
                random_op = (
                    2 * rand(d, d)
                    - np.ones((d, d))
                    + 1j * (2 * rand(d, d) - np.ones((d, d)))
                )

                # Transforming random_op into a hermitian operator. Note that random_herm is also
                # positive semidefinite by construction.
                random_herm = random_op.T.conj() @ random_op

                partial_meas.append(random_herm)

            partial_sum = np.sum(partial_meas, axis=0)

            # CHECKING if partial_sum is full-rank
            full_rank = nalg.matrix_rank(partial_sum, tol=BOUND, hermitian=True) == d

            if full_rank:

                # Initializing an empty list.
                measurement = []

                # This is the operator I will use to rescale the partial_meas list. It is only the
                # inverse square root of partial_sum.
                inv_sqrt = nalg.inv(sqrtm(partial_sum))

                # Generating the random measurement operators and appending to the list measurement.
                error = False
                for j in range(0, m):

                    # Rescaling partial_meas[i]
                    M = inv_sqrt @ partial_meas[j] @ inv_sqrt
                    # Enforcing hermiticity
                    M = 0.5 * (M + M.conj().T)

                    measurement.append(M)

                    # CHECKING if M is positive semidefinite. Recall that eigh produces ordered eige
                    # nvalues. It suffices to check if the first eigenvalue is non-negative. If it
                    # is not the case, the boolean variable error is changed to True and another ite
                    # ration of the while loop will start.
                    eigval, eigvet = nalg.eigh(M)

                    if eigval[0] < -BOUND:
                        error = True

                if not error:

                    # Last check. CHECKING if the measurement operators sum to identity
                    sum = np.sum(measurement, axis=0)
                    sum_to_identity = nalg.norm(sum - np.eye(d)) < BOUND

                    # If the last check is satisfied, then `measurement` is finally appended to M.
                    if sum_to_identity:
                        measurements_list.append(measurement)
                        break

        # The execution is finished in the case where this function cannot produce a suitable measu-
        # rement.
        if attempts == 10:
            raise RuntimeError("a random measurement cannot be generated")

    return measurements_list


def find_opt_prep(M, d, n, m, bias):

    """
    Find optimal preparations
    -------------------------

    This function acts jointly with `opt_state` function. The objective is to generate the set of
    optimal states for a set of measurements M. It produces a dictionary opt_preps that contains the
    optimal preparations for a given combination of measurement operators.

    Inputs
    ------
    M: a list of lists whose elements are matrices of size d x d.
        M is a list containing n lists. Each list inside M corresponds to a different measurement in
        the QRAC task. For each measurement there are m measurement operators.
    d: an integer.
        d represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements.
    m: an integer.
        m represents the number of outcomes of each measurement.
    bias: a dictionary of size n * m ** n. bias.keys() are tuples of n + 2 coordinates. bias.value
    s() are floats.
        The dictionary bias represents a order-(n+2) tensor encoding the bias in some given QRAC.

    Output
    ------
    opt_preps: a dictionary in which the keys are given by an n-tuple and the content of each entry
    is a matrix of size d x d.
        Every element of this dictionary corresponds to a rank-one density matrix for a given prepa-
        ration of Alice. In a n-dits QRAC, in which the dits are labelled as x_1, x_2, ..., x_n, for
        every combination of dits, a state is prepared by Alice and sent to Bob. The dictionary opt_
        preps contains the optimal preparations related to each combination of dits. It is optimal
        in the sense that it maximizes the figure-of-merit "success probability" for a given set of
        measurements.

    Example
    -------
    Let us say that a QRAC with 3 dits is set. Then, we indicate a combination of measurement opera-
    tors by a tuple of size 3. The tuple (3, 4, 5) indicates the third outcome for the first measu-
    rement, the fourth outcome for the second measurement and the fifth for the third measurement.
    The optimal preparation for the combination of measurement operators (3, 4, 5) is saved in the
    dictionary opt_preps whose key corresponds to the tuple (3, 4, 5).
    """

    # Creating an empty dictionary.
    opt_preps = {}

    # The variable indexes_of_x list all the possible tuples of size n with elements ranging from
    # 0 to m - 1.
    indexes_of_x = product(range(0, m), repeat=n)

    # This for runs over all n-tuples of indexes_of_x. In practice, it is a for nested n times.
    for i in indexes_of_x:

        # Initializing an empty list.
        operators_list = []

        for j in range(0, n):
            for k in range(0, m):

                # Here, I am running over all tuples i, and indexes j and k. The tuple (i, j,
                # k) corresponds exactly to the keys of the dicitonary bias. Then, for each ele-
                # ment bias[(i, j, k)] I multiply the correct measurement operator and append to
                # operators_list. Finally, I compute the optimal preparation for these measurement
                # operators.
                operators_list.append(bias[(i, j, k)] * M[j][k])

        opt_preps[i] = opt_state(operators_list, d)

    return opt_preps


def opt_state(operators_list, d):

    """
    Optimal state
    -------------

    Complementary function for `find_opt_prep` function. In this function, operators_list represents
    an ordered list of measurement operators of different measurements. The first element corres-
    ponds to a certain measurement operator of the first measurement and so on.

    For obtaining the optimal preparation for a certain combination of measurement operators, I am
    computing the eigenvector related to the largest eigenvalue of "average success probability".
    From this eigenvector, I generate the optimal state.

    Inputs
    ------
    operators_list: a list with n matrices of size d x d.
    d: an integer.
        d represents the dimension of the measurement operators.

    Output
    ------
    rho: an array of size d x d.
        The optimal state preparation for the combination of measurement operators in operators_
        list.
    """

    # The "average success probability" is defined as simply the sum of the elements in operators_
    # list.
    sum = np.sum(operators_list, axis=0)

    # numpy.eigh provides the eigenvalues and eigenvectors ordered from the smallest to the largest.
    eigval, eigvet = nalg.eigh(sum)

    # Just selecting the eigenvector related to the latest (also largest) eigenvalue.
    rho = np.outer(eigvet[:, d - 1], eigvet[:, d - 1].conj())

    return rho


def find_opt_meas(opt_preps, d, n, m, bias, mosek_accuracy):

    """
    Find optimal measurements
    -------------------------

    This function acts jointly with `opt_meas` function. It solves a single step of the see-saw al-
    gorithm by optimizing all of the measurements of Bob independently. It receives as input a dic-
    tionary containing the optimal preparations for all combinations of measurement operators of
    distinct measurements of Bob.

    The optimizations of all measurements can be made independently. To optimize over the i-th mea-
    surement we sum over all indexes of the dictionary opt_preps. Remember that the keys of opt_p
    reps are n-tuples. Then, a list opt_preps_sum of sums is provided to `opt_meas` function.

    Inputs
    ------
    opt_preps: a dictionary in which the keys are given by an n-tuple and the content of each entry
    is a matrix of size d x d.
    d: an integer.
        d represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements.
    m: an integer.
        m represents the number of outcomes of each measurement.
    bias: a dictionary of size n * m ** n. bias.keys() are tuples of n + 2 coordinates. bias.value
    s() are floats.
        The dictionary bias represents a order-(n+2) tensor encoding the bias in some given QRAC.
    mosek_accuracy: a float.
        Feasibility tolerance used by the interior-point optimizer for conic problems in the solver
        MOSEK. Here it is used for the primal and the dual problem.

    Output
    ------
    prob sum: a float
        It contains the value of the optimization of the "average success probability" after a sin-
        gle step of the see-saw algorithm.
    M: a list of lists whose elements are matrices of size d x d.
        M is the same list as in the function `find_opt_prep` after a single step of the see-saw al-
        gorithm.
    """

    # Defining empty variables.
    M = []
    prob_sum = 0

    # Each step of this loop stands for a different measurement.
    for j in range(0, n):

        # Initializing an empty list.
        opt_preps_sum = []

        for k in range(0, m):

            indexes = product(range(0, m), repeat=n)

            # Summing through all indexes of the variable "indexes". This sum is weighted by the
            # correct bias element, bias[(i, j, k)].
            sum = np.sum([bias[(i, j, k)] * opt_preps[i] for i in indexes], axis=0)

            # Appending m sums to opt_preps_sum
            opt_preps_sum.append(sum)

        # Solving the problem for the j-th measurement
        prob_value, meas_value = opt_meas(opt_preps_sum, d, n, m, mosek_accuracy)
        prob_sum += prob_value
        M.append(meas_value)

    return prob_sum, M


def opt_meas(opt_preps_sum, d, n, m, mosek_accuracy):

    """
    Optimal measurement
    -------------------

    Complementary function for `find_opt_meas` function. This function takes as input one of Bob's
    measurements M[i] and a collection of optimal preparations opt_preps with respect to M.

    The structure of this function is a simple SDP optimization for the objective function prob_
    guess.

    Inputs
    ------
    opt_preps_sum: list of d matrices of size d x d.
        This list contains all of the preparations of Alice summed over all indexes but the i-th.
    d: an integer.
        d represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements. I only need n here to calculate the factor
        multiplying prob_guess.
    m: an integer.
        m represents the number of outcomes of the measurements.
    mosek_accuracy: a float.
        Feasibility tolerance used by the interior-point optimizer for conic problems in the solver
        MOSEK. Here it is used for the primal and the dual problem.


    Outputs
    -------
    prob.value: A float.
        Numerical solution of prob via CVXPY.
    M_vars: a list of m matrices of size d x d.
        The optimized measurement M[i] after a single iteration of the see-saw algorithm.
    """

    # Creating a list of measurements. Each element of this list will be a CVXPY variable.
    M_vars = []

    # Appending the variables to the list:
    for i in range(0, m):
        M_vars.append(cp.Variable((d, d), hermitian=True))

    # Defining objective function:
    prob_guess = 0
    for i in range(0, m):
        prob_guess += cp.trace(M_vars[i] @ opt_preps_sum[i])

    # Defining constraints:
    constr = []
    for i in range(0, m):

        # M must be positive semi-definite.
        constr.append(M_vars[i] >> 0)

    # The elements of M_vars must sum to identity.
    sum = cp.sum(M_vars, axis=0)
    constr.append(sum == np.eye(d))

    # Defining the SDP and solving. CVXPY cannot recognize that the objective function is real.
    prob = cp.Problem(cp.Maximize(cp.real(prob_guess)), constr)

    # Sometimes MOSEK cannot solve a problem with high accuracy. However, if the seed is changed, a
    # solution is possible for some cases. So here we apply a random unitary to the preparations,
    # which is equivalent to producing new seeds. If the problem still remains unfeasilble, we raise
    # an error.
    attempts = 0
    while attempts < 20 and prob.value is None:
        attempts += 1
        try:
            prob.solve(
                solver="MOSEK",
                mosek_params={
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": mosek_accuracy,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": mosek_accuracy,
                },
            )
        except Exception:

            # Generating a d-dimensional random unitary with scipy.stats.unitary_group.
            rdm_U = unitary_group.rvs(d)
            prob_guess = 0

            for i in range(0, m):
                prob_guess += cp.trace(
                    M_vars[i] @ (rdm_U @ opt_preps_sum[i] @ rdm_U.conj().T)
                )

            # Reformulating the problem
            prob = cp.Problem(cp.Maximize(cp.real(prob_guess)), constr)

            pass

    if prob.value is None:
        raise RuntimeError("Solver 'MOSEK' failed. Try to reduce 'MOSEK' accuracy.")

    # Updating M_vars by its optimal value.
    for i in range(0, m):
        M_vars[i] = M_vars[i].value

    return prob.value, M_vars


# ==================================================================================================
# ---------------------------------------------- MAIN ----------------------------------------------
# ==================================================================================================
def find_QRAC_value(
    n: int,
    d: int,
    seeds: int,
    m: int = None,
    verbose: bool = True,
    prob_bound: float = PROB_BOUND,
    meas_bound: float = BOUND,
    mosek_accuracy: float = MOSEK_ACCURACY,
    bias: str = None,
    weight=None,
):
    """
    Find the Quantum Random Acces Code quantum value
    ------------------------------------------------

    Main function. Here I perform a see-saw optimization for the nˆd --> 1 QRAC. This function
    can be described in a few elementary steps, as follows:

        1. Create a set of n random measurements with m outcomes acting dimension d with generate_
        random_measurements.
        2. For this set of measurements, find the set optimal preparations using `find_opt_prep`.
        3. Optimize the measurements for the set of optimal preparations found in step 2 using `find
        _opt_meas`.
        4. Check if the variables prob_value and M are converging. If not, return to step 2 till the
        difference between previous_prob_value and prob_value is smaller than prob_bound. Also, the
        loop should terminate either when max_norm_difference is smaller than meas_bound or when the
        number of iterations is bigger than the constant ITERATIONS.

    This function also checks if the obtained optimal measurements are MUBs by activating determine_
    meas_status function.

    Inputs
    ------
    n: an integer.
        n represents the number of distinct measurements.
    d: an integer.
        The dimension of the measurement operators.
    seeds: an integer.
        Represents the number of random seeds as the starting points of the see-saw algorithm.
    m: an integer. [optional]
        The number of outcomes for the measurements. If no value is attributed to m, then m =
        d.
    verbose: True or False. True by default. [optional]
        If true, it activates the function `printings` that produces a report about the computation.
    prob_bound: a float. [optional]
        Convergence criterion for the variable prob_value. When the difference between prob_value
        and previous_prob_value is less than prob_bound, the algorithm interprets prob_value = previ
        ous_prob_value.
    meas_bound: a float. [optional]
        The same criterion as in prob_bound but for the norms of the measurement operators in the
        variable M.
    mosek_accuracy: a float. [optional]
        Feasibility tolerance used by the interior-point optimizer for conic problems in the solver
        MOSEK. Here it is used for the primal and the dual problem.
    bias: a dictionary of size n * m ** n. bias.keys() are tuples of n + 2 coordinates. bias.value
    s() are floats.
        The dictionary bias represents a order-(n+2) tensor encoding the bias in some given QRAC.
    weight: a float or a list of floats of size m.
        The variable weight carries the weight (or weights) with which the QRAC is biased. If it is
        a single float, it must be a positive number between 0 and 1. If it is a list of floats, it
        must sum to 1.
        Note that, for the case in which the variable weight is a single float, this variable is
        symmetrical. That is, setting weight = 0.5 has the same effect as making bias = None.

    Output
    ------
    report: a dictionary.
        report contains the following keys:
        n: an integer
            n represents the number of distinct measurements of the QRAC.
        dimension: an integer.
            Dimension of the measurement operators.
        seeds: an integer.
            The number of random seeds as the starting points of the see-saw algorithm.
        outcomes: an integer.
            The number of outcomes for the measurements.
        bias: a string.
            The kind of bias used in the QRAC. If bias = None, the QRAC is unbiased.
        weight: a float or a tuple/list of floats.
            The weight used in the bias. It can be a float in the case where bias is single-parame-
            ter, or a tuple/list in the case where bias is multi-parameter.
        optimal value: a float.
            The optimal value computed for the nˆm --> 1 QRAC.
        optimal measurements: a list of lists whose elements are matrices of size d x d.
            optimal measurements contains a nested list. Each fundamental list corresponds to a dif-
            ferent measurement in the QRAC task.
        best seed number: an integer.
            The seed number that achieves the largest prob_value among all seeds.
        seed convergence: a list of integers.
            It contains the number of the seeds whose measurements could converge to meas_bound.
        average time: a float.
            The average time of computation for among all seeds.
        average iterations: a float.
            The average number of iterations until convergence among all seeds.
        rank: a numpy array.
            rank contains the ranks of all measurement operators for the optimal realization.
        projectiveness: a dictionary.
            it contains two keys:
            projective: a numpy array with boolean elements.
                If True, the measurement operator is considered projective.
            errors: a numpy array whose elements are floats.
                It contains a measure of projectiveness for a given operator P. It can be calculated
                using the Frobenius norm of Pˆ2 - P.
        MUB check: a nested dicitonary.
            Its keys are tuples of integers (i, j), with i < j and i, j = 0, ..., m. Every element
            is a dictionary cointaing the variable status, which is obtained from the function check
            _if_MUBs.

    If verbose = True, report is printed using the funtion `printings`.
    """
    # If no value is attributed for the number of outcomes, then I set m = d.
    if m is None:
        m = d

    # Creating an empty dictionary report. If verbose = True, then we print the information con-
    # tained in the report. If not, the function return report.
    report = {}
    report["seeds"] = seeds
    report["n"] = n
    report["outcomes"] = m
    report["dimension"] = d
    report["bias"] = bias
    report["weight"] = weight

    # Starting the entries "optimal_value" and "optimal measurement" as zeros.
    report["optimal value"] = 0
    report["optimal measurements"] = 0

    # Here I am generating the bias and saving it in the tensor bias_tensor. By default, the QRAC
    # is unbiased, so the bias = None.
    bias_tensor = generate_bias(n, m, bias, weight)

    # List of times, and number of iterations for each seed. Also seed_convergence stores the infor-
    # mation of whether the measurements of a given seed have converged below meas_bound. These
    # lists will contain information to be printed in the final report.
    times_list = []
    iterations_list = []
    seed_convergence = []

    for i in range(0, seeds):

        # Marking the start of the time count.
        start_time = tp.process_time()

        # Generating a list of n random measurements.
        M = generate_random_measurements(n, d, m)

        # Defining the stopping conditions. The previous_M variable is started as the "null" measu-
        # rement, so to speak. It is just a dummy initialization.
        previous_prob_value = 0
        prob_value = 1
        previous_M = [
            [np.zeros((d, d), dtype=float) for i in range(0, m)] for j in range(0, n)
        ]
        max_norm_difference = 1
        iter_count = 0

        # The variable prob_value converges faster than M, so that the stopping condition for prob_
        # value can be smaller. For the measurements, we either stop when max_norm_difference is
        # smaller than meas_bound or when the number of iterations is bigger than ITERATIONS.
        while (abs(previous_prob_value - prob_value) > prob_bound) or (
            max_norm_difference > meas_bound and iter_count < ITERATIONS
        ):

            previous_prob_value = prob_value

            # The two lines below correspond to two a single round of see-saw.
            opt_preps = find_opt_prep(M, d, n, m, bias_tensor)
            prob_value, M = find_opt_meas(
                opt_preps, d, n, m, bias_tensor, mosek_accuracy
            )

            norm_difference = []
            for a in range(0, n):
                for b in range(0, m):
                    norm_difference.append(nalg.norm(M[a][b] - previous_M[a][b]))

            max_norm_difference = max(norm_difference)
            previous_M = M
            iter_count += 1

        # Saving the information on whether, for a given seed, there was convergence of measurements
        # or not.
        if iter_count < ITERATIONS:
            seed_convergence.append(i + 1)

        # Selecting the largest problem value from all distinct random seeds.
        if prob_value > report["optimal value"]:
            report["optimal value"] = prob_value
            report["optimal measurements"] = M
            report["best seed number"] = i + 1

        # Just append the information of the computation time and the number of iterations.
        times_list.append(tp.process_time() - start_time)
        iterations_list.append(iter_count)

    # Saving data in the dictionary report to be printed at the ending of the computation.
    report["seed convergence"] = seed_convergence
    report["average time"] = sum(times_list) / seeds
    report["average iterations"] = sum(iterations_list) / seeds

    (
        report["rank"],
        report["projectiveness"],
        report["MUB check"],
    ) = determine_meas_status(report["optimal measurements"], d, n, m)
    # verbose is True by default. If True, it prints a report of the computation. If False, it re-
    # turns the dictionary report.
    if verbose:
        printings(report)
    else:
        return report


def printings(report):

    """
    Printings
    ---------

    This function simply print a report of the computation if verbose = True. If consists of two
    parts: a summary of the computation and an analysis of the optimal realization found by the see-
    saw algorithm.

    Inputs
    ------
    report: a dictionary.
        report contains the following keys:
        n: an integer
            n represents the number of distinct measurements of the QRAC.
        dimension: an integer.
            Dimension of the measurement operators.
        seeds: an integer.
            The number of random seeds as the starting points of the see-saw algorithm.
        outcomes: an integer.
            The number of outcomes for the measurements.
        bias: a string.
            The kind of bias used in the QRAC. If bias = None, the QRAC is unbiased.
        weight: a float or a tuple/list of floats.
            The weight used in the bias. It can be a float in the case where bias is single-parame-
            ter, or a tuple/list in the case where bias is multi-parameter.
        optimal value: a float.
            The optimal value computed for the nˆm --> 1 QRAC.
        optimal measurements: a list of lists whose elements are matrices of size d x d.
            optimal measurements contains a nested list. Each fundamental list corresponds to a dif-
            ferent measurement in the QRAC task.
        best seed number: an integer.
            The seed number that achieves the largest prob_value among all seeds.
        seed convergence: a list of integers.
            It contains the number of the seeds whose measurements could converge to meas_bound.
        average time: a float.
            The average time of computation for among all seeds.
        average iterations: a float.
            The average number of iterations until convergence among all seeds.
        rank: a numpy array.
            rank contains the ranks of all measurement operators for the optimal realization.
        projectiveness: a dictionary.
            it contains two keys:
            projective: a numpy array with boolean elements.
                If True, the measurement operator is considered projective.
            errors: a numpy array whose elements are floats.
                It contains a measure of projectiveness for a given operator P. It can be calculated
                using the Frobenius norm of Pˆ2 - P.
        MUB check: a nested dicitonary.
            Its keys are tuples of integers (i, j), with i < j and i, j = 0, ..., m. Every element
            is a dictionary cointaing the variable status, which is obtained from the function check
            _if_MUBs.
    """

    # This command is to allow printing superscripts in the prompt.
    superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    # Printing header
    print(
        f"\n" + f"=" * 80 + f"\n" + f" " * 32 + f"QRAC-tools v1.0\n" + f"=" * 80 + f"\n"
    )

    # Printing the first part of the report
    print(
        colors.BOLD
        + colors.FAINT
        + f"-" * 28
        + f" Summary of computation "
        + f"-" * 28
        + colors.END
        + f"\n"
    )

    if report["seed convergence"] == []:
        report["seed convergence"].append(None)

    print(
        f"Number of random seeds: {report['seeds']}\n"
        f"Best seed: {report['best seed number']} \n"
        f"Seeds whose measurements converged below meas_bound: "
        f"{', '.join(str(i) for i in report['seed convergence'])}\n"
        f"Average time until convergence: {round(report['average time'], 5)} s\n"
        f"Average number of iterations until convergence: "
        f"{round(report['average iterations'])}\n"
    )

    # Printing the second part of the report. Analysis of the optimal realisation.
    print(
        colors.BOLD
        + colors.FAINT
        + f"-" * 21
        + f" Analysis of the optimal realisation "
        + f"-" * 22
        + colors.END
        + f"\n"
    )

    if report["bias"] is not None:
        print(
            f"Kind of bias: {report['bias']}\n"
            + f"Weight: {round(report['weight'], 5)}"
            if isinstance(report["weight"], (float, int))
            else f"Kind of bias: {report['bias']}\n" + f"Weights: {report['weight']}"
        )

    print(
        f"Optimal value for the "
        f"{report['n']}{str(report['outcomes']).translate(superscript)}-->1 QRAC:"
        f" {report['optimal value'].round(12)}"
        f"\n"
    )

    # Printing the ranks of the measurement operators.
    print(colors.CYAN + f"Measurement operator ranks" + colors.END)
    line = 0
    for i in report["rank"]:
        print(f"M[{str(line)}] ranks: ", f"  ".join(map(str, i)))
        line += 1

    print("")

    # Printing whether the measurement operators are projective or not.
    print(colors.CYAN + f"Measurement operator projectiveness" + colors.END)

    projective = report["projectiveness"]["projective"]
    errors = report["projectiveness"]["errors"]

    for i in range(0, report["n"]):
        for j in range(0, report["outcomes"]):
            print(
                f"M[{str(i)}, {str(j)}]:  Projective\t\t"
                f"{str(float('%.3g' % errors[i][j]))}"
                if projective[i][j]
                else f"M[{str(i)}, {str(j)}]:  Not projective\t"
                f"{str(float('%.3g' % errors[i][j]))}"
            )

    if report["MUB check"] is not None:
        keys = list(report["MUB check"].keys())

        print(" ")
        print(colors.CYAN + f"Mutually unbiasedness of measurements" + colors.END)

        for i in keys:
            print(
                f"M[{str(i[0])}] and M[{str(i[1])}]:  MUM\t\t"
                f"{str(float('%.3g' % report['MUB check'][i]['max error']))}"
                if report["MUB check"][i]["boolean"]
                else f"M[{str(i[0])}] and M[{str(i[1])}]:  Not MUM\t\t"
                f"{str(float('%.3g' % report['MUB check'][i]['max error']))}"
            )

    # Printing the footer of the report.
    print(" ")
    print(f"-" * 30 + f" End of computation " + f"-" * 30)


def determine_meas_status(M, d, n, m):

    """
    Determine measurement status
    ----------------------------

    This function simply checks the status of the optimized measurements. It checks whether the mea-
    surement operators are Hermitian, positive semi-definite, rank-one, projective and if they sum
    to identity, not necessarily in this order. To finish it checks if all of the pairs of measure-
    ments are constructed out of mutually unbiased bases using the function `check_if_MUBs`.

    Inputs
    ------
    M: a list of lists whose elements are matrices of size d x d.
        M is a list containing n lists. Each list inside M corresponds to a different measurement in
        the QRAC task. For each measurement there are m measurement operators.
    d: an integer.
        d represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements.
    m: an integer.
        m represents the number of outcomes of the measurements.

    Output
    ------
    rank: a numpy array.
        rank contains the ranks of all measurement operators for the optimal realization.
    projectiveness: a dictionary.
        it contains two keys:
        projective: a numpy array with boolean elements.
            If True, the measurement operator is considered projective.
        errors: a numpy array whose elements are floats.
            It contains a measure of projectiveness for a given operator P. It can be calculated
            using the Frobenius norm of Pˆ2 - P.
    MUB check: a nested dicitonary.
        Its keys are tuples of integers (i, j), with i < j and i, j = 0, ..., m. Every element is
        a dictionary cointaing the variable status, which is obtained from the function check_if_
        MUBs.
    """

    # Flag for the MUB checking. If the measurement operators are neither rank-one or projective,
    # there is no sense on checking if they can be constructed out of MUBs.
    flag = True

    # Checking if the measurement operators are Hermitian.
    for i in range(0, n):
        for j in range(0, m):
            if nalg.norm(M[i][j] - M[i][j].conj().T) > BOUND:
                raise RuntimeError("measurement operators are not Hermitian")

    # Checking if the measurement operators are positive semi-definite.
    for i in range(0, n):
        for j in range(0, m):
            eigval, eigvet = nalg.eigh(M[i][j])
            if eigval[0] < -BOUND:
                raise RuntimeError(
                    "measurement operators are not positive semi-definite"
                )

    # Checking if the measurement operators sum to identity.
    for i in range(0, n):
        sum = np.sum(M[i], axis=0)
        if nalg.norm(sum - np.eye(d)) > BOUND:
            raise RuntimeError("measurement operators does not sum to identity")

    # This will return me an array with the ranks of all measurement operators.
    rank = nalg.matrix_rank(M, tol=BOUND, hermitian=True)
    # Now checking if all of the measurement operators are rank-one.
    if not (rank == np.ones((n, m), dtype=np.int8)).all():
        flag = False

    # Checking if the measurement operators are projective.
    projective = np.empty((n, m))
    errors = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):

            errors[i][j] = nalg.norm(M[i][j] @ M[i][j] - M[i][j])

            if errors[i][j] > BOUND:
                projective[i][j] = False
                flag = False
            else:
                projective[i][j] = True

    # The dictionary projectiveness is returned. It contains the information on whether some mea-
    # surement operator is projective or not and a measure, which is simply the Frobenius norm of
    # Mˆ2 - M.
    projectiveness = {"projective": projective, "errors": errors}

    # Checking if the measurements are constructed out of Mutually Unbiased Bases. The dicitonary
    # MUB_check is returned cointaing the desired information. If flag is False, MUB_check is an
    # empty variable.
    if flag:
        MUB_check = {}
        for i in range(0, n):
            for j in range(i + 1, n):
                MUB_check[(i, j)] = check_if_MUBs(M[i], M[j], m)
    else:
        MUB_check = None

    return rank, projectiveness, MUB_check


def check_if_MUBs(P, Q, m, mub_bound=MUB_BOUND):

    """
    Check if two measurements are mutually unbiased
    -----------------------------------------------

    This function works jointly with `determine_meas_status` function. It simply gets two d-dimen-
    sional measurements P and Q, and checks if they are constructed out of Mutually Unbiased Bases.
    Check appendix II of the supplementary material of the reference for details.

    Inputs
    ------
    P: a list with m matrices of size d x d.
        A measurement with m outcomes acting on dimension d.
    Q: a list with m matrices of size d x d.
        Another measurement with m outcomes acting on dimension d.
    m: an integer.
        m represents the number of outcomes of the measurements.
    mub_bound: a float.
        If PQP-P < mub_bound, then the equation PQP = P is considered satisfied and the measurements
        P and Q are considered mutually unbiased.

    Output
    ------
    status: a dictionary with two entries. A boolean variable and a float.
        The first key of this dictionary is named boolean. If True, P and Q are considered mutually
        unbiased up to the numerical precision of mub_bound. The second key is named max_error and
        contains the maximum norm difference by which the measurement operators of P and Q do not
        satisfy the equations mPQP = P and mQPQ = Q.

    Reference
    ---------
    1. A. Tavakoli, M. Farkas, D. Rosset, J.-D. Bancal, J. Kaniewski, Mutually unbiased bases and
    symmetric informationally complete measurements in Bell experiments, Sci. Adv., vol. 7, issue 7.
    DOI: 10.1126/sciadv.abc3847.
    """
    status = {"boolean": True}

    errors = []
    for i in range(0, m):
        for j in range(0, m):

            errorP = nalg.norm(m * P[i] @ Q[j] @ P[i] - P[i])
            errorQ = nalg.norm(m * Q[j] @ P[i] @ Q[j] - Q[j])

            errors.append(errorP)
            errors.append(errorQ)

            if errorP > mub_bound or errorQ > mub_bound:
                status["boolean"] = False

    status["max error"] = max(errors)

    return status


def generate_bias(n, m, bias, weight):

    """
    Generate bias
    -------------

    This function generates a dictionary in which the keys are tuples of the form

    i = ((x_1, x_2, x_3, ... , x_n), y, b)

    where x_1, x_2, x_3, ... , x_n and b range from 0 to m - 1 and y ranges from 0 to n - 1. The
    object bias tensor[i] represents a order-(n+2) tensor encoding the bias in some given QRAC.

    If bias = None, the QRAC is unbiased and all of the n * m ** n elements in bias_tensor are
    uniform. The elements of bias_tensor are strictly bigger than zero whenever i[2] == i[0][i[1]].

    There are eight possible kinds of bias, as follows.
    1. "XDIAG". Bias in the input string x. The elements of the main diagonal of the order-n tensor
    i[0] are prefered with probability `weight`. For the n = 2 QRAC, it translates as

    x_1/x_2 | 0  1  2 ... m-1
    0       | *  .  .       .
    1       | .  *  .       .
    2       | .  .  *       .
    :       |
    m-1   | .  .  .       *

    2. "XCHESS". Bias in the input string x. The elements of the order-n tensor i[0] are prefered
    with probability `weight`, in an array that resembles a chess table. If n = 2,

    x_1/x_2 | 0  1  2 ... m-1
    0       | .  *  .       *
    1       | *  .  *       .
    2       | .  *  .       *
    :       |
    m-1   | *  .  *       .

    3. "XPARAM". Bias in the input string x. The element (0, 0, ..., 0) of the order-n tensor i[0]
    is prefered probability `weight`.

    4. "XPLANE". Bias in the input string x. The "plane" x_1 = 0 in the order-n tensor i[0] is pre-
    frered with probability `weight`.

    5. "YPARAM". Bias in the requested digit y. The element 0 of i[1] is prefered with probability
    `weight`.

    6. "YVEC". Bias in the requested digit y. The y-th element of i[1] is prefered with probability
    weight[y].

    7. "BPARAM". Bias in the retrieved output b. The element 0 of i[2] is prefered with probability
    `weight`.

    8. "BVEC". Bias in the retrieved output b. The b-th element of i[2] is prefered with probability
    weight[b].

    Inputs
    ------
    n: an integer.
        n represents the number of distinct measurements.
    m: an integer.
        m represents the number of outcomes of the measurements.
    bias: a string or an empty variable.
        It encodes the type of bias desired for the computation. There are eight possibilities: "XDI
        AG", "XCHESS", "XPARAM", "XPLANE", "YPARAM", "YVEC", "BPARAM" and "BVEC". If bias = None,
        the QRAC is unbiased.
    weight: a float or a list of floats of size m.
        The variable weight carries the weight (or weights) with which the QRAC is biased. If it
        is a single float, it must be a positive number between 0 and 1. If it is a list of floats,
        it must sum to 1.
        Note that, for the case in which the variable weight is a single float, this variable is
        symmetrical. That is, setting weight = 0.5 has the same effect as making bias = None.

    Output
    ------
    bias_tensor: a dictionary of size n * m ** n. bias_tensor.keys() are tuples of n + 2 coordina-
    tes. bias_tensor.values() are floats.
        bias_tensor represents a order-(n+2) tensor encoding the bias in some given QRAC.
    """

    # This first part of the funtion is just for assertions.

    # sw/mw stands for single/multiple weights.
    valid_bias_types_sw = ("XDIAG", "XCHESS", "XPARAM", "XPLANE", "YPARAM", "BPARAM")
    valid_bias_types_mw = ("YVEC", "BVEC")

    assert bias is None or bias in valid_bias_types_sw or bias in valid_bias_types_mw, (
        "Available options for bias are: "
        "XDIAG, XCHESS, XPARAM, XPLANE, YPARAM, YVEC, BPARAM and BVEC."
    )

    if bias is not None:
        assert weight is not None, "a value for `weight` must be provided"

    if bias in valid_bias_types_sw:
        assert 0 <= weight <= 1, "`weight` must range between 0 and 1."

    elif bias in valid_bias_types_mw:

        assert isinstance(
            weight, (list, tuple)
        ), "For YVEC and BVEC bias, `weight` must be a list or a tuple"

        if bias == "YVEC":

            assert len(weight) == n, "the expected size of `weight` is n"
            assert round(sum(weight), 7) == 1, "the weights must sum to one"

        elif bias == "BVEC":

            assert len(weight) == m, "the expected size of `weight` is m"
            assert round(sum(weight), 7) == 1, "the weights must sum to one"

    # Now, the code actually starts.

    # Initializing an empty dicitonary.
    bias_tensor = {}

    # Creating an iterable for feeding bias_tensor. The keys of bias_tensor will be the tuples defi-
    # ned by the iterable indexes.
    indexes = product(product(range(0, m), repeat=n), range(0, n), range(0, m))

    for i in indexes:

        # Enforcing the QRAC condition.
        if i[2] == i[0][i[1]]:

            # The elements must be uniform. There n * m ** n elements in bias_tensor in total.
            bias_tensor[i] = 1 / (n * m**n)

            # Separating in cases. If bias = None, none of the below cases will match, and the re-
            # sulting bias_tensor will be unbiased.
            if bias is None:
                continue

            elif bias == "XDIAG":

                # If i[0] is a diagonal element, then it is prefered with `weight`. If not, it
                # is prefered with 1 - weight. The other cases follow similarly.
                if len(set(i[0])) == 1:

                    # The constants multiplying and dividing bias_tensor[i] are an effect of the re-
                    # normalization due to the introduction of bias.
                    bias_tensor[i] = (m**n) * weight * bias_tensor[i] / m
                else:
                    bias_tensor[i] = (
                        (m**n) * (1 - weight) * bias_tensor[i] / (m**n - m)
                    )

            elif bias == "XCHESS":

                # Here the normalization depends on the parity of m ** n. Recall that parity(m) ==
                # parity(m ** n), for positive n and m.
                if m % 2 == 0:
                    if sum(i[0]) % 2 == 1:
                        bias_tensor[i] = 2 * weight * bias_tensor[i]
                    else:
                        bias_tensor[i] = 2 * (1 - weight) * bias_tensor[i]
                else:
                    if sum(i[0]) % 2 == 1:
                        bias_tensor[i] = (
                            2 * (m**n) * weight * bias_tensor[i] / (m**n - 1)
                        )
                    else:
                        bias_tensor[i] = (
                            2 * (m**n) * (1 - weight) * bias_tensor[i] / (m**n + 1)
                        )

            elif bias == "XPARAM":

                if i[0] == (0,) * n:
                    bias_tensor[i] = (m**n) * weight * bias_tensor[i]
                else:
                    bias_tensor[i] = (
                        (m**n) * (1 - weight) * bias_tensor[i] / (m**n - 1)
                    )

            elif bias == "XPLANE":

                if i[0][0] == 0:
                    bias_tensor[i] = m * weight * bias_tensor[i]
                else:
                    bias_tensor[i] = m * (1 - weight) * bias_tensor[i] / (m - 1)

            elif bias == "YPARAM":

                if i[1] == 0:
                    bias_tensor[i] = n * weight * bias_tensor[i]
                else:
                    bias_tensor[i] = n * (1 - weight) * bias_tensor[i] / (n - 1)

            elif bias == "BPARAM":

                if i[2] == 0:
                    bias_tensor[i] = m * weight * bias_tensor[i]
                else:
                    bias_tensor[i] = m * (1 - weight) * bias_tensor[i] / (m - 1)

            elif bias == "YVEC":

                bias_tensor[i] = n * weight[i[1]] * bias_tensor[i]

            elif bias == "BVEC":

                bias_tensor[i] = m * weight[i[2]] * bias_tensor[i]

        else:
            bias_tensor[i] = 0

    return bias_tensor


def find_classical_value(
    n: int, d: int, m: int = None, verbose=True, bias: str = None, weight=None
):

    """
    Find the Quantum Random Acces Code classical value
    --------------------------------------------------

    This function is the classical analogous of `find_QRAC_value`. Unlike `find_QRAC_value`, this
    function finds the optimal value of a RAC (Random Acess Code) by optimizing over all encoding
    and decoding strategies.

    In a RAC, one desires to encode n digits ranging from 0 to m - 1 in another digit ranging from 0
    to d - 1. In total, there are d^(m^n) encoding strategies and m^(d*n) decoding strategies, so
    this function scales double exponentially. It performs a simple maximization over all combina-
    tions of encoding and decoding strategies.

    Cases you can expect to compute in less than one hour, for tuples of (n, d, m):
    (2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 3, 2), (2, 3, 3), (2, 4, 2), (2, 5, 2), (2, 6, 2),
    (2, 7, 2), (2, 8, 2), (3, 2, 2), (3, 3, 2), (3, 4, 2) and (4, 2, 2).

    Cases (2, 4, 2), (2, 5, 2), (2, 6, 2), (2, 7, 2) are (2, 8, 2) equivalent and trivial (return 1
    as the classical value).

    Inputs
    ------
    n: an integer.
        n represents the number of encoded digits.
    d: an integer.
        d represents the size of the message to be passed to Bob.
    m: an integer. [optional]
        m represents the size of the digits of Alice. If Alice has n digits, then these each of the-
        se digits range from 0 to m - 1. By default, m = d.
    verbose: True or False. True by default. [optional]
        If true, it prints a small report cointaing the classical value if the proposed RAC. If fal-
        se, it returns the classical value.
    bias: a dictionary of size n * m ** n. bias.keys() are tuples of n + 2 coordinates. bias.value
    s() are floats.
        The dictionary bias represents a order-(n+2) tensor encoding the bias in some given RAC.
    weight: a float or a list of floats of size m.
        The variable weight carries the weight (or weights) with which the RAC is biased. If it is a
        single float, it must be a positive number between 0 and 1. If it is a list of floats, it
        must sum to 1.
        Note that, for the case in which the variable weight is a single float, this variable is
        symmetrical. That is, setting weight = 0.5 has the same effect as making bias = None.

    Outputs
    -------
    If verbose = True, it prints a small report containing the classical value for the given RAC. If
    not, it returns

    classical_probability: a float.
        The classical value for the n^m --> 1 RAC.
    """

    if verbose:
        # Printing header
        print(
            f"\n"
            + f"=" * 80
            + f"\n"
            + f" " * 32
            + f"QRAC-tools v1.0\n"
            + f"=" * 80
            + f"\n"
        )

    # If no value is attributed for the number of outcomes, then I set m = d.
    if m is None:
        m = d

    # Enumerating all possible strategies. The first line represents all possible encodings while
    # the second line represents all possible decodings.
    strategies = product(
        product(range(d), repeat=m**n), product(range(m), repeat=d * n)
    )

    # The variable iterable is an auxiliary variable. It will be transformed into the list `indexes`
    # after.
    iterable = product(product(range(m), repeat=n), range(n))
    indexes = []

    # The variable iterable if equivalent to the tuples in `generate_bias`, except that it does not
    # contain an entry for b. This loop converts the tuple (x_1, x_2, ..., x_n) into a decimal num-
    # ber and saves in the last entry of `indexes`.
    for i in iterable:
        decimal = 0
        N = n - 1
        for j in i[0]:
            decimal += j * m**N
            N -= 1
        indexes.append((i, decimal))
    indexes = [((a), b, c) for ((a), b), c in indexes]

    bias_tensor = generate_bias(n, m, bias, weight)

    # Starting the algorithm by creating a initializing the empty list of classical_probability.
    classical_probability = []
    start = tp.time()

    for i in strategies:

        strategy_prob = 0
        for j in indexes:

            # Selecting what messaging digit will be send to Bob.
            message = i[0][j[2]]

            # Based on the received digit, Bob produces output b.
            b = i[1][message * n + j[1]]

            # This is simply the RAC condition.
            if b == j[0][j[1]]:

                # This is a deterministic strategy. If the RAC condition is satisfied, then probabi-
                # lity == 1 * bias_tensor.
                strategy_prob += bias_tensor[(j[0], j[1], b)]

        classical_probability.append(strategy_prob)

    total_time = tp.time() - start

    # Optimizing over all strategies
    classical_probability = max(classical_probability)

    # Printing the report
    if verbose:

        print(
            colors.BOLD
            + colors.FAINT
            + f"-" * 28
            + f" Summary of computation "
            + f"-" * 28
            + colors.END
            + f"\n"
        )

        print(f"Total time of computation: {round(total_time, 5)} s")

        if bias is not None:
            print(
                f"Kind of bias: {bias}\n" + f"Weight: {round(weight, 5)}"
                if isinstance(weight, (float, int))
                else f"Kind of bias: {bias}\n" + f"Weights: {weight}"
            )

        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        print(
            f"Optimal value for the "
            f"{n}{str(m).translate(superscript)}-->1 RAC:"
            f" {round(classical_probability, 10)}"
            f"\n"
        )

        # Printing the footer of the report.
        print(f"-" * 30 + f" End of computation " + f"-" * 30)

    else:
        return classical_probability
