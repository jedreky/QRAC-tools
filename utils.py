#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
# Created by  : Gabriel Pereira Alves, Jedrzej Kaniewski and Nicolas Gigena
# Created Date: March 16, 2022
# e-mail: gpereira@fuw.edu.pl
# version = '1.0'
# ==================================================================================================
"""This file contains the main functions that produce an estimation to the quantum and classical va-
lues of a QRAC."""
# ==================================================================================================
# Imports: time, cvxpy, numpy, scipy, itertools and warnings.
# ==================================================================================================

import time as tp
import cvxpy as cp
import numpy as np
import scipy as sp
import constants as const
import warnings


import numpy.linalg as nalg

from numpy.random import rand
from numpy.random import random
from scipy.linalg import sqrtm
from scipy.stats import unitary_group
from itertools import product


class colors:
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    END = "\033[0m"
    RED = "\033[91m"


def generate_random_measurements(n, d, m, diagonal):

    """
    Generate random measurements
    ----------------------------

    This function generates n random measurements with m outcomes and dimension d.

    First, it generates m complex random operators called random_op. It then transforms random_op
    into a positive semidefinite and hermitian matrix. Finally, it rescales all random_op operators
    to guarantee they sum to identity.

    If diagonal = "classical", the same procedure is followed, but with diagonal matrices. This gua-
    rantees that all measurements are diagonal in the same basis.

    Input
    -----
    n: an integer.
        The number of distinct measurements.
    d: an integer.
        The dimension of the measurement operators.
    m: an integer.
        The number of outcomes for each measurement.
    diagonal: a boolean.
        If True, generate_random_measurements produces diagonal random measurements. If false, the
        random measurements correspond to POVMs.

    Output
    ------
    M: a list of n lists. Each sub-list of M contains m matrices of size d x d.
        M represents a list with n d-dimensional m-outcome random measurements.
    """

    # Initializing an empty list of measurements.
    measurements_list = []
    for i in range(0, n):

        # This code tries to generate a measurement 10 times. If, in a certain iteration, a suitable
        # measurement is not produced, it passes to the next iteration.
        attempts = 0
        while attempts < const.MEAS_ATTEMPTS:
            attempts += 1

            # Initializing an empty list of partial measurements.
            partial_meas = []

            if diagonal:

                # Creating m random diagonal matrices and appending them to the list partial_meas.
                for j in range(0, m):

                    diagonal_matrix = np.zeros((d, d), dtype=float)

                    for k in range(0, d):

                        # random() generates a random number in the interval (0, 1].
                        diagonal_matrix[k, k] = random()

                    # By construction, diagonal_matrixthey is positive-semidefinite and hermitian.
                    partial_meas.append(diagonal_matrix)

            else:

                # Creating m random complex operators and appending them to the list partial_meas.
                for j in range(0, m):

                    # Generating random complex operators. Each entry should have both real and i-
                    # maginary parts contained in the interval (-1, +1].
                    random_op = (
                        2 * rand(d, d)
                        - np.ones((d, d))
                        + 1j * (2 * rand(d, d) - np.ones((d, d)))
                    )

                    # Transforming random_op into a hermitian operator. Note that random_herm is al-
                    # so positive semidefinite.
                    random_herm = random_op.T.conj() @ random_op

                    partial_meas.append(random_herm)

            partial_sum = np.sum(partial_meas, axis=0)

            # Checking if partial_sum is full-rank
            full_rank = nalg.matrix_rank(partial_sum, tol=const.BOUND, hermitian=True) == d

            if full_rank:

                # Initializing an empty list to append a single measurement.
                measurement = []

                # This operator is used to rescale the partial_meas list. It corresponds to the in-
                # verse square root of partial_sum.
                inv_sqrt = nalg.inv(sqrtm(partial_sum))

                # This loop rescales all operators in partial_meas.
                error = False
                for j in range(0, m):

                    M = inv_sqrt @ partial_meas[j] @ inv_sqrt

                    # Enforcing hermiticity.
                    M = 0.5 * (M + M.conj().T)

                    measurement.append(M)

                    # Checking if M is positive semidefinite. Numpy.linalg.eigh produces ordered
                    # eigenvalues. So it suffices to check if the first eigenvalue is non-negative.
                    # If it is not the case, the boolean variable error is changed to True and an-
                    # other attempt to generate a measurement will start.
                    eigval, eigvet = nalg.eigh(M)

                    if eigval[0] < -const.BOUND:
                        error = True

                if not error:

                    # Last check. Checking if the measurement operators sum to identity.
                    sum = np.sum(measurement, axis=0)
                    sum_to_identity = nalg.norm(sum - np.eye(d)) < const.BOUND

                    if sum_to_identity:
                        measurements_list.append(measurement)
                        break

        # The execution is finished if this function cannot produce a suitable measurement in 10
        # attempts.
        if attempts == 10:
            raise RuntimeError("a random measurement cannot be generated")

    return measurements_list


def find_opt_prep(n, d, m, M, bias):

    """
    Find the optimal preparations
    -----------------------------

    In the see-saw algorithm, there are three elementary steps. The first is to generate a seed and
    use it as a starting point for the algorithm. Second, it maximizes the state and third, the
    measurements. Steps two and three are iterated until the problem converges to a local maximum.

    This functions performs the second step, i.e., the maximization of the state. As for each input
    of Alice there is a distinct preparation, this function computes each one of these optimal pre-
    parations and outputs them in the dictionary optimal_preps.

    The optimization of each preparation is performed by the auxiliary function `optimize_preps`.

    Inputs
    ------
    n: an integer.
        The number of distinct measurements.
    d: an integer.
        The dimension of the measurement operators.
    m: an integer.
        The number of outcomes for each measurement.
    M: a list of n lists. Each sub-list of M contains m matrices of size d x d.
        M represents a list with n d-dimensional m-outcome measurements.
    bias: a dictionary of size n * m ** (n + 1). bias.keys() are (n + 2)-tuples. bias.values() are
    floats.
        The dictionary bias represents a order-(n + 2) tensor encoding the normalization and the
        bias in a given QRAC.

    Output
    ------
    optimal_preps: a dictionary of size m ** n. optimal_preps.keys() are n-tuples. optimal_preps.va-
    lues() are d x d matrices.
        optimal_preps is a dictionary containing the optimal preparations for a given set of meas-
        urements M. As for each input of Alice there is a distinct preparation, the keys of optimal_
        preps correspond to the input of Alice. Similarly, the values of optimal_preps correspond to
        the optimal state prepared when Alice receives the input in optimal_preps.keys().
    """

    # Creating an empty dictionary to accommodate Alice's optimal preparations.
    optimal_preps = {}

    # The variable alice_inputs list all possible tuples of size n with elements ranging from 0 to
    # m - 1. These tuples correspond to distinct inputs of Alice.
    alice_inputs = product(range(0, m), repeat=n)

    # Iterating over all Alice's inputs. Since Alice's inputs are a string of n characters, this for
    # is, in effect, a for nested n times.
    for i in alice_inputs:

        # Initializing and empty list. meas_operators_list cointains the measurement operators re-
        # lated to the i-th input of Alice.
        meas_operators_list = []

        for j in range(0, n):
            for k in range(0, m):

                # To compute the optimal preparation of a given input i of Alice, one can simply
                # calculate the pure state associated with the largest eigenvalue of the operator
                # sum(meas_operators_list). Each element of meas_operators_list corresponds to a k-th
                # measurement operator of the j-th measurement multiplied by the correct normaliza-
                # tion and bias (the (i, j, k) element of the dictionary bias).
                meas_operators_list.append(bias[(i, j, k)] * M[j][k])

        optimal_preps[i] = optimize_preps(meas_operators_list, d)

    return optimal_preps


def optimize_preps(operators_list, d):

    """
    Optimize preparations
    ---------------------

    Complementary function for `find_opt_prep`. It computes the the optimal preparation of a given
    input of Alice.

    To obtain the optimal state related to a certain preparation of Alice, one can simply calculate
    the pure state associated with the largest eigenvalue of the operator sum(meas_operators_list).

    Inputs
    ------
    meas_operators_list: a list with n matrices of size d x d.
    d: an integer.
        d represents the dimension of the measurement operators.

    Output
    ------
    optimal_prep: a d x d matrix.
        The optimal state preparation for a single input of Alice.
    """

    sum = np.sum(operators_list, axis=0)

    # numpy.linalg.eigh gives the eigenvalues in ascending order. The eigenvectors follow the order
    # of their respective eigenvalues.
    eigval, eigvet = nalg.eigh(sum)

    # Just selecting the eigenvector related to the largest eigenvalue.
    optimal_prep = np.outer(eigvet[:, d - 1], eigvet[:, d - 1].conj())

    return optimal_prep


def find_opt_meas(n, d, m, optimal_preps, bias, mosek_accuracy):

    """
    Find the optimal measurements
    -----------------------------

    This function performs the third step of the see-saw algorithm by optimizing all of the measure-
    ments of Bob.

    In a QRAC, the optimization can be made independently for each measurement. For this reason,
    this function sums the preparations over the inputs of Alice, so the resulting operator is inde-
    pendent of her inputs. This allows the optimization to be carried out independently for each
    measurement.

    The optimize_meas function is used in a complementary way. It performs an SDP to obtain a single
    optimal measurement and the corresponding problem value.

    Inputs
    ------
    n: an integer.
        The number of distinct measurements.
    d: an integer.
        The dimension of the measurement operators.
    m: an integer.
        The number of outcomes for each measurement.
    optimal_preps: a dictionary of size m ** n. optimal_preps.keys() are n-tuples. optimal_preps.va-
    lues() are d x d matrices.
        optimal_preps is a dictionary containing the optimal preparations for a given set of meas-
        urements M. As for each input of Alice there is a distinct preparation, the keys of optimal_
        preps correspond to the input of Alice. Similarly, the values of optimal_preps correspond to
        the optimal state prepared when Alice receives the input in optimal_preps.keys().
    bias: a dictionary of size n * m ** (n + 1). bias.keys() are (n + 2)-tuples. bias.values() are
    floats.
        The dictionary bias represents a order-(n + 2) tensor encoding the normalization and the
        bias in a given QRAC.
    mosek_accuracy: a float.
        Feasibility tolerance used by the interior-point optimizer for conic problems in the solver
        MOSEK. Here it is used for the primal and the dual problem. It is passed as a parameter to
        the optimize_meas function.

    Output
    ------
    problem_value: a float
        The optimized value of the QRAC functional for a single step of the see-saw algorithm.
    M: a list of n lists. Each sub-list of M contains m matrices of size d x d.
        M represents a list with n d-dimensional m-outcome measurements after optimization.
    """

    # Defining empty variables for the measurements and problem value.
    M = []
    problem_value = 0

    # Each step of this loop stands for a different measurement.
    for j in range(0, n):

        # Initializing an empty list of summed optimal preparations. Each element corresponds to the
        # summed preparations for a given input of Alice.
        opt_preps_sum = []

        for k in range(0, m):

            alice_inputs = product(range(0, m), repeat=n)

            # Summing through all Alice's inputs. This sum is weighted by the correct bias element,
            # bias[(i, j, k)].
            sum = np.sum(
                [bias[(i, j, k)] * optimal_preps[i] for i in alice_inputs], axis=0
            )

            # Appending m sums to opt_preps_sum
            opt_preps_sum.append(sum)

        # Solving the problem for the j-th measurement
        partial_prob_value, measurement_value = optimize_meas(
            d, m, opt_preps_sum, mosek_accuracy
        )
        problem_value += partial_prob_value
        M.append(measurement_value)

    return problem_value, M


def optimize_meas(d, m, opt_preps_sum, mosek_accuracy):

    """
    Optimize measurement
    --------------------

    Complementary function for `find_opt_meas`. This function is a simple SDP program for the objec-
    tive function guessing_probability. It relies on the solver MOSEK as implemented by the python
    package CVXPY.

    Inputs
    ------
    d: an integer.
        The dimension of the measurement operators.
    m: an integer.
        The number of outcomes for each measurement.
    opt_preps_sum: a list of m matrices of size d x d.
        Each element corresponds to the summed optimal preparations for a given input of Alice.
    mosek_accuracy: a float.
        Feasibility tolerance used by the interior-point optimizer for conic problems in the solver
        MOSEK. Here it is used for the primal and the dual problem.

    Outputs
    -------
    prob.value: A float.
        Numerical solution of `prob` via MOSEK.
    M_vars: a list of m matrices of size d x d.
        M_vars represents a d-dimensional m-outcome measurement after optimization.
    """

    # Creating a list of measurements. Each element of this list will be a CVXPY variable.
    M_vars = []

    # Appending the variables to the list.
    for i in range(0, m):
        M_vars.append(cp.Variable((d, d), hermitian=True))

    # Defining the objective function.
    guessing_probability = 0
    for i in range(0, m):
        guessing_probability += cp.trace(M_vars[i] @ opt_preps_sum[i])

    # Defining the constraints to the problem.
    constr = []
    for i in range(0, m):

        # The measurement operators must be positive semi-definite.
        constr.append(M_vars[i] >> 0)

    # The elements of M_vars must sum to identity.
    sum = cp.sum(M_vars, axis=0)
    constr.append(sum == np.eye(d))

    # Defining the SDP problem. CVXPY cannot recognize that the objective function is real, so the
    # function cvxpy.real() is used.
    prob = cp.Problem(cp.Maximize(cp.real(guessing_probability)), constr)

    # Sometimes MOSEK cannot solve a problem with high accuracy. However, if the seed is changed, a
    # solution is possible in some cases. Here a random unitary is applied to the preparations,
    # which is equivalent to producing new seeds. If the problem still remains unfeasible, we raise
    # an error.
    attempts = 0
    while attempts < const.SOLVE_ATTEMPTS and prob.value is None:
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

            # Reformulating the problem.
            prob = cp.Problem(cp.Maximize(cp.real(prob_guess)), constr)

            pass

    if prob.value is None:
        raise RuntimeError("Solver 'MOSEK' failed. Try to reduce 'MOSEK' accuracy.")

    # Updating M_vars by its optimal value.
    for i in range(0, m):
        M_vars[i] = M_vars[i].value

    return prob.value, M_vars


def find_optimal_value(
    n: int,
    d: int,
    seeds: int,
    m: int = None,
    bias: str = None,
    weight=None,
    bias_tensor=None,
    diagonal: bool = False,
    verbose: bool = True,
    prob_bound: float = const.PROB_BOUND,
    meas_bound: float = const.BOUND,
    mosek_accuracy: float = const.MOSEK_ACCURACY,
):
    """
    Find the optimal value
    ----------------------

    The objective of find_optimal_value is to estimate either the quantum or the classical value of
    a nˆm --> 1 QRAC whose dimension of the state preparations is d. Besides the three integers n, m
    and d, the user must input the desired number of starting points in the variable seeds. For each
    starting point, the code generates a random POVM and uses it to determine the optimal prepara-
    tions. After iterating over all starting points, it returns the largest optimal value obtained.
    The user can decide whether it wants to estimate the quantum or classical value by setting the
    variable diagonal equals to True or False. If True, this function forces the random measurements
    to be diagonal matrices, so that that no quantum advantage is observed.

    find_optimal_value relies on a see-saw optimization to estimate the quantum and classical values
    for a QRAC. This algorithm can be summarized in a few elementary steps, as follows:

        1. Create a set of n random measurements with m outcomes acting in dimension d.
        2. For this set of measurements, find the set optimal preparations using `find_opt_prep`.
        3. Optimize the measurements for the set of optimal preparations found in step 2 using `find
        _opt_meas`.
        4. Check the convergence of variables prob_value and M. In negative case, return to step 2.

    Once the see-saw procedure is over, find_optimal_value prints a detailed report with relevant
    information about the computation and the optimal set of measurements. This report can be disa-
    bled by setting the variable verbose = False. In this case, find_optimal_value still returns the
    same information, but now in the form of a dictionary, `report`, allowing the user to manipulate
    these data.

    Also, find_optimal_value allows the user to introduce bias in a given QRAC. It can be controlled
    by the variables `bias` and `weight`, which can be inputted in the argument of find_optimal_va-
    lue. Check the documentation of generate_bias function for more details.

    Inputs
    ------
    n: an integer.
        The number of distinct measurements.
    d: an integer.
        The dimension of the measurement operators.
    seeds: an integer.
        Represents the number of random seeds as the starting points of the see-saw algorithm.
    m: an integer.
        The number of outcomes for each measurement. If no value is attributed to m, then the code
        sets m = d.
    bias: a string or an empty variable.
        It encodes the type of bias desired. There are eight possibilities: "Y_ONE", "Y_ALL", "B_
        ONE", "B_ALL", "X_ONE", "X_DIAG", "X_CHESS" and "X_PLANE", "Y_ONE". If bias = None, the QRAC
        is unbiased. Check the documentation of the `generate_bias` function for more details.
    weight: a float or a list/tuple of floats.
        This variable carries the weight with which the QRAC is biased. If `bias` consists on a sin-
        gle-parameter family, weight must be a float ranging from 0 to 1. If `bias` consists on a
        multi-parameter family, weight must be a list (or tuple) of floats summing to 1.
    bias_tensor: a dictionary with n * m**(n + 1) entries.
        bias_tensor is an optional variable that can be inputted when the user wants to compute a
        biased QRAC scenario different of the ones comprised by the `bias` variable. bias_tensor.ke
        ys() are tuples of integers and bias_tensor.values() are floats belonging to the range (0,1]
        that encode the normalization of a biased QRAC.
    diagonal: a boolean.
        If True, find_optimal_value produces diagonal random measurements as the starting point for
        the see-saw algorithm. Since all measurements are diagonal at the same basis, no quantum ad-
        vantage is produced in this case. If false, the random measurements correspond to POVMs.
    verbose: a boolean.
        If True, verbose activates the function `printings` which produces a report about the compu-
        tation. If False, it makes find_optimal_value returns the same information in the dictionary
        `report`.
    prob_bound: a float.
        Convergence criterion for the variable prob_value. When the difference between prob_value
        and previous_prob_value is less than prob_bound, the algorithm interprets prob_value = previ
        ous_prob_value.
    meas_bound: a float.
        The same criterion as in prob_bound but for the norms of the measurement operators in the
        variable M.
    mosek_accuracy: a float.
        Feasibility tolerance used by the interior-point optimizer for conic problems in the solver
        MOSEK. Here it is used for the primal and the dual problem.

    Output
    ------
    report: a dictionary.
        report is returned if verbose = False. To avoid repetition of the documentation, check the
        documentation of the function `printings` for details about report.keys().
    """

    # Asserting that `diagonal` is a boolean variable.
    assert diagonal is True or diagonal is False, (
        "diagonal must be either True or False"
    )

    # If no value is attributed for the number of outcomes, then m = d.
    if m is None:
        m = d

    # Creating an empty dictionary report. If verbose = True, find_optimal_value prints the informa-
    # tion contained in the report. If not, it returns report.
    report = {}

    # Saving the input data in the dictionary.
    report["n"] = n
    report["dimension"] = d
    report["seeds"] = seeds
    report["outcomes"] = m
    report["bias"] = bias
    report["weight"] = weight
    report["diagonal"] = diagonal
    report["mosek accuracy"] = mosek_accuracy

    # Initializing entries "optimal value" and "optimal measurements".
    report["optimal value"] = 0
    report["optimal measurements"] = 0

    # If the user has not provided any bias_tensor, then it is generated by generate_bias. If not,
    # the code asserts if the inputted bias_tensor has the correct form.
    if bias_tensor is None:
        bias_tensor = generate_bias(n, m, bias, weight)
    else:
        assert isinstance(bias_tensor, dict), "bias_tensor must be a dicitonary"
        assert len(bias_tensor) == n * m ** (
            n + 1
        ), "len(bias_tensor) must be n * m**(n + 1)"
        assert all(
            isinstance(i, (int, float)) for i in list(bias_tensor.values())
        ), "bias_tensor.values() must be ints or floats"

        tuples = list(product(product(range(0, m), repeat=n), range(0, n), range(0, m)))

        assert (
            list(bias_tensor.keys()) == tuples
        ), "bias_tensor.keys() format not supported"

        if np.abs(sum(bias_tensor.values()) - 1) > const.BOUND:
            warnings.warn(
                colors.RED
                + "\nbias_tensor is not normalized. The result cannot be interpreted as a probabili"
                + "ty\n"
                + colors.END
            )
        if any(i < 0 for i in list(bias_tensor.values())):
            warnings.warn(
                colors.RED
                + "\nSome elements of bias_tensor are negative. The result cannot be interpreted as"
                + " a probability\n"
                + colors.END
            )

        non_qrac_tuples = []
        for i in tuples:
            if i[2] != i[0][i[1]]:
                non_qrac_tuples.append(i)
        for i in non_qrac_tuples:
            if bias_tensor[i] != 0:
                warnings.warn(colors.RED + "\nThis is not a QRAC\n" + colors.END)
                break

    # Initializing empty lists. times_list stores the total time for each seed. iterations_list sto-
    # res the number of iterations of the see-saw algorithm for each seed. optimal_values stores the
    # optimal values for each seed.
    times_list = []
    iterations_list = []
    optimal_values = []

    for i in range(0, seeds):

        # Starting point of the time count.
        start_time = tp.process_time()

        # Generating a list of n random measurements.
        M = generate_random_measurements(n, d, m, diagonal)

        # Defining the stopping conditions. The previous_M variable is started as the "null" measu-
        # rement, so to speak. It is just a dummy initialization.
        previous_prob_value = 0
        prob_value = 1
        previous_M = [
            [np.zeros((d, d), dtype=float) for i in range(0, m)] for j in range(0, n)
        ]
        max_norm_difference = 1
        iter_count = 0

        # For the measurements, we either stop when max_norm_difference is smaller than meas_bound
        # or when the number of iterations is bigger than the constant const.ITERATIONS.
        while (abs(previous_prob_value - prob_value) > prob_bound) or (
            max_norm_difference > meas_bound and iter_count < const.ITERATIONS
        ):

            previous_prob_value = prob_value

            # The two lines below correspond to two a single round of the see-saw.
            opt_preps = find_opt_prep(n, d, m, M, bias_tensor)
            prob_value, M = find_opt_meas(
                n, d, m, opt_preps, bias_tensor, mosek_accuracy
            )

            # Updating max_norm_difference.
            norm_difference = []
            for a in range(0, n):
                for b in range(0, m):
                    norm_difference.append(nalg.norm(M[a][b] - previous_M[a][b]))
            max_norm_difference = max(norm_difference)

            previous_M = M
            iter_count += 1

        # Once the see-saw is over, prob_value is appended to optimal_values.
        optimal_values.append(prob_value)
        iterations_list.append(iter_count)

        # Selecting the largest problem value from all distinct random seeds.
        if prob_value > report["optimal value"]:
            report["optimal value"] = prob_value
            report["optimal measurements"] = M
            report["best seed number"] = i + 1

        # Here, the execution finishes for one of the seeds.
        times_list.append(tp.process_time() - start_time)

    # This allows the user to know how good the values produced for each QRAC are. It counts how
    # many optimal values are close to the largest value, in terms of the solver accuracy.
    close_to_larg_value = 0
    for i in range(seeds):
        if report["optimal value"] - optimal_values[i] < mosek_accuracy:
            close_to_larg_value += 1

    # Saving data in report. To be printed at the ending of the computation.
    report["close to largest value"] = close_to_larg_value
    report["average time"] = sum(times_list) / seeds
    report["average iterations"] = sum(iterations_list) / seeds

    # Evaluating the realization found for the best seed.
    (
        report["rank"],
        report["projectiveness"],
        report["MUB check"],
    ) = check_meas_status(n, d, m, report["optimal measurements"])

    if verbose:
        printings(report)
    else:
        return report


def printings(report):

    """
    Printings
    ---------

    This function simply prints a report of the computation. It consists of two parts: a summary of
    the computation and an analysis of the optimal realization found for the best seed.

    Inputs
    ------
    report: a dictionary.
        report contains the following keys:
        n: an integer
            The number of distinct measurements.
        dimension: an integer.
            The dimension of the measurement operators.
        seeds: an integer.
            Represents the number of random seeds as the starting points of the see-saw algorithm.
        outcomes: an integer.
            The number of outcomes for each measurement.
        bias: a string or an empty variable.
            It encodes the type of bias desired. There are eight possibilities: "Y_ONE", "Y_ALL",
            "B_ONE", "B_ALL", "X_ONE", "X_DIAG", "X_CHESS" and "X_PLANE", "Y_ONE". If bias = None,
            the QRAC is unbiased. Check the documentation of the `generate_bias` function for more
            details.
        weight: a float or a list/tuple of floats.
            This variable carries the weight with which the QRAC is biased. If `bias` consists on a
            single-parameter family, weight must be a float ranging from 0 to 1. If `bias` consists
            on a multi-parameter family, weight must be a list (or tuple) of floats summing to 1.
        diagonal: a boolean.
            If True, `printings` displays a report for the classical value. If false, the report
            displays the quantum value.
        mosek_accuracy: a float.
            Feasibility tolerance used by the interior-point optimizer for conic problems in the
            solver MOSEK. Here it is used for the primal and the dual problem.
        optimal value: a float.
            The optimal value computed for the given QRAC.
        optimal measurements: a list of lists whose elements are d x d matrices.
            optimal measurements contains a nested list. Each sub-list contains the optimal measure-
            ment found after optimization.
        best seed number: an integer.
            The seed number that achieves the largest prob_value among all seeds.
        close to largest value: an integer.
            "close to largest value" informs the user how many seeds produced a value that is less
            than `mosek_accuracy` distant to the largest optimal value found among all seeds.
        average time: a float.
            The average computation time among all seeds.
        average iterations: a float.
            The average number of iterations until convergence among all seeds.
        rank: a numpy array.
            rank contains the ranks of all measurement operators for the optimal realization.
        projectiveness: a dictionary.
            It contains two keys:
            projective: a numpy array whose elements are boolean.
                projective is an array of projectiveness for all optimal measurement operators ob-
                tained in find_optimal_value. True if, for a given measurement operator M, the Fro-
                benius norm of M² - M is smaller than the constant const.BOUND.
            projectiveness measure: a numpy array whose elements are floats.
                It contains the measure of projectiveness that corresponds to the Frobenius norm of
                M² - M, for a given measurement operator M.
        MUB check: a nested dicitonary.
            report["MUB check"].keys() are tuples of integers (i, j), where i and j represent the
            indices of two distinct measurements. report["MUB check"].values() are dictionaries
            which tell the user if the pair of measurements (i, j) is mutually unbiased. See the do-
            cumentation of the `check_if_MUBs` function for details.
    """

    # This command allows printing superscripts in the prompt.
    superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    # Printing the header.
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

    print(
        f"Number of random seeds: {report['seeds']}\n"
        + f"Average time for each seed: {round(report['average time'], 5)} s\n"
        + f"Average number of iterations: "
        + f"{round(report['average iterations'])}\n"
        + f"Seeds {report['mosek accuracy']} close to the best value: "
        + f"{report['close to largest value']}\n"
        if report["seeds"] > 1
        else f"Number of random seeds: {report['seeds']}\n"
        + f"Total time of computation: {round(report['average time'], 5)} s\n"
        + f"Total number of iterations: "
        + f"{round(report['average iterations'])}\n"
    )

    # Printing the second part of the report. Analysis of the optimal realization.
    print(
        colors.BOLD
        + colors.FAINT
        + f"-" * 15
        + f" Analysis of the optimal realization for seed #{report['best seed number']} "
        + f"-" * 16
        + colors.END
        + f"\n"
        if report["best seed number"] < 10
        else colors.BOLD
        + colors.FAINT
        + f"-" * 15
        + f" Analysis of the optimal realization for seed #{report['best seed number']} "
        + f"-" * 15
        + colors.END
        + f"\n"
    )

    if report["bias"] is not None:
        print(
            f"Type of bias: {report['bias']}\n"
            + f"Weight: {round(report['weight'], 5)}"
            if isinstance(report["weight"], (float, int))
            else f"Type of bias: {report['bias']}\n" + f"Weights: {report['weight']}"
        )

    print(
        f"Estimation of the classical value for the "
        + f"{report['n']}{str(report['outcomes']).translate(superscript)}-->1 QRAC:"
        + f" {report['optimal value'].round(12)}"
        + f"\n"
        if report["diagonal"]
        else
        f"Estimation of the quantum value for the "
        + f"{report['n']}{str(report['outcomes']).translate(superscript)}-->1 QRAC:"
        + f" {report['optimal value'].round(12)}"
        + f"\n"
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
    projec_measure = report["projectiveness"]["projectiveness measure"]

    for i in range(0, report["n"]):
        for j in range(0, report["outcomes"]):
            print(
                f"M[{str(i)}, {str(j)}]:  Projective\t\t"
                + f"{str(float('%.3g' % projec_measure[i][j]))}"
                if projective[i][j]
                else f"M[{str(i)}, {str(j)}]:  Not projective\t"
                + f"{str(float('%.3g' % projec_measure[i][j]))}"
            )

    if report["MUB check"] is not None:
        keys = list(report["MUB check"].keys())

        print(" ")
        print(colors.CYAN + f"Mutual unbiasedness of measurements" + colors.END)

        for i in keys:
            print(
                f"M[{str(i[0])}] and M[{str(i[1])}]:  MUM\t\t"
                f"{str(float('%.3g' % report['MUB check'][i]['MUM measure']))}"
                if report["MUB check"][i]["MUM"]
                else f"M[{str(i[0])}] and M[{str(i[1])}]:  Not MUM\t\t"
                f"{str(float('%.3g' % report['MUB check'][i]['MUM measure']))}"
            )

    # Printing the footer of the report.
    print("")
    print(f"-" * 30 + f" End of computation " + f"-" * 30)


def check_meas_status(n, d, m, M):

    """
    Check measurement status
    ------------------------

    This function consists of several checks and small numerical calculations for the optimal set of
    measurements obtained in find_optimal_value. These procedures are summarized as follows:

    1.  Check if the measurement operators are Hermitian.
    2.  Check if the measurement operators are positive semidefinite.
    3.  Check if the measurement operators sum to identity.
    4.  Calculate the rank of the measurement operators.
    5.  Check if the measurement operators are projective.
    6.  If at least measurements are rank-one projective, then check how close these measurements
        are to being mutually unbiased. That is, it checks if these two measurements can be cons-
        tructed out of Mutually Unbiased Bases.

    Inputs
    ------
    n: an integer.
        The number of distinct measurements.
    d: an integer.
        The dimension of the measurement operators.
    m: an integer.
        The number of outcomes for each measurement.
    M: a list of n lists. Each sub-list of M contains m matrices of size d x d.
        M represents a list with n d-dimensional m-outcome measurements after optimization.

    Output
    ------
    rank: a numpy array.
        rank contains the ranks of all measurement operators for the optimal realization.
    projectiveness: a dictionary.
        It contains two keys:
        projective: a numpy array whose elements are boolean.
            projective is an array of projectiveness for all optimal measurement operators obtained
            in find_optimal_value. True if, for a given measurement operator M, the Frobenius norm
            of M² - M is smaller than the constant const.BOUND.
        projetiveness measure: a numpy array whose elements are floats.
            It contains the measure of projectiveness that corresponds to the Frobenius norm of M² -
            M, for a given measurement operator M.
    MUB_check: a nested dicitonary.
        MUB_check.keys() are tuples of integers (i, j), where i and j represent the indices of two
        distinct measurements. MUB_check.values() are dictionaries which contains the information of
        whether the pair of measurements (i, j) is mutually unbiased. See the documentation of the
        `check_if_MUBs` function for details.
    """

    # Checking if the measurement operators are Hermitian.
    for i in range(0, n):
        for j in range(0, m):
            if nalg.norm(M[i][j] - M[i][j].conj().T) > const.BOUND:
                raise RuntimeError("measurement operators are not Hermitian")

    # Checking if the measurement operators are positive semi-definite.
    for i in range(0, n):
        for j in range(0, m):
            eigval, eigvet = nalg.eigh(M[i][j])
            if eigval[0] < -const.BOUND:
                raise RuntimeError(
                    "measurement operators are not positive semi-definite"
                )

    # Checking if the measurement operators sum to identity.
    for i in range(0, n):
        sum = np.sum(M[i], axis=0)
        if nalg.norm(sum - np.eye(d)) > const.BOUND:
            raise RuntimeError("measurement operators does not sum to identity")

    # The following line returns an array with the ranks of all measurement operators.
    rank = nalg.matrix_rank(M, tol=const.BOUND, hermitian=True)

    # Checking if the measurement operators are projective.
    projective = np.empty((n, m))
    projec_measure = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):

            projec_measure[i][j] = nalg.norm(M[i][j] @ M[i][j] - M[i][j])

            if projec_measure[i][j] > const.BOUND:
                projective[i][j] = False
            else:
                projective[i][j] = True

    # Both entries `projective` and `projectiveness measure` are oredered such that the entry [i, j]
    # refers to the measurement operator from measurement i, outcome j.
    projectiveness = {
        "projective": projective,
        "projectiveness measure": projec_measure,
    }

    # This block checks if there is at least one pair of rank-one projective measurements. If yes,
    # this function checks if all such pairs can be constructed out of Mutually Unbiased Bases.
    rank1_projective_operators = rank == projective
    rank1_projec_measurements, i = (0, 0)
    while i < n and rank1_projec_measurements < 2:
        if (
            rank1_projective_operators[i]
            == [
                1,
            ]
            * m
        ).all():
            rank1_projec_measurements += 1
        i += 1

    # Mutually Unbiased Bases check. It relies on the function check_if_MUBs.
    if rank1_projec_measurements == 2:
        MUB_check = {}
        for i in range(0, n):
            for j in range(i + 1, n):
                MUB_check[(i, j)] = check_if_MUBs(m, M[i], M[j])
    else:
        MUB_check = None

    return rank, projectiveness, MUB_check


def check_if_MUBs(m, P, Q):

    """
    Check if two measurements are mutually unbiased
    -----------------------------------------------

    Complementary function for `check_meas_status`. It simply gets two d-dimensional m-outcome meas-
    urements P and Q, and checks if they can be constructed out of Mutually Unbiased Bases (MUBs).
    See appendix II of the supplementary material of the reference below for details.

    Inputs
    ------
    m: an integer.
        The number of outcomes for each measurement.
    P: a list of m d x d matrices.
        A d-dimensional m-outcome measurement. Its measurement operators are rank-one projective.
    Q: a list of m d x d matrices.
        Another d-dimensional m-outcome measurement whose measurement operators are rank-one projec-
        tive.

    Output
    ------
    status: a dictionary.
        status contains the following keys:
            MUM: a boolean.
                True if P and Q are considered mutually unbiased up to the numerical precision of
                the constant const.MUB_BOUND
            MUM measure: a float.
                It contains the measure of mutual unbiasedness of measurements P and Q. If p and q
                are measurement operators of rank-one projective measurements P and Q, respectively,
                then this measure can be written as max{m * pqp - p, m * qpq - q}, where max is ta-
                ken over all combinations of p and q.

    Reference
    ---------
    1. A. Tavakoli, M. Farkas, D. Rosset, J.-D. Bancal, J. Kaniewski, Mutually unbiased bases and
    symmetric informationally complete measurements in Bell experiments, Sci. Adv., vol. 7, issue 7.
    DOI: 10.1126/sciadv.abc3847.
    """
    status = {"MUM": True}

    partial_measures = []
    for i in range(0, m):
        for j in range(0, m):

            partial_measP = nalg.norm(m * P[i] @ Q[j] @ P[i] - P[i])
            partial_measQ = nalg.norm(m * Q[j] @ P[i] @ Q[j] - Q[j])

            partial_measures.append(partial_measP)
            partial_measures.append(partial_measQ)

            if partial_measP > const.MUB_BOUND or partial_measQ > const.MUB_BOUND:
                status["MUM"] = False

    status["MUM measure"] = max(partial_measures)

    return status


def generate_bias(n, m, bias, weight):

    """
    Generate bias
    -------------

    This function generates a dictionary in which the keys are tuples of the form

    i = ((x_0, x_1, x_2, ... , x_{n - 1}), y, b)

    where x_0, x_1, ... , x_{n - 1} and b range from 0 to m - 1 and y ranges from 0 to n - 1. The
    object bias_tensor[i] represents an order-(n + 2) tensor encoding the bias and the normalization
    in a given n^m --> 1 QRAC.

    If bias = None, the QRAC is unbiased and all of the n * m ** n elements in bias_tensor are uni-
    form. For a QRAC, the elements of bias_tensor are strictly bigger than zero whenever i[2] ==
    i[0][i[1]].

    If bias is not None, then it must assume one of the following types:

        1.  "Y_ONE". Bias in the requested character y of Bob. The element 0 of i[1] is weighted
            according to the variable `weight`. It consists on a single-paramater family of bias in
            which we have one of the requested characters versus all.

        2.  "Y_ALL". Bias in the requested character y of Bob. The y-th element of i[1] is weighted
            according to weight[y]. Differently of the previous case, this is a multi-parameter fa-
            mily of bias.

        3.  "B_ONE". Bias in the retrieved character b of Bob. The element 0 of i[2] is weighted ac-
            cording to the variable `weight`.

        4.  "B_ALL". Bias in the retrieved character b of Bob. The b-th element of i[2] is weighted
            according to weight[b].

        5.  "X_ONE". Bias in the input string x_0, x_1, x_2, ... , x_{n - 1} of Alice. The element
            (0, 0, ..., 0) of the tensor i[0] is weighted according to the variable `weight`.

        6.  "X_DIAG". Bias in the input string of Alice. The elements of the main diagonal of the
            tensor i[0] are is weighted according to the variable `weight`.

        7.  "X_CHESS". Bias in the input string of Alice. The elements of the tensor i[0] whose sum
            is odd are weighted according to the variable `weight`. For n = 2, the arrangement be-
            tween preferred and non-preferred elements resembles a chess table.

        8.  "X_PLANE". Bias in the input string of Alice. The hyperplane corresponding to x_0 = 0 in
            the i[0] tensor is preferred with probability `weight`.

    Inputs
    ------
    n: an integer.
        The number of distinct measurements.
    m: an integer.
        The number of outcomes for each measurement.
    bias: a string or an empty variable.
        It encodes the type of bias desired. There are eight possibilities: "Y_ONE", "Y_ALL", "B_
        ONE", "B_ALL", "X_ONE", "X_DIAG", "X_CHESS" and "X_PLANE", "Y_ONE". If bias = None, the QRAC
        is unbiased.
    weight: a float or a list of floats of size m.
        The variable weight carries the amount of bias the user desires in a given input (x or y) or
        output. If `bias` is a single-parameter family, then `weight` must be a float ranging from
        0 to 1. If `bias is a multi-parameter family, `weight` must be a list (or tuple) of floats
        summing to 1.

    Output
    ------
    bias: a dictionary of size n * m**(n + 1). bias.keys() are (n + 2)-tuples. bias.values() are
    floats.
        The dictionary bias represents a order-(n + 2) tensor encoding the normalization and the
        bias in a given QRAC.
    """

    # First asserting the input parameters.

    # sw/mw stands for single/multiple weights.
    valid_bias_types_sw = ("X_ONE", "X_DIAG", "X_CHESS", "X_PLANE", "Y_ONE", "B_ONE")
    valid_bias_types_mw = ("Y_ALL", "B_ALL")

    assert bias is None or bias in valid_bias_types_sw or bias in valid_bias_types_mw, (
        "Available options for bias are: "
        "X_ONE, X_DIAG, X_CHESS, X_PLANE, Y_ONE, Y_ALL, B_ONE and B_ALL."
    )

    if bias is not None:
        assert weight is not None, "a value for `weight` must be provided"

    if bias in valid_bias_types_sw:
        assert 0 <= weight <= 1, "`weight` must range between 0 and 1."

    elif bias in valid_bias_types_mw:

        assert isinstance(
            weight, (list, tuple)
        ), "For Y_ALL and B_ALL bias, `weight` must be a list or a tuple"

        if bias == "Y_ALL":

            assert len(weight) == n, "the expected size of `weight` is n"
            assert round(sum(weight), 7) == 1, "the weights must sum to one"

        elif bias == "B_ALL":

            assert len(weight) == m, "the expected size of `weight` is m"
            assert round(sum(weight), 7) == 1, "the weights must sum to one"

    # Initializing an empty dicitonary.
    bias_tensor = {}

    # Creating an iterable to represent the keys of bias_tensor.
    indexes = product(product(range(0, m), repeat=n), range(0, n), range(0, m))

    for i in indexes:

        # Enforcing the QRAC condition.
        if i[2] == i[0][i[1]]:

            # The elements must be normalized. There are n * m**n elements in total.
            bias_tensor[i] = 1 / (n * m**n)

            # Separating in cases. If bias is None, none of the below cases will match, and the re-
            # sulting tensor will be only normalized.
            if bias is None:
                continue

            elif bias == "Y_ONE":

                # If i[1] == 0, it is weighted according to `weight`. If not, it is weighted accor-
                # ding to 1 - weight distributed uniformly for the other n - 1 requested characters.
                # The factor n is a normalization factor for the bias in the requested character.
                if i[1] == 0:
                    bias_tensor[i] = n * weight * bias_tensor[i]
                else:
                    bias_tensor[i] = n * (1 - weight) * bias_tensor[i] / (n - 1)

            elif bias == "Y_ALL":

                # If i[1] == y, it is weighted according to weight[y].
                bias_tensor[i] = n * weight[i[1]] * bias_tensor[i]

            elif bias == "B_ONE":

                # For bias in the retrieved character b, the normalization factor is m.
                if i[2] == 0:
                    bias_tensor[i] = m * weight * bias_tensor[i]
                else:
                    bias_tensor[i] = m * (1 - weight) * bias_tensor[i] / (m - 1)

            elif bias == "B_ALL":

                bias_tensor[i] = m * weight[i[2]] * bias_tensor[i]

            elif bias == "X_ONE":

                # For bias in the input string the normalization factor is m**n.
                if i[0] == (0,) * n:
                    bias_tensor[i] = (m**n) * weight * bias_tensor[i]
                else:

                    # If i[0] != (0,0,...,0), we distribute 1 - weight over the m**n - 1 remaining
                    # strings.
                    bias_tensor[i] = (
                        (m**n) * (1 - weight) * bias_tensor[i] / (m**n - 1)
                    )

            elif bias == "X_DIAG":

                if len(set(i[0])) == 1:

                    # There are m elements in the diagonal of i[0], so we distribute `weight` uni-
                    # formly over these m elements.
                    bias_tensor[i] = (m**n) * weight * bias_tensor[i] / m
                else:
                    bias_tensor[i] = (
                        (m**n) * (1 - weight) * bias_tensor[i] / (m**n - m)
                    )

            elif bias == "X_CHESS":

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

            elif bias == "X_PLANE":

                if i[0][0] == 0:
                    bias_tensor[i] = m * weight * bias_tensor[i]
                else:
                    bias_tensor[i] = m * (1 - weight) * bias_tensor[i] / (m - 1)
        else:
            bias_tensor[i] = 0

    return bias_tensor


def find_classical_value(
    n: int,
    d: int,
    m: int = None,
    bias: str = None,
    weight=None,
    through_dec=True,
    verbose=True,
):

    """
    Find the classical value
    ------------------------

    find_classical_value is operated similarly as find_optimal_value. Differently from the latter,
    this function computes the classical value of a given RAC by two different methods. The first
    one is done by exhaustive search, so it simply selects the enconding and decoding strategies
    that produce the best performance for the given RAC. The second method fix a decoding strategy,
    and converts it into a set of diagonal measurements. After it, it computes the optimal prepara-
    tions and the value achieved by these preparations and measurements. Finally it selects the best
    value among all decoding strategies.

    In a RAC, one desires to encode n characters ranging from 0 to m - 1 into another character
    ranging from 0 to d - 1. In total, there are d**(m**n) encoding strategies and m**(d * n) decod-
    ing strategies. Thus, the first method scales scales double exponentially, while the second sca-
    les exponentially.

    As the cost per strategy is lower for exhaustive search, you can expect this method to be better
    for the cases where the alphabet of integers is small, particulary for (2, 2, 2) and (2, 3, 2),
    where (n, d, m) represents a tuple for integers n, d and m.

    Using the first method, the user can expect to compute the following cases in less than one hour
    : (2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 3, 2), (2, 3, 3), (2, 4, 2), (2, 5, 2), (2, 6, 2), (2, 7,
    2), (2, 8, 2), (3, 2, 2), (3, 3, 2), (3, 4, 2) and (4, 2, 2).

    As for the second method the analogous list is longer than for the first, we suffice to say that
    for n, d, m < 5, all tuples can be computed in less than one hour except for the cases (3, 4,
    4), (4, 3, 4), (4, 4, 2) and (4, 4, 4).

    Inputs
    ------
    n: an integer.
        n represents the number of encoded characters.
    d: an integer.
        d represents the size of the message to be passed to Bob.
    m: an integer.
        m represents the cardinality of the characters of Alice.
    bias: a string or an empty variable.
        It encodes the type of bias desired. There are eight possibilities: "Y_ONE", "Y_ALL", "B_
        ONE", "B_ALL", "X_ONE", "X_DIAG", "X_CHESS" and "X_PLANE", "Y_ONE". If bias = None, the RAC
        is unbiased. Check the documentation of the `generate_bias` function for more details.
    weight: a float or a list/tuple of floats.
        This variable carries the weight with which the RAC is biased. If `bias` consists on a sin-
        gle-parameter family, weight must be a float ranging from 0 to 1. If `bias` consists on a
        multi-parameter family, weight must be a list (or tuple) of floats summing to 1.
    through_dec: a boolean. True by default.
        If True, the function computes the classical value using the second method. As the second
        method is faster for most choices of n, d and m, this is the default.
    verbose: a boolean. True by default.
        If true, it prints a small report cointaing the classical value and the total computation
        time. If false, it simply returns the classical value.

    Outputs
    -------
    report: a dictionary.
        report is returned if verbose = False. To avoid repetition of the documentation, check the
        documentation of the function `classical_printings` for details about report.keys().
    """

    # If no value is attributed for the number of outcomes, then I set m = d.
    if m is None:
        m = d

    # Creating an empty dictionary report. If verbose = True, find_classical_value prints the infor-
    # mation contained in the report. If not, it returns report.
    report = {}

    # Saving the input data in the dictionary.
    report["n"] = n
    report["dimension"] = d
    report["outcomes"] = m
    report["bias"] = bias
    report["weight"] = weight
    report["through decodings"] = through_dec

    bias_tensor = generate_bias(n, m, bias, weight)

    # Initializing an empty list of classical_probability.
    classical_probability = []

    if through_dec:

        start = tp.time()

        # Initializing the indexes for the loops ahead.
        decodings = product(range(m), repeat=d * n)
        indexes = list(product(product(range(m), repeat=n), range(n), range(m)))

        # These matrices are simply d-dimensional rank-one projectors in the computational basis.
        projectors = [np.zeros((d, d), dtype=float) for i in range(d)]
        for i in range(d):
            projectors[i][i, i] = 1

        for i in decodings:

            # In this method, we reuse find_opt_prep to compute the best enconding for a given de-
            # coding. To do this, we first convert the decoding strategies into measurements. As
            # these measurements are deterministic, they can be written as diagonal matrices. Here,
            # we initialize these measurements and convert the decodings into measurements.
            measurements = [
                [np.zeros((d, d), dtype=float) for i in range(m)] for j in range(n)
            ]
            for a in range(n):
                for b in range(m):
                    for c in range(d):
                        if i[c * n + a] == b:
                            measurements[a][b] += projectors[c]

            preparations = find_opt_prep(n, d, m, measurements, bias_tensor)

            # Since we have measurements and preparations, we can compute the value of the RAC func-
            # tional.
            functional = 0
            for j in indexes:
                functional += bias_tensor[j] * np.trace(
                    preparations[j[0]] @ measurements[j[1]][j[2]]
                )

            classical_probability.append(functional)

    else:

        start = tp.time()

        # Enumerating all possible strategies. The first product represents all possible encodings
        # while the second represents all possible decodings.
        strategies = product(
            product(range(d), repeat=m**n), product(range(m), repeat=d * n)
        )

        # The variable iterable is just an auxiliary variable. It will be transformed into the list
        # `indexes` afterwards.
        iterable = product(product(range(m), repeat=n), range(n))
        indexes = []

        # The variable iterable is equivalent to the tuples in `generate_bias`, except that it does
        # not contain the entry for b. This loop converts the tuple (x_1, x_2, ..., x_n) into a de-
        # cimal number and saves it in the last entry of `indexes`.
        for i in iterable:
            decimal = 0
            N = n - 1
            for j in i[0]:
                decimal += j * m**N
                N -= 1
            indexes.append((i, decimal))
        indexes = [((a), b, c) for ((a), b), c in indexes]

        for i in strategies:

            strategy_prob = 0
            for j in indexes:

                # Selecting what messaging digit will be send to Bob.
                message = i[0][j[2]]

                # Based on the received digit, Bob produces output b.
                b = i[1][message * n + j[1]]

                # This is simply the RAC condition.
                if b == j[0][j[1]]:

                    # This is a deterministic strategy. If the RAC condition is satisfied, then pro-
                    # bability = 1 * bias_tensor.
                    strategy_prob += bias_tensor[(j[0], j[1], b)]

            classical_probability.append(strategy_prob)

    report["total time"] = tp.time() - start

    # Optimizing over all strategies.
    report["classical value"] = max(classical_probability)

    max_index = classical_probability.index(max(classical_probability))

    # This paragraph retrieves the optimal strategy by providing the first index of classical_proba-
    # bility that achieves the largest value. It simply converts a decimal number into a base m num-
    # ber. This way saves both memory and computation time from the above loops.
    strategy_len = d * n if through_dec else m**n + d * n
    optimal_strategy = np.zeros(strategy_len, dtype=int)
    while max_index > 0:
        optimal_strategy[strategy_len - 1] = max_index % m
        max_index = int(max_index / m)
        strategy_len -= 1

    report["optimal strategy"] = (
        tuple(optimal_strategy)
        if through_dec
        else (
            tuple(optimal_strategy[: m**n]),
            tuple(optimal_strategy[m**n : m**n + d * n]),
        )
    )

    # Counting how many strategies achieve the optimal value.
    optimal_strategies_count = 0
    for i in classical_probability:
        if i == report["classical value"]:
            optimal_strategies_count += 1

    report["optimal strategies number"] = optimal_strategies_count

    if verbose:
        classical_printings(report)
    else:
        return report


def classical_printings(report):

    """
    Printings for find_classical_value
    ----------------------------------

    This function simply prints a report for the computation of find_classical_value. It also con-
    sists of two parts: a summary of the computation and an analysis of the optimal realization.

    Inputs
    ------
    report: a dictionary.
        report contains the following keys:
        n: an integer.
            It represents the number of encoded characters.
        dimension: an integer.
            It represents the size of the message to be passed to Bob. It also represents the dimen-
            sion of the quantum system once the decodings are transformed to measurements.
        outcomes: an integer.
            It represents the cardinality of the characters of Alice. It also represents the number
            of outcomes once the decodings are transformed to measurements.
        bias: a string or an empty variable.
            It encodes the type of bias desired. There are eight possibilities: "Y_ONE", "Y_ALL",
            "B_ONE", "B_ALL", "X_ONE", "X_DIAG", "X_CHESS" and "X_PLANE", "Y_ONE". If bias = None,
            the RAC is unbiased. Check the documentation of the `generate_bias` function for more
            details.
        weight: a float or a list/tuple of floats.
            This variable carries the weight with which the RAC is biased. If `bias` consists on a
            single-parameter family, weight must be a float ranging from 0 to 1. If `bias` consists
            on a multi-parameter family, weight must be a list (or tuple) of floats summing to 1.
        through decodings: a boolean.
            It encodes whether the used method was exhaustive search or search through decodings.
            The produced report is different for each of the methods.
        total time: a float.
            The total time of computation for a given method.
        classical value: a float.
            The classical value computed for the given RAC.
        optimal strategy: a tuple of integers.
            The first optimal strategy found that achieves the computed value. If through decodings=
            True, 'optimal strategy' corresponds to a single tuple of integers representing the a
            decoding strategy. If False, 'optimal strategy' corresponds to two tuples representing
            the encoding and the decoding strategy, respectively.
        optimal strategies number: a positive integer.
            The total number of strategies achieving the computed value.
    """

    # This command allows printing superscripts in the prompt.
    superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    # Printing the header.
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

    checked_str = report["outcomes"] ** (report["dimension"] * report["n"])
    if not report["through decodings"]:
        checked_str = (
            report["dimension"] ** (report["outcomes"] ** report["n"]) * checked_str
        )

    print(f"Total time of computation: {round(report['total time'], 6)} s")
    print(f"Total number of strategies checked: {checked_str}")
    print(
        f"Average time for each strategy: {round(report['total time'] / checked_str, 6)} s\n"
    )

    # Printing the second part of the report. Analysis of the optimal realization.
    print(
        colors.BOLD
        + colors.FAINT
        + f"-" * 21
        + f" Analysis of the optimal realization "
        + f"-" * 22
        + colors.END
        + f"\n"
    )

    if report["bias"] is not None:
        print(
            f"Type of bias: {report['bias']}\n"
            + f"Weight: {round(report['weight'], 5)}"
            if isinstance(report["weight"], (float, int))
            else f"Type of bias: {report['bias']}\n" + f"Weights: {report['weight']}"
        )

    print(
        f"Computation of the classical value for the "
        + f"{report['n']}{str(report['outcomes']).translate(superscript)}-->1 RAC:"
        + f" {round(report['classical value'], 12)}"
    )

    print(
        f"Number of decoding strategies achieving the computed value: "
        + f"{report['optimal strategies number']}\n"
        if report["through decodings"]
        else f"Number of strategies achieving the computed value: "
        + f"{report['optimal strategies number']}\n"
    )

    print(
        colors.CYAN + f"First strategy found achieving the computed value" + colors.END
    )

    if report["through decodings"]:
        print(f"Decoding: {report['optimal strategy']}")
    else:
        print(
            f"Encoding: {report['optimal strategy'][0]}\n"
            + f"Decoding: {report['optimal strategy'][1]}"
        )

    # Printing the footer of the report.
    print("")
    print(f"-" * 30 + f" End of computation " + f"-" * 30)
