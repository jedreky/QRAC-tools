"""
This file contains the main functions for optimizing an nˆ(dim) --> 1 QRAC. The main function can be
accessed by the command 'find_QRAC_value'.
"""

import cvxpy as cp
import numpy as np
import numpy.linalg as nalg
from numpy.random import rand
from scipy.linalg import sqrtm
from itertools import product


def generate_random_measurements(dim, out):

    """
    Generate random measurements
    ----------------------------

    This function generates a random measurement with 'out' outcomes and dimension 'dim'. The first
    step is to generate 'out' random complex operators named 'random_op'. Then, it transforms these
    operators into a positive semidefinite matrix and stores them in the list 'partial_meas'. By re-
    scaling 'partial_sum' using the sum of all 'random_herm' operators, it can produce a random mea-
    surement.

    Input
    -----
    dim: an integer
        The dimension of the measurement operators.
    out: an integer
        The number of outcomes for the generated measurement.

    Output
    ------
    measurement: a list of 'out' matrices of size dim x dim.
        The variable measurement represent a random measurement with 'out' outcomes and dimension
        'dim'.
    """
    # Bound for the checkings.
    BOUND = 1e-7

    # Trying to generate the measurement for 10 times. If, in a certain iteration, a suitable measu-
    # rement is not produced, the code skips the iteration and tries again using the command 'conti-
    # nue'.
    attempts = 0
    while attempts < 10:
        attempts += 1

        # Initializing an empty list.
        partial_meas = []

        # Creating d random complex operators and appending them to the list 'partial_meas'.
        for i in range(0, out):

            # Generating random complex operators. Each entry should have both the real and imagina-
            # ry parts between [-1, +1].
            random_op = (
                2 * rand(dim, dim)
                - np.ones((dim, dim))
                + 1j * (2 * rand(dim, dim) - np.ones((dim, dim)))
            )

            # Transforming random_op into a hermitian operator. Note that 'random_herm' is also po-
            # sitive semidefinite by construction.
            random_herm = random_op.T.conj() @ random_op

            partial_meas.append(random_herm)

        partial_sum = np.sum(partial_meas, axis=0)

        # CHECKING if 'partial_sum' is full-rank
        if nalg.matrix_rank(partial_sum, tol=BOUND, hermitian=True) != dim:
            continue

        # Initializing an empty list.
        measurement = []

        # This is the operator I will use to rescale the 'partial_meas' list. It is only the inverse
        # square root of 'partial_sum'.
        inv_sqrt = nalg.inv(sqrtm(partial_sum))

        # Generating the random measurement operators and appending to the list 'measurement'.
        flag = False
        for i in range(0, out):

            # Rescaling 'partial_meas[i]'
            M = inv_sqrt @ partial_meas[i] @ inv_sqrt
            # Enforcing hermiticity
            M = 0.5 * (M + M.conj().T)

            measurement.append(M)

            # CHECKING if M is positive semidefinite. Recall that eigh produces ordered eigenvalues.
            # It suffices to check if the first eigenvalue is non-negative. In negative case, flag
            # is changed to True and another iteration of the while loop is started with 'continue'.
            eigval, eigvet = nalg.eigh(M)
            if eigval[0] < -BOUND:
                flag = True

        if flag:
            continue

        # Last check. CHECKING if the measurement operators sum to identity
        sum = np.sum(measurement, axis=0)
        if nalg.norm(sum - np.eye(dim)) > BOUND:
            continue

        return measurement

    raise RuntimeError("A random measurement cannot be generated.")


def find_opt_prep(M, dim, n, out, bias):

    """
    Find optimal preparations
    -------------------------

    This function acts jointly with 'opt_state' function. The objective is to generate the set of
    optimal states for a set of measurements M. It produces a dictionary 'opt_preps' that contains
    the optimal preparations for a given combination of measurement operators.

    Inputs
    ------
    M: a list of lists whose elements are matrices of size dim x dim.
        M is a list containing n lists. Each list inside M corresponds to a different measurement in
        the QRAC task. For each measurement there are 'out' measurement operators.
    dim: an integer.
        dim represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements.
    out: an integer.
        out represents the number of outcomes of each measurement.
    bias: a dictionary of size n * out ** n. bias.keys() are tuples of n + 2 coordinates. bias.value
    s() are floats.
        The dictionary 'bias' represents a order-(n+2) tensor enconding the bias in some given QRAC.

    Output
    ------
    opt_preps: a dictionary in which the keys are given by an n-tuple and the content of each entry
    is a matrix of size dim x dim.
        Every element of this dictionary corresponds to a rank-one density matrix for a given prepa-
        ration of Alice. In a n-dits QRAC, in which the dits are labelled as x_1, x_2, ..., x_n, for
        every combination of dits, a state is prepared by Alice and sent to Bob. The dictionary 'opt
        _preps' contains the optimal preparations related to each combination of dits. It is optimal
        in the sense that it maximizes the figure-of-merit 'success probability' for a given set of
        measurements.

    Example
    -------
    Let us say that a QRAC with 3 dits is set. Then, we indicate a combination of measurement opera-
    tors by a tuple of size 3. The tuple (3, 4, 5) indicates the third outcome for the first measu-
    rement, the fourth outcome for the second measurement and the fifth for the third measurement.
    The optimal preparation for the combination of measurement operators (3, 4, 5) is saved in the
    dictionary 'opt_preps' whose key corresponds to the tuple '(3, 4, 5)'.
    """

    # Creating an empty dictionary.
    opt_preps = {}

    # The variable 'indexes_of_x' list all the possible tuples of size n with elements ranging from
    # 0 to 'out' - 1.
    indexes_of_x = product(range(0, out), repeat = n)

    # This for runs over all n-tuples of 'indexes_of_x'. In practice, it is a for nested n times.
    for i in indexes_of_x:

        # Initializing an empty list.
        operators_list = []

        for j in range(0, n):
            for k in range(0, out):

                # Here, I am running over all tuples 'i', and indexes 'j' and 'k'. The tuple (i, j,
                # k) corresponds exactly to the keys of the dicitonary 'bias'. Then, for each ele-
                # ment bias[(i, j, k)] I multiply the correct measurement operator and append to
                # operators_list. Finally, I compute the optimal preparation for these measurement
                # operators.
                operators_list.append(bias[(i, j, k)] * M[j][k])

        opt_preps[i] = opt_state(operators_list, dim)

    return opt_preps


def opt_state(operators_list, dim):

    """
    Optimal state
    -------------

    Complementary function for 'find_opt_prep' function. In this function, 'operators_list' repre-
    sents an ordered list of measurement operators of different measurements. The first element cor-
    responds to a certain measurement operator of the first measurement and so on.

    For obtaining the optimal preparation for a certain combination of measurement operators, I am
    computing the eigenvector related to the largest eigenvalue of "average success probability".
    From this eigenvector, I generate the optimal state.

    Inputs
    ------
    operators_list: a list with n matrices of size dim x dim.
    dim: an integer.
        dim represents the dimension of the measurement operators.

    Output
    ------
    rho: an array of size dim x dim.
        The optimal state preparation for the combination of measurement operators in 'operators_lis
        t'.
    """

    # The 'average success probability' is defined as simply the sum of the elements in 'operators_
    # list'.
    sum = np.sum(operators_list, axis=0)

    # numpy.eigh provides the eigenvalues and eigenvectors ordered from the smallest to the largest.
    eigval, eigvet = nalg.eigh(sum)

    # Just selecting the eigenvector related to the latest (also largest) eigenvalue.
    rho = np.outer(eigvet[:, dim - 1], eigvet[:, dim - 1].conj())

    return rho


def find_opt_meas(opt_preps, dim, n, out, bias):

    """
    Find optimal measurements
    -------------------------

    This function acts jointly with 'opt_meas' function. It solves a single step of the see-saw al-
    gorithm by optimizing all of the measurements of Bob independently. It receives as input a dic-
    tionary containing the optimal preparations for all combinations of measurement operators of
    distinct measurements of Bob.

    The optimizations of all measurements can be made independently. To optimize over the i-th mea-
    surement we sum over all indexes of the dictionary 'opt_preps'. Remember that the keys of 'opt_p
    reps' are n-tuples. Then, a list 'opt_preps_sum' of sums is provided to 'opt_meas' function.

    Inputs
    ------
    opt_preps: a dictionary in which the keys are given by an n-tuple and the content of each entry
    is a matrix of size dim x dim.
    dim: an integer.
        d represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements.
    out: an integer.
        out represents the number of outcomes of each measurement.
    bias: a dictionary of size n * out ** n. bias.keys() are tuples of n + 2 coordinates. bias.value
    s() are floats.
        The dictionary 'bias' represents a order-(n+2) tensor enconding the bias in some given QRAC.

    Output
    ------
    prob sum: a float
        It contains the value of the optimization of the "average success probability" after a sin-
        gle step of the see-saw algorithm.
    M: a list of lists whose elements are matrices of size dim x dim.
        M is the same list as in the function 'find_opt_prep' after a single step of the see-saw al-
        gorithm.
    """

    # Defining empty variables.
    M = []
    prob_sum = 0

    # Each step of this loop stands for a different measurement.
    for j in range(0, n):

        # Initializing an empty list.
        opt_preps_sum = []

        for k in range(0, out):

            indexes = product(range(0, out), repeat = n)

            # Summing through all indexes of 'indexes'. This sum is weighted by the correct bias
            # element, bias[(i, j, k)].
            sum = np.sum([bias[(i, j, k)] * opt_preps[i] for i in indexes], axis = 0)

            # Appending 'out' sums to 'opt_preps_sum'
            opt_preps_sum.append(sum)

        # Solving the problem for the j-th measurement
        prob_value, meas_value = opt_meas(opt_preps_sum, dim, n, out)
        prob_sum += prob_value
        M.append(meas_value)

    return prob_sum, M


def opt_meas(opt_preps_sum, dim, n, out):

    """
    Optimal measurement
    -------------------

    Complementary function for 'find_opt_meas' function. This function takes as input one of Bob's
    measurements M[i] and a collection of optimal preparations 'opt_preps' with respect to M.

    The structure of this function is a simple SDP optimization for the objective function 'prob_gue
    ss'.

    Inputs
    ------
    opt_preps_sum: list of d matrices of size dim x dim.
        This list contains all of the preparations of Alice summed over all indexes but the i-th.
    dim: an integer.
        dim represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements. I only need n here to calculate the factor
        multiplying 'prob_guess'.
    out: an integer.
        out represents the number of outcomes of the measurements.

    Outputs
    -------
    prob.value: A float.
        Numerical solution of 'prob' via CVXPY.
    M_vars: a list of 'out' matrices of size dim x dim.
        The optimized measurement M[i] after a single iteration of the see-saw algorithm.
    """

    # Creating a list of measurements. Each element of this list will be a CVXPY variable.
    M_vars = []

    # Appending the variables to the list:
    for i in range(0, out):
        M_vars.append(cp.Variable((dim, dim), hermitian=True))

    # Defining objective function:
    prob_guess = 0
    for i in range(0, out):
        prob_guess += cp.trace(M_vars[i] @ opt_preps_sum[i])

    # Defining constraints:
    constr = []
    sum = 0
    for i in range(0, out):

        # M must be positive semi-definite.
        constr.append(M_vars[i] >> 0)

    # The elements of M_vars must sum to identity.
    sum = cp.sum(M_vars, axis=0)
    constr.append(sum == np.eye(dim))

    # Defining the SDP and solving. CVXPY cannot recognize the objective function is real.
    prob = cp.Problem(cp.Maximize(cp.real(prob_guess)), constr)
    prob.solve(solver="MOSEK")

    # Updating 'M_vars' by its optimal value.
    for i in range(0, out):
        M_vars[i] = M_vars[i].value

    return prob.value, M_vars


############################################### MAIN ###############################################

def find_QRAC_value(
    n: [int],
    dim: [int],
    seeds: [int],
    out: [int] = None,
    meas_status: [bool] = True,
    PROB_BOUND: [float] = 1e-9,
    MEAS_BOUND: [float] = 5e-7,
    bias: [str] = None,
    weight = None
):
    """
    Find the Quantum Random Acces Code quantum value
    ------------------------------------------------

    Main function. Here I perform a see-saw optimization for the nˆ(dim) --> 1 QRAC. This function
    can be described in a few elementary steps, as follows:

        1. Create a set of n random measurements with 'out' outcomes acting dimension 'dim' with 'ge
        nerate_random_measurements'.
        2. For this set of measurements, find the set optimal preparations using 'find_opt_prep'.
        3. Optimize the measurements for the set of optimal preparations found in step 2 using 'find
        _opt_meas'.
        4. Check if the variables 'prob_value' and 'M' are converging. If not, return to step 2 till
        the difference between 'previous_prob_value' and 'prob_value' is smaller than PROB_BOUND.
        Also, the loop should terminate either when 'max_norm_difference' is smaller than MEAS_BOUND
        or when the number of iterations is bigger than 'iterations.

    This function also checks if the obtained optimal measurements are MUBs by activating 'determine
    _meas_status' function.

    Inputs
    ------
    n: an integer.
        n represents the number of distinct measurements.
    dim: an integer.
        The dimension of the measurement operators.
    seeds: an integer.
        Represents the number of random seeds as the starting point of the see-saw algorithm.
    out: an integer. [optional]
        The number of outcomes for the measurements. If no value is attributed to 'out', then out =
        dim.
    meas_status: True or False. True by default. [optional]
        If true, it activates the function determine_meas_status for details about the optimized
        measurements.
    PROB_BOUND: a float. [optional]
        Convergence criterion for the variable 'prob_value'. When the difference between 'prob_va-
        lue' and 'previous_prob_value' is less than PROB_BOUND, the algorithm interprets prob_value
        = previous_prob_value.
    MEAS_BOUND: a float. [optional]
        The same criterion as in PROB_BOUND but for the norms of the measurement operators in the
        variable M.
    bias: a dictionary of size n * out ** n. bias.keys() are tuples of n + 2 coordinates. bias.value
    s() are floats.
        The dictionary 'bias' represents a order-(n+2) tensor enconding the bias in some given QRAC.
    weight: a float or a list of floats of size 'out'.
        The varible 'weight' carries the weight (or weights) with which the QRAC is biased. If it is
        a single float, it must be a positive number between 0 and 1. If it is a list of floats, it
        must sum to 1.
        Note that, for the case in which 'weight' is a single float, this variable is symmetrical.
        That is, setting weight = 0.5 has the same effect as making bias = None.

    Output
    ------
    The optimized value of the 'average success probability' for the nˆ(dim) --> 1 QRAC.
    """
    # Starting variables max_prob_value and M_optimal.
    max_prob_value = 0
    M_optimal = 0

    # Maximum number of iterations for the see-saw loop.
    iterations = 100

    # If no value is attributed for the number of outcomes, then I set out = dim.
    if out is None:
        out = dim

    print("")

    # Here I am generating the bias and saving it in the tensor 'bias_tensor'. By default, the QRAC
    # is unbiased, so the 'bias = None'.
    bias_tensor = generate_bias(n, out, bias, weight)

    for i in range(0, seeds):

        M = []
        for j in range(0, n):

            # The code is finished in the case where generate_random_measurements cannot produce a
            # suitable measurement. See generate_random_measurements(d) for details.
            random_measurement = generate_random_measurements(dim, out)
            assert np.shape(random_measurement) == (
                out,
                dim,
                dim,
            ), "Encountered measurement object of unexpected shape."
            M.append(random_measurement)

        # Defining the stopping conditions. The previous_M variable is started as the "null" measu-
        # rement, so to speak. It is just a dummy initialization.
        previous_prob_value = 0
        prob_value = 1
        previous_M = [
            [np.zeros((dim, dim), dtype=float) for i in range(0, out)]
            for j in range(0, n)
        ]
        max_norm_difference = 1
        iter_count = 0

        # The variable 'prob_value' converges faster than 'M', so that the stopping condition for
        # 'prob_value' can be smaller. For the measurements, we either stop when 'max_norm_differen-
        # ce' is smaller than MEAS_BOUND or when the number of iterations is bigger than 'itera-
        # tions'.
        while (abs(previous_prob_value - prob_value) > PROB_BOUND) or (
            max_norm_difference > MEAS_BOUND and iter_count < iterations
        ):

            previous_prob_value = prob_value

            # The two lines below correspond to two a single round of see-saw.
            opt_preps = find_opt_prep(M, dim, n, out, bias_tensor)
            prob_value, M = find_opt_meas(opt_preps, dim, n, out, bias_tensor)

            norm_difference = []
            for a in range(0, n):
                for b in range(0, out):
                    norm_difference.append(nalg.norm(M[a][b] - previous_M[a][b]))

            max_norm_difference = max(norm_difference)
            previous_M = M
            iter_count += 1

        # Print message that only 'prob_value' converged.
        if iter_count >= iterations:
            print(
                "The measurements have not converged below the MEAS_BOUND for the seed #"
                + str(i + 1)
                + "."
            )

        # Selecting the largest problem value from all distinct random seeds.
        if prob_value > max_prob_value:
            max_prob_value = prob_value
            M_optimal = M
            seed_number = i

    # Just printing the max_prob_value.
    print(
        "The optimized value for the "
        + str(n)
        + "ˆ"
        + str(out)
        + "-->1 QRAC is "
        + str(prob_value.round(10))
        + ", found by the seed #"
        + str(seed_number + 1)
        + "."
    )

    # meas_status is True by default.
    if meas_status:
        determine_meas_status(M_optimal, dim, n, out)


def determine_meas_status(M, dim, n, out):

    """
    Determine measurement status
    ----------------------------

    This function simply checks the status of the optimized measurements. It checks whether the mea-
    surement operators are Hermitian, positive semi-definite, rank-one, projective and if they sum
    to identity, not necessarily in this order. To finish it checks if all of the pairs of measure-
    ments are constructed out of mutually unbiased bases using the function 'check_if_MUBs'.

    Inputs
    ------
    M: a list of lists whose elements are matrices of size dim x dim.
        M is a list containing n lists. Each list inside M corresponds to a different measurement in
        the QRAC task. For each measurement there are 'out' measurement operators.
    dim: an integer.
        dim represents the dimension of the measurement operators.
    n: an integer.
        n represents the number of distinct measurements.
    out: an integer.
        out represents the number of outcomes of the measurements.

    Output
    ------
    Features about the input measurement M.
    """
    # Bound for the checkings.
    BOUND = 1e-7

    # Flag for the MUB checking. If the measurement operators are neither rank-one or projective,
    # there is no sense on checking if they can be constructed out of MUBs.
    flag = True

    print("")

    # Checking if the measurement operators are Hermitian.
    for i in range(0, n):
        for j in range(0, out):
            if nalg.norm(M[i][j] - M[i][j].conj().T) > BOUND:
                raise RuntimeError("Measurement operators are not Hermitian.")

    # Checking if the measurement operators are positive semi-definite.
    for i in range(0, n):
        for j in range(0, out):
            eigval, eigvet = nalg.eigh(M[i][j])
            if eigval[0] < -BOUND:
                raise RuntimeError(
                    "Measurement operators are not positive semi-definite."
                )

    # Checking if the measurement operators sum to identity.
    for i in range(0, n):
        sum = np.sum(M[i], axis=0)
        if nalg.norm(sum - np.eye(dim)) > BOUND:
            raise RuntimeError("Measurement operators does not sum to identity.")

    # This will return me an array with the ranks of all measurement operators.
    rank = nalg.matrix_rank(M, tol=BOUND, hermitian=True)
    # Now checking if all of the measurement operators are rank-one.
    if not (rank == np.ones((n, out), dtype=np.int8)).all():
        flag = False
        print("Measurement operators are not rank-one!")

        # Printing the ranks of the measurement operators.
        line = 0
        for i in rank:
            print("M[" + str(line) + "] ranks: ", "  ".join(map(str, i)))
            line += 1

    # Boolean variable for a line break.
    break_line = True

    # Checking if the measurement operators are projective.
    for i in range(0, n):
        for j in range(0, out):
            if nalg.norm(M[i][j] @ M[i][j] - M[i][j]) > BOUND:
                if break_line:
                    print("")
                    break_line = False
                flag = False
                print(
                    "Measurement operator M["
                    + str(i)
                    + "]["
                    + str(j)
                    + "] is not projective!"
                )

    # Checking if the measurements are constructed out of Mutually Unbiased Bases.
    if flag:
        for i in range(0, n):
            for j in range(i + 1, n):
                status = check_if_MUBs(M[i], M[j], out)
                if status["boolean"]:
                    print(
                        "M["
                        + str(i)
                        + "] and M["
                        + str(j)
                        + "] are mutually unbiased. The "
                        "maximum norm difference is "
                        + str(status["max error"].round(8))
                        + "."
                    )
                else:
                    print(
                        "M["
                        + str(i)
                        + "] and M["
                        + str(j)
                        + "] are not mutually unbiased. "
                        "The maximum norm difference is "
                        + str(status["max error"].round(8))
                        + "."
                    )


def check_if_MUBs(P, Q, out, MUB_BOUND = 1e-5):

    """
    Check if two measurements are mutually unbiased
    -----------------------------------------------

    This function works jointly with 'determine_meas_status' function. It simply gets two 'dim'-di-
    mensional measurements P and Q, and checks if they are constructed out of Mutually Unbiased Ba-
    ses. Check appendix II of the supplementary material of the reference for details.

    Inputs
    ------
    P: a list with 'out' matrices of size dim x dim.
        A measurement with 'out' outcomes acting on dimension 'dim'.
    Q: a list with 'out' matrices of size dim x dim.
        Another measurement with 'out' outcomes acting on dimension 'dim'.
    out: an integer.
        out represents the number of outcomes of the measurements.
    MUB_BOUND: a float.
        If PQP-P < MUB_BOUND, then the equation PQP = P is considered satisfied and the measurements
        P and Q are considered mutually unbiased.

    Output
    ------
    status: a dictionary with two entries. A boolean variable and a float.
        The first key of this dictionary is named 'boolean'. If 'True', P and Q are considered mutu-
        ally unbiased up to the numerical precision of MUB_BOUND. The second key is named 'max_er-
        ror' and contains the maximum norm difference by which the measurement operators of P and Q
        do not satisfy the equations PQP = P and QPQ = Q.

    Reference
    ---------
    1. A. Tavakoli, M. Farkas, D. Rosset, J.-D. Bancal, J. Kaniewski, Mutually unbiased bases and
    symmetric informationally complete measurements in Bell experiments, Sci. Adv., vol. 7, issue 7.
    DOI: 10.1126/sciadv.abc3847.
    """
    status = {"boolean": True}

    errors = []
    for i in range(0, out):
        for j in range(0, out):

            errorP = nalg.norm(out * P[i] @ Q[j] @ P[i] - P[i])
            errorQ = nalg.norm(out * Q[j] @ P[i] @ Q[j] - Q[j])

            errors.append(errorP)
            errors.append(errorQ)

            if errorP > MUB_BOUND or errorQ > MUB_BOUND:
                status["boolean"] = False

    status["max error"] = max(errors)

    return status


def generate_bias(n, out, bias, weight):

    """
    Generate bias
    -------------

    This function generates a dictionary in which the keys are tuples of the form

    i = ((x_1, x_2, x_3, ... , x_n), y, b)

    where x_1, x_2, x_3, ... , x_n and b range from 0 to out - 1 and y ranges from 0 to n - 1. The
    object bias tensor[i] represents a order-(n+2) tensor enconding the bias in some given QRAC.

    If bias = None, the QRAC is unbiased and all of the n * out ** n elements in bias_tensor are
    uniform. The elements of bias_tensor are strictly bigger than zero whenever i[2] == i[0][i[1]].

    There are seven possible kinds of bias, as follows.
    1. "XDIAG". Bias in the input string x. The elements of the main diagonal of the order-n tensor
    i[0] are prefered with probability 'weight'. For the n = 2 QRAC, it translates as

    x_1/x_2 | 0  1  2 ... out-1
    0       | *  .  .       .
    1       | .  *  .       .
    2       | .  .  *       .
    :       |
    out-1   | .  .  .       *

    2. "XCHESS". Bias in the input string x. The elements of the order-n tensor i[0] are prefered
    with probability 'weight', in an array that resembles a chess table. If n = 2,

    x_1/x_2 | 0  1  2 ... out-1
    0       | .  *  .       *
    1       | *  .  *       .
    2       | .  *  .       *
    :       |
    out-1   | *  .  *       .

    3. "XPARAM". Bias in the input string x. The element (0, 0, ..., 0) of the order-n tensor i[0]
    is prefered probability 'weight'

    4. "YPARAM". Bias in the requested digit y. The element 0 of i[1] is prefered with probability
    'weight'.

    5. "YVEC". Bias in the requested digit y. The y-th element of i[1] is prefered with probability
    'weight[y]'.

    6. "BPARAM". Bias in the retrived output b. The element 0 of i[2] is prefered with probability
    'weight'.

    7. "BVEC". Bias in the retrived output b. The b-th element of i[2] is prefered with probability
    'weight[b]'.

    Inputs
    ------
    n: an integer.
        n represents the number of distinct measurements.
    out: an integer.
        out represents the number of outcomes of the measurements.
    bias: a string or an empty variable ('None').
        It encodes the type of bias desired for the computation. There are seven possibilities: "XDI
        AG", "XCHESS", "XPARAM", "YPARAM", "YVEC", "BPARAM" and "BVEC".
    weight: a float or a list of floats of size 'out'.
        The varible 'weight' carries the weight (or weights) with which the QRAC is biased. If it is
        a single float, it must be a positive number between 0 and 1. If it is a list of floats, it
        must sum to 1.
        Note that, for the case in which 'weight' is a single float, this variable is symmetrical.
        That is, setting weight = 0.5 has the same effect as making bias = None.

    Output
    ------
    bias_tensor: a dictionary of size n * out ** n. bias_tensor.keys() are tuples of n + 2 coordina-
    tes. bias_tensor.values() are floats.
        bias_tensor represents a order-(n+2) tensor enconding the bias in some given QRAC.
    """

    # Initializing an empty dicitonary.
    bias_tensor = {}

    # Creating an iterable for feeding bias_tensor. The keys of bias_tensor will be the tuples defi-
    # ned by the iterable 'indexes'.
    indexes = product(product(range(0, out), repeat = n), range(0, n), range(0, out))

    for i in indexes:

        # Enforcing the QRAC condition.
        if i[2] == i[0][i[1]]:

            # The elements must be uniform. There n * out**n elements in bias_tensor in total.
            bias_tensor[i] = 1 / (n * out**n)

            # Separating in cases. If bias = None, none of the below cases will match, and the re-
            # sulting bias_tensor will be unbiased.
            if bias == "XDIAG":

                assert 0 <= weight <= 1, "For the XDIAG bias, 'weight' must range between 0 and 1."

                # This is a normalization constant for the XDIAG case. It ensures that sum(bias_tens
                # or.values()) == 1.
                norm = (2 * weight - 1) / (out ** (n - 1)) - weight + 1

                # If i[0] is a diagonal element, then it is prefered with 'weight'. If not, it is
                # prefered with 1 - weight. The other cases follow similarly.
                if len(set(i[0])) == 1:
                    bias_tensor[i] = weight * bias_tensor[i] / norm
                else:
                    bias_tensor[i] = (1 - weight) * bias_tensor[i] / norm

            elif bias == "XCHESS":

                assert 0 <= weight <= 1, "For the XCHESS bias, 'weight' must range between 0 and 1."

                # Here the normalization depends on the parity of out ** n. Recall that parity(out)
                # == parity(out ** n), for positive n and out.
                if out % 2 == 0:
                    norm = 0.5
                else:
                    norm = (1 + (n - 2 * n * weight) / (n * out ** n)) / 2

                if sum(i[0]) % 2 == 1:
                    bias_tensor[i] = weight * bias_tensor[i] / norm
                else:
                    bias_tensor[i] = (1 - weight) * bias_tensor[i] / norm

            elif bias == "XPARAM":

                assert 0 <= weight <= 1, "For the XPARAM bias, 'weight' must range between 0 and 1."

                norm = 1 - weight - (n - 2 * n * weight) / (n * out ** n)

                if i[0] == (0,) * n:
                    bias_tensor[i] = weight * bias_tensor[i] / norm
                else:
                    bias_tensor[i] = (1 - weight) * bias_tensor[i] / norm

            elif bias == "YPARAM":

                assert 0 <= weight <= 1, "For the YPARAM bias, 'weight' must range between 0 and 1."

                norm = 1 - weight - 1 / n + (2 * weight) / n

                if i[1] == 0:
                    bias_tensor[i] = weight * bias_tensor[i] / norm
                else:
                    bias_tensor[i] = (1 - weight) * bias_tensor[i] / norm

            elif bias == "YVEC":

                assert len(weight) == n, "For the YVEC bias, the expected size of 'weight' is n."

                assert round(sum(weight), 7) == 1, "For the YVEC bias, the weights must sum to one."

                bias_tensor[i] = n * weight[i[1]] * bias_tensor[i] / sum(weight)

            elif bias == "BPARAM":

                assert 0 <= weight <= 1, "For the BPARAM bias, 'weight' must range between 0 and 1."

                norm = 1 - weight - 1 / out + (2 * weight) / out

                if i[2] == 0:
                    bias_tensor[i] = weight * bias_tensor[i] / norm
                else:
                    bias_tensor[i] = (1 - weight) * bias_tensor[i] / norm

            elif bias == "BVEC":

                assert len(weight) == out, "For the BVEC bias, the expected size of 'weight' is d."

                assert round(sum(weight), 7) == 1, "For the BVEC bias, the weights must sum to one."

                bias_tensor[i] = out * weight[i[2]] * bias_tensor[i] / sum(weight)

        else:
            bias_tensor[i] = 0

    return bias_tensor
