"""
QRACs
-----
A Quantum Random Acess Code, mostly known by its acronym QRAC, is a task in which n dits are encoded
into a qudit. The aim of this task is to recover one of the dits chosen uniformly at random. A QRAC
is generally represented by its parameters nˆ(d) --> 1, in which n is the number of dits, and d is
the size of the dits, as well as the dimension of the qudit.

This program encounters the maximum value of a QRAC by performing a see-saw optimization. Encounter-
ing the 'maximum value' of a QRAC means that we aim to maximize the functional associated with it.

The main function of this code is implemented in find_QRAC_value(n, d, seeds), in which it is pos-
sible to input n, d, and the number of seeds to be used in the see-saw optimization.

References
----------
1. A. Ambainis, D. Leung, L. Mancinska and M. Ozols, Quantum Random Access Codes with Shared Random-
ness, available in arXiv:0810.2937.
"""
import cvxpy as cp
import numpy as np
import numpy.linalg as nalg
from numpy.random import rand
from scipy.linalg import sqrtm
from itertools import product

def generate_random_measurements(d):

    """
This function generates a random measurement with d outcomes and dimension d. The first step is to
generate d random linear operators named 'random_lin'. Then, it transforms these operators into a
hermitian matrix and stores them in the list 'partial_meas'. By rescaling 'partial_sum' using the
sum of all 'random_herm' operators, it can produce a random measurement.

Input
-----
d: an integer
    The dimension of the measurement operators as well as the number of outcomes for the generated
    measurement.

Output
------
measurement: a list of d bidimensional arrays of size d x d.
    The variable measurement represent a random measurement with d outcomes and dimension d.
    """
    # Bound for the checkings.
    BOUND = 1e-7

    # Creating empty auxiliary variables.
    partial_meas = []
    partial_sum = 0

    # Creating 'd' random hermitian operators and appending them to the list 'partial_meas'.
    for i in range(0, d):

        random_lin = 2 * rand(d, d) - np.ones((d, d)) + 1j * (2 * rand(d, d) - np.ones((d, d)))
        # Random linear operator

        random_herm = random_lin.T.conj() @ random_lin
        # Random Hermitian operator
        partial_meas.append(random_herm)
        partial_sum += random_herm

    # Checking if 'partial_sum' is full-rank
    if (nalg.matrix_rank(partial_sum, tol = BOUND, hermitian = True) != d):
        print("WARNING: 'partial_sum' is not full-rank \n")
        return 0

    # Empty list of measurements
    measurement = []

    # This is the operator I will use to rescale the 'partial_meas' list. It is only the inverse
    # square root of 'partial_sum'.
    inv_sqrt = nalg.inv(sqrtm(partial_sum))

    # Generating the random measurement operators
    for i in range(0, d):

        # Rescaling 'partial_meas[i]'
        M = inv_sqrt @ partial_meas[i] @ inv_sqrt
        # Enforcing hermiticity
        M = 0.5 * ( M + M.conj().T )
        # Checking if 'M' is positive semi-definite
        eigval, eigvet = nalg.eigh(M)
        for j in range(0, d):
            if(eigval[j] < BOUND):
                print("WARNING: Random measurement operators are not positive semi-definite.\n")
                return 0
        measurement.append(M)

    # Last check. Checking if the measurement operators sum to identity
    sum = 0
    for i in range(0, d):
        sum += measurement[i]
    if nalg.norm(sum - np.eye(d)) > BOUND:
        print("WARNING: Random measurement operators doesn't sum to identity\n")
        return 0

    return measurement

def find_opt_prep(M, d, n):

    """
This function acts jointly with opt_state(meas_oprt, d, n). The objective is to generate the set of
optimal states for a set of measurements M. It produces a dictionary 'opt_preps' that contains the
optimal preparations for a given combination of measurement operators.

Inputs
------
M: a list of lists whose elements are arrays of size d x d.
    M is a list containing n lists. Each list inside M corresponds to a different measurement in the
    QRAC task. For each measurement there are d measurement operators.
d: an integer.
    d represents the number of outcomes of the measurements.
n: an integer.
    n represents the number of distinct measurements.

Output
------
opt_preps: a dictionary in which the keys are given by an n-tuple and the content of each entry is
an array of size d x d.
    In a n-dits QRAC, in which the dits are labelled as x_1, x_2, ..., x_n, for every combination of
    dits, a state is prepared and sent to Bob. The dictionary 'opt_preps' contains the optimal pre-
    parations related to each combination of dits. It is optimal in the sense that it maximizes the
    figure-of-merit 'average success probability' for a given set of measurements.

Example
-------
Let us say that a QRAC with 3 dits is set. Then, we indicate a combination of measurement operators
by a tuple of size 3. The tuple (3, 4, 5) indicates the third outcome for the first measurement, the
fourth outcome for the second measurement and the fifth for the third measurement. The optimal pre-
paration for the combination of measurement operators (3, 4, 5) is saved in the dictionary
'opt_preps' whose key corresponds to the tuple '(3, 4, 5)'.
    """

    # Creating an empty dictionary.
    opt_preps = {}

    # The variable 'indexes' list all the possible tuples of size n with elements ranging from 0 to
    # d - 1.
    indexes = product([i for i in range(0, d)], repeat = n)

    # This for runs over all n-tuples of 'indexes'. In practice, it is a for nested n times.
    for i in indexes:

        # I am just retrieving the indexes from the 'i'-th tuple and saving the corresponding mea-
        # surement operators in this list 'meas_oprt'. Then, I compute the optimal preparation for
        # these measurement operators.
        meas_oprt = [M[j][i[j]] for j in range(0, n)]
        opt_preps[i] = opt_state(meas_oprt, d, n)

    return opt_preps

def opt_state(meas_oprt, d, n):

    """
Complementary function for find_opt_prep(M, d, n). In this function, 'meas_oprt' represents an or-
dered list of measurement operators of different measurements. The first element corresponds to a
certain measurement operator of the first measurement and so on.

For obtaining the optimal preparation for a certain combination of measurement operators, I am com-
puting the eigenvector related to the largest eigenvalue of 'average success probability'. From this
eigenvector, I generate the optimal state.

Inputs
------
meas_oprt: a list with n are arrays of size d x d.
d: an integer.
    d represents the number of outcomes of the measurements.
n: an integer.
    n represents the number of distinct measurements.

Output
------
rho: an array of size d x d.
    The optimal state preparation for the combination of measurement operators in the 'meas_oprt'
    list.
    """

    # The 'average success probability' is defined as simply the sum of the elements in 'meas_oprt'.
    sum = 0
    for i in range(0, n):
        sum += meas_oprt[i]

    # numpy.eigh provides the eigenvalues and eigenvectors ordered from the smallest to the largest.
    eigval, eigvet = nalg.eigh(sum)

    # Just selecting the eigenvector related to the latest (also largest) eigenvalue.
    rho = np.outer(eigvet[:, d - 1], eigvet[:, d - 1].conj())

    return rho

def find_opt_meas(opt_preps, d, n):

    """
This function acts jointly with opt_meas(opt_preps_sum, d, n). It solves a single step of the see-
saw algorithm by optimizing all of the measurements of Bob independently. It receives as input a
dictionary containing the optimal preparations for all combinations of measurement operator of dis-
tinct measurements of Bob.

The optimizations of all measurements can be made independently. To optimize over the i-th measure-
ment we sum over all indexes of the dictionary 'opt_preps' but the i-th. Remember that the keys of
'opt_preps' are n-tuples. Then, a list 'opt_preps_sum' of sums is provided to 'opt_meas' function.

Inputs
------
opt_preps: a dictionary in which the keys are given by an n-tuple and the content of each entry is
an array of size d x d.
d: an integer.
    d represents the number of outcomes of the measurements.
n: an integer.
    n represents the number of distinct measurements.

Output
------
prob_sum: a float.
    It contains the value of the optimization of the 'average success probability' after a single
    step of the see-saw algorithm.
M: a list of lists whose elements are arrays of size d x d.
    M is the same list as in the function find_opt_prep(M, d, n) after a single step of the see-saw
    algorithm.
    """

    # Defining empty variables. I am reusing M to replace the new measurements after optimization.
    M = []
    prob_sum = 0

    # Each step of this loop stands for a different measurement.
    for i in range(0, n):

        # Defining empty variables again.
        opt_preps_sum = []
        for j in range(0, d):

            sum = np.zeros((d, d))
            indexes = product([i for i in range(0, d)], repeat = n)

            # Summing through all indexes of 'indexes' but the i-th.
            for k in indexes:
                if (k[i] == j):
                    sum = sum + opt_preps[k]

            # Appending d sums to 'opt_preps_sum'
            opt_preps_sum.append(sum)

        # Solving the problem for the i-th measurement
        prob_value, meas_value = opt_meas(opt_preps_sum, d, n)
        prob_sum += prob_value
        M.append(meas_value)

    return prob_sum, M

def opt_meas(opt_preps_sum, d, n):

    """
Complementary function for find_opt_meas(opt_preps, d, n). This function takes as input one of Bob's
measurements M[i] and a collection of optimal preparations 'opt_preps' with respect to M.

The structure of this function is a simple SDP optimization for the objective function 'prob_guess'.

Inputs
------
opt_preps_sum: list of d arrays of size d x d.
    This list contains all of the preparations of Alice summed over all indexes but the i-th.
d: an integer.
    d represents the number of outcomes of the measurements.
n: an integer.
    n represents the number of distinct measurements. I only need n here to calculate the factor
    multiplying 'prob_guess'.

Outputs
-------
prob.value: A float.
    Numerical solution of 'prob' via CVXPY.
M_vars: a list of d arrays of size d x d.
    The optimized measurement M[i] after a single iteration of the see-saw algorithm.
    """

    # Creating a list of measurements. Each element of this list will be a CVXPY variable.
    M_vars = []

    # Appending the variables to the list:
    for i in range(0, d):
        M_vars.append(cp.Variable((d, d), hermitian = True))

    # Defining objective function:
    prob_guess = 0
    for i in range(0, d):
        prob_guess += (1/(n*d**n)) * cp.trace(M_vars[i]@opt_preps_sum[i])

    # Defining constraints:
    constr = []
    sum = 0
    for i in range(0, d):

        # M must be positive semi-definite.
        constr.append(M_vars[i] >> 0)

        # The elements of M must sum to identity.
        sum += M_vars[i]
    constr.append(sum == np.eye(d))

    # Defining the SDP and solving. CVXPY cannot recognize the objective function is real.
    prob = cp.Problem(cp.Maximize(cp.real(prob_guess)), constr)
    prob.solve(solver = 'MOSEK')

    # Updating 'M_vars' by its optimal value.
    for i in range (0, d):
        M_vars[i] = M_vars[i].value

    return prob.value, M_vars

############################################### MAIN ###############################################

def find_QRAC_value(n, d, seeds, meas_sts = True, precision = False):

    """
Main function. Here I perform a see-saw optimization for the nˆ(d) --> 1 QRAC. This function can be
described in a few elementary steps, as follows:

1. Create a set of n d-dimensional random measurements with generate_random_measurements(d).
2. For this set of measurements, find the set optimal preparations using find_opt_prep(M, d, n).
3. Optimize the measurements for the set of optimal preparations found in step 2 using find_opt_meas
(opt_preps, d, n).
4. Check if the obtained solution to the problem is converging. If not, return to step 2.

This function also checks if the obtained optimal measurements are MUBs by activating meas_status
function.

Inputs
------
n: an integer.
    n represents the number of distinct measurements.
d: an integer.
    d represents the number of outcomes of the measurements.
seeds: an integer.
    Represents the number of random seeds as the starting point of the see-saw algorithm.
meas_sts: True or False. True by default.
    If true, it activates the function meas_status for details about the optimized measurements.
precision: True or False. False by default.
    Argument to be passed to the function meas_status. If True, instead of producing a binary answer
    to the question of whether two measurements are MUBs or not, it recovers the maximum precision
    with which we can say that a pair of measurements are MUBs.

Outputs
-------
max_prob_value: a float
    The optimized value of the 'average success probability' for the nˆ(d) --> 1 QRAC.
    """

    # Starting variables max_prob_value and M_optimal.
    max_prob_value = 0
    M_optimal = 0

    for i in range(0, seeds):

        M = []
        for j in range(0, n):

            # generate_random_measurements(d) should retrieves zero if the measurement operators are
            # not positive-semidefinite.
            random_measurement = 0
            while random_measurement == 0:
                random_measurement = generate_random_measurements(d)
            M.append(random_measurement)

        # Definiing the stopping conditions.
        previous_prob_value = 0
        prob_value = 1
        previous_M = [[(1/d) * np.eye(d) for i in range(0, d)] for j in range(0, n)]
        stop_cond = 1
        j = 0
        while (abs(previous_prob_value - prob_value) > 1e-10) or stop_cond > 5 * 1e-7:
        # prob_value converges faster than the measurements, so that the stopping condition for
        # prob_value can be smaller.

            previous_prob_value = prob_value

            # The two lines below correspond two a single round of see-saw.
            opt_preps = find_opt_prep(M, d, n)
            prob_value, M = find_opt_meas(opt_preps, d, n)

            cond = []
            for a in range(0, n):
                for b in range(0, d):
                    cond.append(nalg.norm(M[a][b] - previous_M[a][b]))

            stop_cond = max(cond)
            previous_M = M

        # Selecting the largest problem value from all distinct random seeds.
        if (prob_value > max_prob_value):
            max_prob_value = prob_value
            M_optimal = M

    # meas_sts is True by default.
    if meas_sts:
        meas_status(M_optimal, n, d, precision)

    return max_prob_value

def meas_status(M, n, d, precision = False):

    """
This function simply checks the status of the optimized measurements. It checks whether the measure-
ment operators are Hermitian, positive semi-definite, rank-one, projective and if they sum to iden-
tity, not necessarily in this order. To finish it checks if all of the pairs of measurements are
constructed out of mutually unbiased bases using the function check_if_MUBS(P, Q, d).

Inputs
------
M: a list of lists whose elements are arrays of size d x d.
    M is a list containing n lists. Each list inside M corresponds to a different measurement in the
    QRAC task. For each measurement there are d measurement operators.
d: an integer.
    d represents the number of outcomes of the measurements.
n: an integer.
    n represents the number of distinct measurements.
precision: True or False. False by default.
    Argument to be passed to the function check_if_MUBS. If True, instead of producing a binary
    answer to the question of whether two measurements are MUBs or not, it recovers the maximum
    precision with which we can say that a pair of measurements are MUBs.

Output
------
Features about the input measurement M.
    """
    # Bound for the checkings.
    BOUND = 1e-7

    # Checking if the measurement operators are Hermitian.
    for i in range(0, n):
        for j in range(0, d):
            if nalg.norm(M[i][j] - M[i][j].conj().T) > BOUND:
                    return print("Measurement operators are not Hermitian.\n")

    # Checking if the measurement operators are positive semi-definite.
    for i in range(0, n):
        for j in range(0, d):
            eigval, eigvet = nalg.eigh(M[i][j])
            for k in range(0, d):
                if(eigval[j] < - 1 * BOUND):
                    return print("Measurement operators are not positive semi-definite.\n")

    # Checking if the measurement operators sum to identity.
    for i in range(0, n):
        sum = 0
        for j in range(0, d):
            sum += M[i][j]
        if nalg.norm(sum - np.eye(d)) > BOUND:
            return print("Measurement operators doesn't sum to identity\n")

    # This will return me an array with the ranks of all measurement operators.
    rank = nalg.matrix_rank(M, tol = BOUND, hermitian = True)
    # Now checking if all of the measurement operators are rank-one.
    if (rank == np.ones((n, d), dtype = np.int8)).all() != True:
        return print("Measurement operators are not rank-one!\n")

    # Checking if the measurement operators are projective.
    for i in range(0, n):
        for j in range(0, d):
            if nalg.norm(M[i][j] @ M[i][j] - M[i][j]) > BOUND:
                return print("Measurement operators are not projective!\n")

    # Checking if the measurements are constructed out of Mutually Unbiased Bases.
    if precision:
        for i in range(0, n):
            for j in range(i + 1, n):
                print('M[' + str(i) + '] and M[' + str(j) + '] are mutually unbiased with precision of at most ' + str(check_if_MUBS(M[i], M[j], d, precision).round(8)))
                print('')
    else:
        for i in range(0, n):
            for j in range(i + 1, n):
                if check_if_MUBS(M[i], M[j], d) == 1:
                    print('M[' + str(i) + '] and M[' + str(j) + '] are mutually unbiased.')
                else:
                    print('M[' + str(i) + '] and M[' + str(j) + '] are not mutually unbiased.')
                print('')

def check_if_MUBS(P, Q, d, precision = False):

    """
This function works jointly with meas_status(M, n, d). It simply gets two d-dimensional measurements
P and Q, and checks if they are constructed out of Mutually Unbiased Bases. Check appendix II of the
supplementary material of the reference for details.

Inputs
------
P: a list with d arrays of size d x d.
    A d-dimensional measurement with d outcomes.
Q: a list with d arrays of size d x d.
    A d-dimensional measurement with d outcomes.
d: an integer.
precision: True or False. False by default.
    If True, instead of producing a binary 0 or 1, it recovers the maximum precision with which we
    can say that a pair of measurements are MUBs.

Output
------
1:
    If the pair of measurements is mutually unbiased.
0:
    If the pair of measurements is not mutually unbiased.
max(precision): a float.
    The maximum precision with which we can say that a pair of measurements are MUBs. It is only re-
    turned if precision = True.

Reference
---------
1. A. Tavakoli, M. Farkas, D. Rosset, J.-D. Bancal, J. Kaniewski, Mutually unbiased bases and sym-
metric informationally complete measurements in Bell experiments, Sci. Adv., vol. 7, issue 7. DOI:
10.1126/sciadv.abc3847.
    """
    # Defining a proper bound to accept the polynomial equation of P and Q as satisfied.
    BOUND = 1e-5

    if precision:
        precision = []
        for i in range(0, d):
            for j in range(0, d):
                precision.append(nalg.norm(d * P[i] @ Q[j] @ P[i] - P[i]))
                precision.append(nalg.norm(d * Q[j] @ P[i] @ Q[j] - Q[j]))
        return max(precision)

    else:
        for i in range(0, d):
            for j in range(0, d):
                if nalg.norm(d * P[i] @ Q[j] @ P[i] - P[i]) > BOUND:
                    return 0
                if nalg.norm(d * Q[j] @ P[i] @ Q[j] - Q[j]) > BOUND:
                    return 0

        return 1
