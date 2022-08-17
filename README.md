# QRACs

A Quantum Random Access Code, mostly known by its acronym QRAC, is a task in which $n$ digits are encoded into a qudit. The aim of this task is to recover one of the digits chosen uniformly at random. A QRAC is generally represented by its parameters $n^d \rightarrow 1$, in which $n$ is the number of digits, and $d$ is the size of the alphabet of digits, as well as the dimension of the qudit.

This program encounters the quantum value of the average success probability for a QRAC by performing a _see-saw_ optimization. This function is implemented in `find_QRAC_value`, in which it is possible to input $n$ and $d$, among other desirable parameters to be used in seesaw optimization.

In addition, our code allows introducing bias in the QRAC. In a nutshell, we say that a QRAC is biased when the assumption that the digits are chosen uniformly is ignored. In this case, the user can choose among certain distributions for the digits.

## Examples

1. *The simplest QRAC:* $2^2 \rightarrow 1$. In this task, $2$ bits are encoded into a qubit. The quantum value of the average success probability can be retrieved by evoking

```
> find_QRAC_value(n = 2, d = 2, seeds = 5)
```
<ul>
where <em>n</em> denotes the number of digits to be encoded and <em>d</em> denotes the size of Alice's alphabet. In this case, we are using bits. The entry <em>seeds</em> is an optimization parameter that indicates the number of starting points for the optimization. For each starting point, a random measurement is generated and inputted as the zero-th iteration for the see-saw algorithm. In the end of the computation, the code outputs the largest average success probability obtained over all starting points, producing the following report.
</ul>

```
================================================================================
                                QRAC-tools v1.0
================================================================================

---------------------------- Summary of computation ----------------------------

Number of random seeds: 5
Best seed: 2
Seeds whose measurements converged below MEAS_BOUND: 1, 2, 3, 4, 5
Average time until convergence: 0.17155 s
Average number of iterations until convergence: 3

--------------------- Analysis of the optimal realisation ----------------------

Optimal value for the 2²-->1 QRAC: 0.8535533484

Measurement operator ranks
M[0] ranks:  1  1
M[1] ranks:  1  1

Measurement operator projectiveness
M[0, 0]:  Projective		7.96e-08
M[0, 1]:  Projective		7.96e-08
M[1, 0]:  Projective		7.96e-08
M[1, 1]:  Projective		7.96e-08

Mutually unbiasedness of measurements
M[0] and M[1]:  MUM		1.13e-07

------------------------------ End of computation ------------------------------
```

The first part of this report is a summary of the computation. It presents the number of random starting points (referred to as *seeds*), the best starting point, and for which of the starting points there was a convergence of the measurements for the established criterion. Here the criterion used is the largest difference between the measurement operators norms obtained by two consecutive iterations of the see-saw algorithm.

In addition, the summary of computation also presents the average process time for each starting point, as well as the average number of iterations until the convergence of the see-saw algorithm.

For the second part of the report, the quantum value for the average success probability is shown, followed by some information about the measurements. Note that this value matches the one found by Ref. [1]. Then, the user can check that the measurement operators are rank-one projective. The number shown in the second column of "Measurement operator projectiveness" corresponds to the Frobenius norm of the operator $M^2 - M$, where $M$ represents an arbitrary measurement operator. In the case where the measurement operators are rank-one projective, the code presents some information of whether each pair of measurements can be constructed out of Mutually Unbiased Basis.

According to Ref. [2], if $P_i$ and $Q_j$ are two rank-one projective measurement operators of two distinct $d$-outcome measurements, then if

$$
d\, P_i Q_j P_i = P_i \quad \text{and} \quad d\, Q_j P_i Q_j = Q_j, \quad \forall P_i, Q_j,
$$

the two measurements can be constructed out of a pair of Mutually Unbiased Basis. In this case, the number in the second column of "Mutually unbiasedness of measurements" represents the largest Frobenius norm of the operators $d\, P_i Q_j P_i - P_i$ and $d\, Q_j P_i Q_j - Q_j$, for all $i$ and $j$.

2. *Increasing the dimension of the quantum system*. In the `find_QRAC_value` procedure, the user is also allowed to specify different values for the dimension of the quantum system and the size of Alice's alphabet. Usually, in a QRAC, these values are the same. So if we are dealing with qubits, we assume that Alice is encoding bits. However, it does not need to be so. Let us say that one desires to encode 2 bits into a qutrit, for instance. In this case, we are still dealing with the $2^2 \rightarrow 1$ QRAC, but we are "cheating" in a certain way, because we are allowing the quantum system to be bigger in dimension. The quantum value of the average success probability for this example can be retrieved by evoking

```
> find_QRAC_value(n = 2, d = 3, seeds = 5, m = 2)
```
<ul>
where <em>n</em> represents the number of encoded digits, <em>d</em> is the dimension of the quantum system and <em>m</em> now is the size of Alice's alphabet. In this case, we are still using bits. The second part of the report should be as follows.
</ul>

```
--------------------- Analysis of the optimal realisation ----------------------

Optimal value for the 2²-->1 QRAC: 0.9045084875

Measurement operator ranks
M[0] ranks:  2  1
M[1] ranks:  1  2

Measurement operator projectiveness
M[0, 0]:  Projective		2.83e-08
M[0, 1]:  Projective		2.53e-08
M[1, 0]:  Projective		2.13e-08
M[1, 1]:  Projective		2.37e-08

------------------------------ End of computation ------------------------------
```
<ul>
Clearly we obtain a bigger average success probability for this problem, as it should be.
</ul>

Note that, unlike the previous example, here the variable *d* assumes the role of the dimension of the quantum system, while *m* is the size of Alice's alphabet. Also, note that before we did not need to specify *m*. As in the standard QRAC these two variables are the same, if *m* is not provided, the code sets `m = d` automatically. Having said that, we decided to keep *d* as the dimension of the quantum system, to avoid misconceptions.

3. *Experimenting bias in the requested digit*. Now, let us say that in the $2^2 \rightarrow 1$ QRAC the user wants to retrieve the first digit of Alice's string with weight `weight`. Clearly, the quantum value of the average success probability will not be the same, since we have preference for the first of the digits. We can can calculate it by setting some value to `weight` and indicating the kind of `bias` one desires:

```
> find_QRAC_value(n = 2, d = 2, seeds = 5, bias = "YPARAM", weight = 0.75)

================================================================================
                                QRAC-tools v1.0
================================================================================

---------------------------- Summary of computation ----------------------------

Number of random seeds: 5
Best seed: 5
Seeds whose measurements converged below MEAS_BOUND: 1, 2, 3, 4, 5
Average time until convergence: 0.16531 s
Average number of iterations until convergence: 3

--------------------- Analysis of the optimal realisation ----------------------

Optimal value for the 2²-->1 QRAC: 0.8952847075

Measurement operator ranks
M[0] ranks:  1  1
M[1] ranks:  1  1

Measurement operator projectiveness
M[0, 0]:  Projective		1.66e-11
M[0, 1]:  Projective		1.66e-11
M[1, 0]:  Projective		6.1e-13
M[1, 1]:  Projective		6.1e-13

Mutually unbiasedness of measurements
M[0] and M[1]:  MUM		2.27e-11

------------------------------ End of computation ------------------------------
```

In this case, "YPARAM" stands for a single-parameter bias in the requested digit, which is, by default, named as *y* in many QRAC scenarios. Check the documentation of the `generate_bias` function for alternative biases.

4. *Trying the testing script*. Finally, the user is also allowed to compare the produced results with the ones found in the literature. By typing

```
> run test.py
```
<ul>
the code will output a few testing cases, as follows:
</ul>

```
================================================================================
                                QRAC-tools v1.0
================================================================================

Testing qubit QRACs
For the 2²-->1 QRAC:
Literature value: 0.8535534   Computed value: 0.8535533   Difference:  4e-08
For the 3²-->1 QRAC:
Literature value: 0.7886751   Computed value: 0.7886751   Difference:  9e-12
For the 4²-->1 QRAC:
Literature value: 0.7414810   Computed value: 0.7414815   Difference:  -5e-07

Testing higher dimensional QRACs
For the 2³-->1 QRAC:
Literature value: 0.7886751   Computed value: 0.7886751   Difference:  1e-08
For the 2⁴-->1 QRAC:
Literature value: 0.7500000   Computed value: 0.7500000   Difference:  3e-13
For the 2⁵-->1 QRAC:
Literature value: 0.7236068   Computed value: 0.7236068   Difference:  1e-10

Testing the y-biased 2²-->1 QRAC for 3 random weights
For weight = 0.625:
Expected quantum value: 0.864467   Computed value: 0.864467   Difference:  3e-10
For weight = 0.972:
Expected quantum value: 0.986178   Computed value: 0.986178   Difference:  3e-09
For weight = 0.586:
Expected quantum value: 0.858743   Computed value: 0.858743   Difference:  1e-09

Testing the b-biased 2²-->1 QRAC for 4 random weights
For weight = 0.150:
Expected quantum value: 0.924779   Computed value: 0.924779   Difference:  2e-11
For weight = 0.589:
Expected quantum value: 0.856443   Computed value: 0.856443   Difference:  5e-11
For weight = 0.484:
Expected quantum value: 0.853646   Computed value: 0.853646   Difference:  1e-14
For weight = 0.936:
Expected quantum value: 0.967768   Computed value: 0.967768   Difference:  6e-11

------------------------------ End of computation ------------------------------
```
<ul>
The first set of QRACs correspond to <em>d</em> = <em>m</em> = 2 and <em>n</em> = 2, 3 and 4. The second set can be retrieved by varying <em>d</em> and </em>m</em> so that <em>d</em> = <em>m</em> and keeping <em>n</em> = 2. Similarly, the third and fourth sets of QRACs can be obtained by setting the bias as "YPARAM", as in example 3, and "BPARAM", respectively.
</ul>

## Requirements

This script requires the packages `time`, `cvxpy`, `numpy`, `scipy` and `itertools`. Moreover, while `cvxpy` comes with some pre-installed solvers, we have not found them particularly reliable. We strongly recommend you to install the solver __MOSEK__, which requires a license that can be obtained [here](https://www.mosek.com/products/academic-licenses/).

The packages mentioned above can be installed by running:

```
pip install -r requirements.txt
```

## References

1. A. Ambainis, D. Leung, L. Mancinska and M. Ozols, Quantum Random Access Codes with Shared Randomness, available in [arXiv:0810.2937](https://arxiv.org/abs/0810.2937).

2. A. Tavakoli, M. Farkas, D. Rosset, J.-D. Bancal and J. Kaniewski, Mutually unbiased bases and symmetric informationally complete measurements in Bell experiments, Sci. Adv. 7 (2021).
