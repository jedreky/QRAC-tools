# Random Access Codes

A Random Access Code (RAC) is a compression task in which a string of $n$ characters is encoded in a lower-dimensional system. The goal of this task is to then recover one of the characters chosen at random. In the classical case, the storage system is a dit and the encoding and decoding processes are functions (perhaps non-deterministic). In the quantum case, the storage system is a qudit. The encoding process consists of preparing various qudit states, while the decoding consists of performing a quantum measurement.

A typical RAC protocol involves two parties, Alice and Bob. Alice is given an $n$-character string $\bf x$, chosen from an alphabet of cardinality $m$, and is asked to encode it into a single character $\mu$ of cardinality $d$. The message $\mu$ and an input $y$ are then sent to Bob, who is asked to retrieve one of the characters of $\bf x$, $x_y$. Upon receiving $\mu$ Bob implements a decoding map that outputs a character $b$. Whenever $b=x_y$, the protocol is considered successful. The integers $n$, $d$ and $m$ completely define the RAC scenario, which we denote by $n^m \smash{\overset{d}{\mapsto}} 1$. The figure of merit commonly used to quantify the performance of the protocol is the _average success probability_ $\bar P$, given by

$$
\bar{P} = \frac{1}{nd^n} \sum_{\mathbf{x}, y} p (b = x_y \, | \, \mathbf{x}, ~ y),
$$

where $p (b=x_y \, | \, \mathbf{x}, ~ y)$ denotes the probability of a successful decoding when string ${\bf x}$ is encoded and character $x_y$ is to be recovered. The factor $\frac{1}{n d^n}$ reflects the assumption that both $\bf x$ and $y$ are uniformly distributed. A quantum realization for this protocol consists of a set of qudit preparations $\rho_{\bf x}$ and measurement operators $\{M_y^{x_y}\}$ such that $p(b=x_y|{\bf x}, y)= \text{tr} (\rho_{\bf x} M_y^{x_y})$, i.e., Alice encodes $\bf x$ in the state of a quantum system, and Bob decodes it via quantum measurements. Different preparations and measurements will result in different values for the figure of merit, and we refer to those achieving its maximal value (_quantum value_) as _optimal quantum strategies_. Finding these strategies and the RAC quantum value is of great interest from the point of view of quantum information processing since it is useful to quantify the advantage that quantum resources can provide [1], or possibly self-test quantum devices [2].  

This package provides functions to find the classical value of a RAC (using exhaustive methods and see-saw optimization) as well as lower bounds on the quantum value (see-saw). Moreover, one can investigate RACs in which the distribution of inputs $\bf x$ and $y$ is not uniform. More specifically, we propose to study a variation of the RAC protocol, which we call _biased_ RACs or simply _b_-RACs, whose ensuing figure of merit is given by

$$
\mathcal{F} =
	\sum_{\mathbf{x}, y} \alpha_{\mathbf{x} y} \, p ( b=x_y \, | \, \mathbf{x},\, y),
$$

where $\alpha_{\mathbf{x} y}$ denotes the components of a $(n+1)$-order tensor, satisfying $\alpha_{\mathbf{x} y}>0$ and $\sum_{\mathbf{x}, y} \alpha_{\mathbf{x} y} = 1$.

In the following, we introduce a requirements section and the three main component functions of RAC-tools: `generate_bias`, `perform_search` and `perform_seesaw`.  The first of these functions, which is not to be accessed by the user, is introduced with the aim of facilitating the user to specify the bias tensor defining a particular _b_-RAC without the need of declaring all of its entries. It allows the user to choose from several simple and natural families of bias tensors, the members of which are specified by one or more parameters. It should be noted that, nonetheless, the user is allowed to build a bias tensor from scratch and pass it as an argument to any of the remaining functions, which can be accessed by the user to compute either the classical or quantum value of the ensuing $b$-RAC.

`perform_search` allows the computation of the classical value of a $b$-RAC. Since classical strategies belong to the convex hull of the deterministic ones, the optimal classical performance is attained with one of the latter, which can be found via an exhaustive search algorithm. `perform_search` implements one of two improved versions of this exhaustive search, to be chosen by the user. These two approaches consist of knowing how to find either the optimal encoding functions for a fixed decoding strategy or the optimal decoding functions for a fixed encoding strategy. Both of these approaches provide a computation time reduction when compared with a pure exhaustive search.

`perform_seesaw` implements an iterative two-step see-saw procedure that, starting from a random set of measurements, finds the optimal preparations for the given measurements in the first step, and the optimal measurements for these preparations in the following. The first step involves solving an eigenvalue problem, whereas the second one involves the solution of a semidefinite problem (SDP).

This package accompanies a paper entitled _Biased Random Access Codes_ by Gabriel Pereira Alves, Nicolas Gigena, and Jędrzej Kaniewski available at [arXiv:2302.08494](https://doi.org/10.48550/arXiv.2302.08494).

## Requirements

This script requires the packages `time`, `numpy`, `scipy`, `itertools`, `warnings` and `cvxpy`. While the first packages are required to perform simple calculations, `cvxpy` is the only non-standard package that is needed to solve the SDPs. It comes with some pre-installed solvers, which we have not found particularly reliable. We strongly recommend you install the solver __MOSEK__, which requires a license that can be obtained [here](https://www.mosek.com/products/academic-licenses/).

The packages mentioned above can be installed by running:
```
pip install -r requirements.txt
```

## The `generate_bias` function

Since our interest is to study the quantum and classical value of biased RACs, the main feature of the RAC-tools package is that it allows the user to introduce bias in the RAC functional and use it to compute either the quantum or classical value through the function `perform_seesaw` or just the classical value through the function `perform_search`. One way of doing this is by building a custom bias tensor and passing it as an argument to either of these functions as a Python dictionary. However, as constructing a bias tensor requires some effort, we provide a functionality that allows the user to choose from several simple and natural families of bias tensors, which can take one or more parameters depending on the option. In what follows, we describe in detail the built-in bias options that the user can access via the `generate_bias` function.

The goal of this function, in short, is to construct a properly normalized bias tensor using only a few previously specified parameters. It is not intended to be called by the user, who should in turn specify the parameters defining the desired bias tensor as arguments of the functions `perform_seesaw` and `perform_search`. In order to do so, the value of two variables, `bias` and `weight`, must be specified. The variable `bias` is a string determining the structure of the bias to be generated, whereas the variable `weight` is a float (or a list of floats) that determines the weights given to different terms in the RAC functional.

As an example, we can consider a general version of the `Y_ONE` bias. It consists of a family of bias tensors in which the input strings $\bf x$ are distributed uniformly, but there is bias in Bob's inputs, as one of the characters of $\bf x$, say $x_k$, is requested more (or less) frequently than the others. If we call $w$ the parameter defining how often Bob is asked to recover $x_k$, then the bias tensor takes the form

$$
\alpha_{\mathbf{x} y} =
	\begin{cases}
		\frac{1}{m^n} w \, & \text{if} ~ y = 0, \\
		\frac{1}{m^n}\frac{(1 - w)}{n - 1} & \text{otherwise}.
	\end{cases}
$$

In order to build this bias tensor via the `generate_bias` function, we need to pass as arguments of either `perform_search` or `perform_seesaw`, the following string and float: `bias="Y_ONE"` and  `weight=`$w$. By symmetry, the `Y_ONE` family considers only biasing the first character against the rest, as biasing other values of $y$ produces analogous results. It is possible, nevertheless, to introduce a bias on the frequency with which any of the characters $x_y$ is requested from Bob. This can be done by setting `bias="Y_ALL"` and `weight=List`, where `List` is a list (or a tuple) of floats of length $n$ adding up to one. In this case, the bias tensor obtained from `generate_bias` takes the form

$$
\alpha_{\mathbf{x} y} =  \frac{1}{m^n} w_y,
$$

where $w_y$ is the weight corresponding to the $y$-th character in the input string $\bf x$ and the factor $\frac{1}{m^n}$ results from the input strings $\bf x$ being uniformly distributed.

For introducing biases in the distribution of the input strings, the package offers several one-parameter families, which we enumerate below.

1. `X_ONE`. Analogous to the `Y_ONE` family, it biases the input $\mathbf{x} = 0^{\times n}$ against the $m^n-1$ remaining strings. The user is allowed to define the weight $w$ that will be given to this first input, which will be used to generate a bias tensor of the form $\alpha_{{\bf x} y}=\alpha_{\bf x}\frac{1}{n}$, where

$$
\alpha_{\bf x}=
	\begin{cases}
	    w & \text{if}~{\bf x}=0^{\times n}, \\
        \frac{1 - w}{m^n - 1} &\text{otherwise.}
	\end{cases}
$$

2. `X_DIAG`. This family of biases gives a special weight to input strings of the form $\mathbf{x} = i^{\times n}$, where $i = 0,\, ...,\, m - 1$. Since there are $m$ of these strings, in terms of the parameter $w$ controlled by the user the distribution of input strings takes the form

$$    
\alpha_{\bf x}=
	\begin{cases}
	    \frac{w}{nm} & \text{if}\; {\bf x}=i^{\times n}, \\
        \frac{1}{n}\frac{1 - w}{m^n - m} & \text{otherwise}
	\end{cases}
$$

3. `X_CHESS`. In this case, the input strings are split into two classes depending on whether $\sum x_{j}$ is odd or even. Since the parity of the total number of strings is the same as that of $m$, when the latter is even, half of the strings go into each of the classes defined above. In that case, in terms of the weight $w$ chosen by the user, the ensuing distribution of input strings is given by

$$
\alpha_{\bf x}=
	\begin{cases}
		\frac{2w}{m^n} & \text{if}\; \sum x_{j}\;\text{odd}, \\
		\frac{2(1 - w)}{m^n} & \text{otherwise}.
	\end{cases}
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On the other hand, if $m$ is odd, the number of strings satisfying $\sum x_{j}$ odd is $\frac{m^n - 1}{2}$. In this case, the distribution of input strings reads

$$
\alpha_{\bf x}=
	\begin{cases}
		\frac{2w}{m^n - 1} & \text{if}\; \sum x_{j}\;\text{odd}, \\
		\frac{2(1 - w)}{m^n + 1} & \text{otherwise}.
	\end{cases}
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For $n = 2$, we can think of the elements of $\alpha_{\bf x}$ as the entries of a matrix, in which case the biased elements are arranged in a pattern that resembles a chess board.

4. `X_PLANE`. As before, the idea of this type of bias is to split the set of strings into two classes defined by the condition $x_0=0$. This corresponds to biasing just the first character of the input string. Since there are $m^{n - 1}$ strings satisfying $x_0=0$, in terms of the parameter $w$ the ensuing distribution of input strings reads

$$
\alpha_{\bf x} =
\begin{cases}
\frac{w}{m^{n - 1}} & \text{if}\; x_{0}=0, \\
\frac{1 - w}{m^n - m^{n - 1}} & \text{otherwise}.
\end{cases}
$$

All the biases introduced so far depend only on $\bf x$ or only on $y$. In the next step, we could take one bias of each kind and combine them, which would lead to a product distribution over $\bf x$ and $y$. However, there is no reason why we should restrict ourselves to product distributions. In the following, we introduce two cases referred to as `B_ONE` and `B_ALL` biases which correspond to non-factorizable distributions of inputs $\mathbf{x}$ and $y$. Since $b=x_y$ is the required condition to have successful decoding, we consider the case where different weights are attributed to the outputs of Bob. The first bias of this kind, `bias=B_ONE`, corresponds to biasing the first output of Bob, $b=0$, against the remaining $m-1$ outputs

$$
\alpha_{\mathbf{x} y} =
	\begin{cases}
		\frac{1}{n} \frac{1}{m^{n - 1}} \, w \, & \text{if} ~ x_y = 0, \\
		\frac{1}{n} \frac{1}{m^{n - 1}}\frac{(1 - w)}{m - 1} & \text{otherwise}.
	\end{cases}
$$

If a more general bias of the outputs is required, then the user can enter `bias="B_ALL"` as an option, in which case they should input as weight a python list (or tuple). The `generate_bias` function will then output a bias tensor of the form

$$
\alpha_{\mathbf{x} y} =
	\frac{1}{n} \frac{1}{m^{n - 1}}  \, w_{x_y},
$$

where $w_{x_y}$ is the weight on the character $x_y$ -- and consequently on the $b$-th output of Bob since $b=x_y$. For the $b$-bias cases, $\frac{1}{n}\frac{1}{m^{n-1}}$ is the normalization factor.

## The `perform_search` function

The goal of `perform_search` is to exactly compute the best classical performance of a given $n^m \smash{\overset{d}{\mapsto}} 1$ RAC. The function can perform this computation either via a pure exhaustive search through the deterministic strategies of a given RAC scenario or by means of two less expensive approaches. The first of these approaches consists of an exhaustive search through the encoding functions. Once an encoding function is fixed, it is possible to determine analytically the optimal decoding function. Thus, an exhaustive search only through the encoding strategies is enough to compute the best classical performance. The second approach is the inverse of the latter: first, the decoding function is fixed and then the optimal encoding is computed.

To operate `perform_search` it is enough to specify in its argument the three integers defining the scenario, $n$, $d$ and $m$, and the search method. The latter can be introduced by declaring either `method=0` for the pure exhaustive search, `method=1` for the search over decoding functions, or `method=2` for the search over encoding functions, which is the default method. Furthermore, the value of $m$ is set by default to coincide with that of $d$, so that the user is not expected to declare it unless these numbers are different.

An example of how this function operates can be seen in the report below, in which the user desires to estimate the classical value of the $2^2 \mapsto 1$ unbiased RAC. The function is therefore called passing as arguments $n=2$, $d=2$ and \verb+method=0+, and once the procedure is finished the report is printed. The _Summary of computation_ section of the report informs the user of the total time of computation as well as the total number of encoding and decoding functions analyzed for the chosen search method, which in this case corresponds to the total number of combinations of encoding and decoding functions, i.e., $d^{m^n} \times m^{dn}$. In addition, the average time taken to iterate over each function (or combination of encoding and decoding functions, if `method=0`) is displayed at _Average time per function_.

```
> perform_search(n=2, d=2, method=0)

================================================================================
                                 RAC-tools v1.0
================================================================================

---------------------------- Summary of computation ----------------------------

Total time of computation: 0.001882 s
Total number of encoding/decoding functions: 256
Average time per function: 7e-06 s

--------------------- Analysis of the optimal realization ----------------------

Computation of the classical value for the 2²-->1 RAC: 0.75
Number of functions achieving the computed value: 24

First functions found achieving the computed value
Encoding:
E: [0, 0, 0, 1]
Decoding:
D₀: [0,  1]
D₁: [0,  1]

------------------------------ End of computation ------------------------------
```

In the second part of the report, the user can see the computed classical value and the number of functions that achieve this value. In addition, the report provides the user with a particular pair of encoding and decoding strategies which attain the optimal value. For the encoding function $E(\mathbf{x})$, the result is displayed in a tuple that is organized in ascending order of $\mathbf{x}$, i.e., $[E(00...0),~ E(00...1),~ ...,~ E((m-1)...(m-1))]$. For the decoding functions $D_y(\mu)$, each row corresponds to a distinct input $y$ of Bob and it is organized in ascending order of $\mu$.

## The `perform_seesaw` function

This function implements the see-saw algorithm described in _Biased Random Access Codes_ [arXiv:2302.08494](https://doi.org/10.48550/arXiv.2302.08494), and its goal is to provide lower bounds to the quantum value of a given $n^m \smash{\overset{d}{\mapsto}} 1$ $b$-RAC. Here we will not provide details about the see-saw algorithm itself, but technical details concerning the operation of `perform_seesaw`.

As is the case with `perform_search`, the `perform_seesaw` function takes as argument the integers defining the scenario, $n$, $d$ and $m$ and the bias tensor, either as a dictionary or via one of the aforementioned built-in options. The user is also asked to pass as an argument the number of starting points for the algorithm by means of the variable `seeds`. Moreover, it is possible to use this function to compute a lower bound to the classical value, by means of the variable `diagonal`. If `diagonal=True`, the function initialize the see-saw algorithm with random diagonal measurements, and the optimization is then restricted to operators which are diagonal in the computational basis. By default `diagonal=False`, and the algorithm optimizes the functional value over POVM measurements.

When called, `perform_seesaw` runs the see-saw algorithm as many times as the number of seeds specified by the user, generating a lower bound to the quantum value per starting point. The best value is therefore the largest among all these lower bounds, implying that the chances of the function providing the actual quantum value of the $b$-RAC increase with the number of seeds, as well as the computation time.

Because the see-saw algorithm is iterative, convergence criteria must be adopted to decide whether the optimal value for a given seed has been attained after a particular number of steps. In the `perform_seesaw` implementation of this algorithm, we impose two convergence criteria, and the procedure is finished whenever the two are satisfied. The first criterion is related to the convergence of the $\mathcal{F}$ value. It is satisfied whenever the difference between two consecutive evaluations of $\mathcal{F}$ is smaller than a value that can be set by the user via the variable `prob_bound`. The default value of this variable is set to $10^{-9}$. The second stopping criterion considers the convergence of the measurements, and it focuses on the distance between the optimal measurement operators in two consecutive iterations of the algorithm. More precisely, we will say that the measurements converged if the following condition

$$
\max_{y, b} \big|\big| M_y^b - N_y^b \big|\big| < t
$$

is satisfied, where $|| \cdot ||$ is the Frobenius norm, and $N_y^b$ and $M_y^b$ denote two consecutive measurement operators associated with the same value of the $y$-th character of the input $\bf x$. The constant $t$ is a threshold that can be defined by the user via the variable `meas_bound`, which as a default takes the value $10^{-7}$. For the evaluation of the condition above, we use the function `norm`, from `numpy.linalg`, to implement the Frobenius norm.

The value of both, `prob_bound` and `meas_bound`, can be passed as an argument to `perform_seesaw`. In addition to the convergence criteria, we have imposed a limit to the number of iterations to be executed by the algorithm, so that if after 200 iterations either the $\mathcal{F}$-value or the measurements fail to converge, the calculation stops. In this case, the message _maximum number of iterations reached_ is displayed as a warning. This limit can be modified by entering a different value to the variable `max_iterations` in the argument of the function.

An example of the operation of `perform_seesaw` can be seen in the report below, in which the user wants to estimate the quantum value of the $2^2 \mapsto 1$ unbiased RAC. As in the case of `perform_searc`, the user passes as arguments $n=2$ and $d=2$ to define the scenario, but now instead of choosing a search method the user introduces the number of starting points to be used by passing `seeds=5`. After finishing the procedure, the function prints a report divided into two parts. The _Summary of the computation_ presents the number of random starting points, the average processing time and the average number of iterations among all starting points. In addition, it shows how many starting points produced an optimal value that is close to the largest value obtained. The interval to consider two values produced by different starting points as close is the accuracy of the solver __MOSEK__, which is set to $10^{-13}$. This informs the user how frequent it is to obtain such an estimation; if this number is much smaller than `seeds`, this indicates that the user should increase the number of starting points in case of a new execution.  

```
> perform_seesaw(n=2, d=2, seeds=5)  

================================================================================
                                 RAC-tools v1.0
================================================================================

---------------------------- Summary of computation ----------------------------

Number of random seeds: 5
Average time for each seed: 0.1586 s
Average number of iterations: 3
Seeds 1e-13 close to the best value: 5

--------------- Analysis of the optimal realization for seed #3 ----------------

Estimation of the quantum value for the 2²-->1 RAC: 0.853553390593

Measurement operator ranks
M[0] ranks:  1  1
M[1] ranks:  1  1

Measurement operator projectiveness
M[0, 0]:  Projective  6.63e-15
M[0, 1]:  Projective  6.64e-15
M[1, 0]:  Projective  6.36e-15
M[1, 1]:  Projective  6.36e-15

Mutual unbiasedness of measurements
M[0] and M[1]:  MUB  4.28e-14

------------------------------ End of computation ------------------------------
```

In the second part, the estimation of the optimal value is reported, followed by information about the set of measurements attaining such value. Note that the reported value matches the one found by Ref. [1]. Next, the report displays the rank of the optimal measurement operators, which is computed using the function `matrix_rank` of `numpy.linalg`. In addition, the user can check whether the measurement operators are projective. The number shown in the second column of _Measurement operator projectiveness_ corresponds to the quantity

$$
\big|\big|(M_y^b)^2 - M_y^b \big|\big|.
$$

For both of these checks, rank and projectiveness, we preset a tolerance of $10^{-7}$.

Lastly, in the case where at least two measurements are rank-one and projective, the function also computes whether each pair of measurements can be constructed out of Mutually Unbiased Bases (MUB). For a pair of rank-one projective measurements, let us say $\{P^a\}_{a=0}^{m-1}$ and $\{Q^b\}_{b=0}^{m-1}$, where $a$ and $b$ denote the $a$-th and the $b$-th outcome, it is enough [3, App. B] to check if

$$
P^a = m \, P^a Q^b P^a ~ \text{and} ~ Q^b = m \, Q^b P^a Q^b,
$$

for all $a,\, b \in \{0,\, 1\, ...,\, m-1\}$.  In this case, the number displayed in the second column of _Mutual unbiasedness of measurements_ represents the measure

$$
\max_{a, b} \left( || m\, P^a Q^b P^a - P^a ||, || m \, Q^b P^a Q^b - Q^b || \right).
$$

For the cases in which this measure is lower than `MUB_BOUND=5e-6`+, the function prints `MUB`. Otherwise, it simply displays `Not MUB`.

## References

1. A. Ambainis, D. Leung, L. Mancinska and M. Ozols, Quantum Random Access Codes with Shared Randomness, available in [arXiv:0810.2937](https://arxiv.org/abs/0810.2937).

2. M. Farkas and J. Kaniewski, Self-testing mutually unbiased bases in the prepare-and-measure scenario, [Phys. Rev. A  **99**, 032316 (2019)](https://link.aps.org/doi/10.1103/PhysRevA.99.032316).

3. A. Tavakoli, M. Farkas, D. Rosset, J.-D. Bancal and J. Kaniewski, Mutually unbiased bases and symmetric informationally complete measurements in Bell experiments, [Sci. Adv. **7** (2021)](https://doi.org/10.1126/sciadv.abc3847).
