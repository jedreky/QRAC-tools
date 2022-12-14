QRACs
-----
<div align="justify">
A Quantum Random Access Code, mostly known by its acronym QRAC, is a task in which n dits are encoded into a qudit. The aim of this task is to recover one of the dits chosen uniformly at random. A QRAC is generally represented by its parameters n<sup>d</sup> &#8594; 1, in which n is the number of dits, and d is the size of the dits, as well as the dimension of the qudit.

This program encounters the maximum value of a QRAC by performing a _see-saw_ optimization. Encountering the maximum value of a QRAC means that we aim to maximize the functional associated with it.

The main function of this code is implemented in 'find_QRAC_value', in which it is possible to input n, d, and the number of seeds to be used in the see-saw optimization. In order to run this script you need to have `numpy`, `scipy`, and `cvxpy` installed. Moreover, while `cvxpy` comes with some pre-installed solvers, we have not found them particularly reliable. We strongly recommend that you install and solver __MOSEK__, which requires a license that can be obtained [here](https://www.mosek.com/products/academic-licenses/).
  
The packages mentioned above can be installed by running:

```
pip install -r requirements.txt
```
  
</div>

Reference
----------
1. A. Ambainis, D. Leung, L. Mancinska and M. Ozols, Quantum Random Access Codes with Shared Randomness, available in [arXiv:0810.2937](https://arxiv.org/abs/0810.2937).
