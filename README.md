QRACs
-----
A Quantum Random Access Code, mostly known by its acronym QRAC, is a task in which n dits are encod-
ed into a qudit. The aim of this task is to recover one of the dits chosen uniformly at random. A
QRAC is generally represented by its parameters nˆ(d) --> 1, in which n is the number of dits, and d
is the size of the dits, as well as the dimension of the qudit.

This program encounters the maximum value of a QRAC by performing a see-saw optimization. Encounter-
ing the 'maximum value' of a QRAC means that we aim to maximize the functional associated with it.

The main function of this code is implemented in find_QRAC_value(n, d, seeds), in which it is possi-
ble to input n, d, and the number of seeds to be used in the see-saw optimization.

Reference
----------
1. A. Ambainis, D. Leung, L. Mancinska and M. Ozols, Quantum Random Access Codes with Shared Random-
ness, available in arXiv:0810.2937.
