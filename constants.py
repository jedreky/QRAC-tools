#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
"""This file contains a list of constants to be used by utils.py"""
# ==================================================================================================

# Threshold for the problem value. The problem value converges much faster than the variables, so we
# can allow a tighter tolerance for the problem value as low as PROB_BOUND.
PROB_BOUND = 1e-9

# BOUND is used as tolerance to check projectivity, rankness, etc, of the measurement operators. It
# is also used as the default value for the tolerance in the variables of the problem.
BOUND = 1e-7

# MUB_BOUND is used as a tolerance in the function check_if_MUBs. This function is less accurate
# than the other checks. Our analycal results allow us to trust in a looser tolerance for this par-
# ticular check.
MUB_BOUND = 5e-6

# This is the solver's maximum accuracy. By default, we set it to be lower than PROB_BOUND and
# BOUND. Making this number tighter might compromise the feasibility of the problems.
MOSEK_ACCURACY = 1e-13

# Maximal number of iterations for the optimization problem.
ITERATIONS = 100

# Total number of attempts to generate a valid measurement.
MEAS_ATTEMPTS = 10

# Total number of attempts to solve an SDP.
SOLVE_ATTEMPTS = 20
