#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: linear_fit.py
"""
Created on Thu Mar 28 10:40:19 2019

@author: Neo(liuniu@smail.nju.edu.cn)
"""


import numpy as np
from numpy import sqrt
from scipy.optimize import curve_fit

__all__ = ["linear_func", "linear_fit"]


# -----------------------------  FUNCTIONS -----------------------------
def linear_func(x, a, b):
    return a + b * x


def linear_fit(x, y, yerr=None, return_mod=False):
    """(Weighted) Linear fitting of y(x) = offset + drift * x

    Parameters
    ---------
    x / y : series

    Returns
    -------
    offset/offset_err : float
        estimate and formal uncertainty of offset
    drift/drift_err : float
        estimate and formal uncertainty of offset
    y_model : array_like of float
        predicition series from linear model of y
    """

    if yerr is None:
        popt, pcov = curve_fit(linear_func, x, y)
    else:
        popt, pcov = curve_fit(
            linear_func, x, y, sigma=yerr, absolute_sigma=True)

    offset, drift = popt
    offset_err, drift_err = sqrt(pcov[0, 0]), sqrt(pcov[1, 1])
    corr = pcov[0, 1] / offset_err / drift_err

    # Prediction
    y_model = linear_func(x, *popt)

    if return_mod:
        return offset, offset_err, drift, drift_err, corr, y_model
    else:
        return offset, offset_err, drift, drift_err, corr


# --------------------------------- END --------------------------------
