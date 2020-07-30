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

__all__ = ["linear_fit"]


# -----------------------------  FUNCTIONS -----------------------------
def linear_func(x, x0, x1):
    return x0 + x1 * x


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
    return_mod :
        flag to determine whether to return the precition, decrepanced.
    """

    if yerr is None:
        popt, pcov = curve_fit(linear_func, x, y)
    else:
        popt, pcov = curve_fit(
            linear_func, x, y, sigma=yerr, absolute_sigma=False)

    # Create a `dict` object to store the results
    res = {}

    # Estimate, `x0` for interception and `x1` for slope
    res["x0"], res["x1"] = popt

    # Formal error and correlation coefficient
    res["x0_err"], res["x1_err"] = sqrt(pcov[0, 0]), sqrt(pcov[1, 1])
    res["corr"] = pcov[0, 1] / offset_err / drift_err

    # Prediction
    res["predict"] = linear_func(x, *popt)

    # Residual
    res["residual"] = y - res["predict"]

    # Chi squared per dof
    if yerr is None:
        res["chi2_dof"] = np.sum(res["residual"]) / (len(y) - 2)
    else:
        res["chi2_dof"] = np.sum(res["residual"] / yerr) / (len(y) - 2)

    # if return_mod:
    #     return offset, offset_err, drift, drift_err, corr, y_model
    # else:
    #     return offset, offset_err, drift, drift_err, corr

    return res


# --------------------------------- END --------------------------------
