#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: gof.py
"""
Created on Thu Sep 27 22:58:59 2018

@author: Neo(liuniu@smail.nju.edu.cn)
"""


from scipy.special import gammainc


# -----------------------------  FUNCTIONS -----------------------------
def gof(obs_num, fdm_num, chi2):
    """Calculate the goodness-of-fit.

    The formula is expressed as below.

    Q = gammq((obs_num - fdm_num) / 2, chi2 / 2). (Numerical Recipes)

    gammq is the incomplete gamma function.

    Parameters
    ----------
    fdm_num : int
        number of freedom
    chi2 : float
        chi square

    Return
    ------
    Q : float
        goodness-of-fit
    """

    Q = gammainc((obs_num - fdm_num)/2., chi2/2.)

    return Q

# --------------------------------- END --------------------------------
