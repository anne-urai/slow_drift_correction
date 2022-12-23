# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:35:41 2022

@author: u0141056
"""


import scipy.stats as spstats
import numpy as np
K, D, M, lags = 1,1,7,1
mean_A = np.ones((K, D, D * lags))
mean_V = np.ones((K, D, M))
mean_b = np.ones((K, D))

As = spstats.norm.rvs(mean_A, np.sqrt(.1))
Vs = spstats.norm.rvs(mean_V, np.sqrt(.1))
bs = spstats.norm.rvs(mean_b, np.sqrt(.1))
mu_init = np.zeros((K, D))
As, bs, Vs, mu0s = As, bs, Vs, mu_init
Vs.shape
# As = np.mean(self.As)
# bs = np.mean(self.bs)
# Vs = np.mean(self.Vs)
# mu0s = np.mean(self.mu_init)

# Instantaneous inputs
# mus = np.empty((K, T, D))
# mus = []
zip(Vs, mu0s)
zip(As)
for k, (V, mu0) in enumerate(zip(Vs, mu0s)):
    print("bla")
    # Initial condition
    mus_k_init = mu0 * np.ones((self.lags, D))

    # Subsequent means are determined by the AR process
    mus_k_ar = np.dot(input[self.lags:, :M], V.T)
    for l in range(self.lags):
        Al = A[:, l*D:(l + 1)*D]
        mus_k_ar = mus_k_ar + np.dot(data[self.lags-l-1:-l-1], Al.T)
    mus_k_ar = mus_k_ar + b