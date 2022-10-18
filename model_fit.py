#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:01:55 2022
@author: urai

see https://codeocean.com/capsule/0617593/tree/v1
"""


import numpy as np
import autograd.numpy.random as npr


from ssm.lds import LDS
from ssm.emissions import _LinearEmissions, _BernoulliEmissionsMixin#, BernoulliEmissions
from ssm.observations import AutoRegressiveObservations
from ssm.preprocessing import pca_with_imputation, interpolate_data
from ssm.transitions import StationaryTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.util import ensure_args_are_lists


# In this script we adjust the original code of the ssm package to fit 
# a Linear Dynamical System (LDS) with Bernoulli emissions and latent slow drifts
# modeled with either (1) a random walk or (2) AR(1) process where phi is allowed to differ from 1


#%% fit with SSM package

# Model structure latent slow drifts
class AutoRegressiveNoInput(AutoRegressiveObservations):
    
    # Code used by Gupta for estimation of sigma_d in the random walk
    
    # Original code can be found in ssm.observations line 1113
    # Rewritten because if you would initialize parameter V as being 0,
    # the estimation would overwrite this and still give an estimate for the parameter
    # This can be checked by commenting out this class and change 'dynamics = AutoRegressiveNoInput(1, D, M=M)' below
    # to 'dynamics = AutoRegressiveObservations(1, D, M=M)' to check this
    
    
    # def m_step(self, expectations, datas, inputs, masks, tags,
    #                 continuous_expectations=None, **kwargs):
        
    #     K, D, M, lags = self.K, self.D, self.M, self.lags

    #     # Collect sufficient statistics
    #     if continuous_expectations is None:
    #         ExuxuTs, ExuyTs, EyyTs, Ens = self._get_sufficient_statistics(expectations, 
    #                                                                         datas, 
    #                                                                         inputs)
    #     else:
    #         ExuxuTs, ExuyTs, EyyTs, Ens = \
    #               self._extend_given_sufficient_statistics(expectations, 
    #                                                       continuous_expectations, 
    #                                                       inputs)

    #     Sigmas = np.zeros((K, D, D))
    #     for k in range(K):
    #           Wk = np.linalg.solve(ExuxuTs[k] + self.J0[k], ExuyTs[k] + self.h0[k]).T

    #         # Solve for the MAP estimate of the covariance
    #           EWxyT =  Wk @ ExuyTs[k]
    #           sqerr = EyyTs[k] - EWxyT.T - EWxyT + Wk @ ExuxuTs[k] @ Wk.T
    #           nu = self.nu0 + Ens[k]
    #           Sigmas[k] = (sqerr + self.Psi0) / (nu + D + 1)

    #     # If any states are unused, set their params to a perturbation of a used state
    #     unused = np.where(Ens < 1)[0]
    #     used = np.where(Ens > 1)[0]
    #     if len(unused) > 0:
    #           for k in unused:
    #               i = np.choice(used)
    #               Sigmas[k] = Sigmas[i]

    #     # Update params via their setter
    #     self.Sigmas = Sigmas
 
    
    # Changed the code to allow the estimation of the AR coefficient
    # Basically, if a parameter shouldn't be estimated, comment out the parameter
    # below in the section 'Update params via their setter'
        
        
    def m_step(self, expectations, datas, inputs, masks, tags,
                continuous_expectations=None, **kwargs):
        """Compute M-step for Gaussian Auto Regressive Observations.
    
        If `continuous_expectations` is not None, this function will
        compute an exact M-step using the expected sufficient statistics for the
        continuous states. In this case, we ignore the prior provided by (J0, h0),
        because the calculation is exact. `continuous_expectations` should be a tuple of
        (Ex, Ey, ExxT, ExyT, EyyT).
    
        If `continuous_expectations` is None, we use `datas` and `expectations,
        and (optionally) the prior given by (J0, h0). In this case, we estimate the sufficient
        statistics using `datas,` which is typically a single sample of the continuous
        states from the posterior distribution.
        """
        K, D, M, lags = self.K, self.D, self.M, self.lags
    
        # Collect sufficient statistics
        if continuous_expectations is None:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._get_sufficient_statistics(expectations, datas, inputs)
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = \
                self._extend_given_sufficient_statistics(expectations, continuous_expectations, inputs)
    
        # Solve the linear regressions
        As = np.zeros((K, D, D * lags))
        Vs = np.zeros((K, D, M))
        bs = np.zeros((K, D))
        Sigmas = np.zeros((K, D, D))
        for k in range(K):
            Wk = np.linalg.solve(ExuxuTs[k] + self.J0[k], ExuyTs[k] + self.h0[k]).T
            As[k] = Wk[:, :D * lags]
            Vs[k] = Wk[:, D * lags:-1]
            bs[k] = Wk[:, -1]
    
            # Solve for the MAP estimate of the covariance
            EWxyT =  Wk @ ExuyTs[k]
            sqerr = EyyTs[k] - EWxyT.T - EWxyT + Wk @ ExuxuTs[k] @ Wk.T
            nu = self.nu0 + Ens[k]
            Sigmas[k] = (sqerr + self.Psi0) / (nu + D + 1)
    
        # If any states are unused, set their params to a perturbation of a used state
        unused = np.where(Ens < 1)[0]
        used = np.where(Ens > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = np.choice(used)
                As[k] = As[i] + 0.01 * npr.randn(*As[i].shape)
                Vs[k] = Vs[i] + 0.01 * npr.randn(*Vs[i].shape)
                bs[k] = bs[i] + 0.01 * npr.randn(*bs[i].shape)
                Sigmas[k] = Sigmas[i]
    
        # Update params via their setter
        #self.As = As
        #self.Vs = Vs
        #self.bs = bs
        self.Sigmas = Sigmas



# # Model structure Bernoulli emissions
class _LinearEmission(_LinearEmissions):
    
    @property
    def params(self):
        return self.Cs, self.Fs, self.ds     # original code
        #return self.Fs, self.ds
        #return self.Fs, self.Cs
        #return self.Fs

    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value    # original code
        #self.Fs, self.Cs = value
        #self.Fs, self.ds = value
        #self.Fs = value
    
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data with the maximum effective dimension
        pca, xs, ll = pca_with_imputation(min(self.D * Keff, self.N),
                                          resids, masks, num_iters=num_iters)

        # Assign each state a random projection of these dimensions
        Cs, ds = [], []
        for k in range(Keff):
            weights = npr.randn(self.D, self.D * Keff)
            weights = np.linalg.svd(weights, full_matrices=False)[2]
            Cs.append((weights @ pca.components_).T)
            ds.append(pca.mean_)

        # Find the components with the largest power
        self.Cs = np.array(Cs)
        self.ds = np.array(ds)

        return pca


# Original code from the package but needed here to run the model
class BernoulliEmission(_BernoulliEmissionsMixin, _LinearEmission):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess
    
    
    
class LDS_noInputDynamics(LDS):
    def __init__(self, N, D, M=0):
    
        dynamics = AutoRegressiveNoInput(1, D, M=M)
        emissions = BernoulliEmission(N, 1, D, M=M)

        init_state_distn = InitialStateDistribution(1, D, M)
        transitions = StationaryTransitions(1, D, M)
        
        self.N, self.K, self.D, self.M = N, 1, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions


#%%
def initLDSandFit(inputDim, inputs, emissions,n_iters):
    stateDim = 1
    lds = LDS_noInputDynamics(1, 1, M = inputDim)
    
    # note: everything that is not specified (in this case Cs, d, F, and sigma) is fitted
    
    lds.dynamics.A = np.ones((stateDim,stateDim))           # dynamics
    lds.dynamics.b = np.zeros(stateDim)                     # bias
    lds.dynamics.mu_init = np.zeros((stateDim,stateDim))    # initial mu
    lds.dynamics.Sigmas_init = np.array([[[0.01]]])         # initial sigma
    lds.dynamics.Vs = np.array([[np.zeros(inputDim)]])      # input dynamics
    

    #lds.emissions.Cs = np.array([[[1]]])
    #lds.emissions.ds = np.array([[[0]]]) # if simulation is not effect coded ds = 0 is huge bias! should be -sens/2
    #lds.emissions.Fs = np.array([[np.zeros(inputDim)]])
    
    
    # if model shouldn't fit slow drift then: 
    # lds.dynamics.Sigmas_init = np.array([[[0]]])         # initial sigma
    # lds.dynamics.Sigmas = np.zeros(lds.dynamics.Sigmas.shape)


    elbos, q = lds.fit(emissions, inputs = inputs, method="laplace_em",
                        variational_posterior="structured_meanfield", 
                        continuous_optimizer='newton',
                        initialize=True, 
                        num_init_restarts=1,
                        num_iters=n_iters, 
                        alpha=0.1)

    # Get the posterior mean of the continuous states (drift)
    state_means = q.mean_continuous_states[0]
    estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]

    # Smooth the data under the variational posterior (to compute emissions)
    predEmissions = np.concatenate(lds.smooth(state_means, emissions, input = inputs))
       
    return predEmissions, estDrift, lds, q, elbos
