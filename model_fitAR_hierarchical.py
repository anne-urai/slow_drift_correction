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
from ssm import hierarchical
import copy

from autograd.scipy.stats import norm
from autograd.misc.optimizers import sgd, adam
from autograd import grad


# In this script we adjust the original code of the ssm package to fit 
# a Linear Dynamical System (LDS) with Bernoulli emissions and latent slow drifts
# modeled with either (1) a random walk or (2) AR(1) process where phi is allowed to differ from 1


#%% fit with SSM package

# Model structure latent slow drifts
class AutoRegressiveNoInput(AutoRegressiveObservations):
    
 
    def m_step(self, expectations, datas, inputs, masks, tags,
                continuous_expectations=None, **kwargs):

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
        self.As = As
        #self.Vs = Vs
        #self.bs = bs
        self.Sigmas = Sigmas



# # Model structure Bernoulli emissions
class _LinearEmission(_LinearEmissions):
    
    @property
    def params(self):
        return self.Fs

    @params.setter
    def params(self, value):
        self.Fs = value
    
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
        #self.Cs = np.array(Cs)
        #self.ds = np.array(ds)

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
    
    
class _Hierarchical(AutoRegressiveNoInput):
    """
    Base class for hierarchical models.  Maintains a parent class and a
    bunch of children with their own perturbed parameters.
    """
    def __init__(self, base_class, *args, tags=(None,), lmbda=0.01, **kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda

        # Top-level parameters (parent)
        self.parent = base_class(*args, **kwargs)

        # Make models for each tag
        self.tags = tags
        self.children = dict()
        for tag in tags:
            ch = self.children[tag] = base_class(*args, **kwargs)
            ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

    @property
    def params(self):
        prms = (self.parent.params,)
        for tag in self.tags:
            prms += (self.children[tag].params,)
        return prms

    @params.setter
    def params(self, value):
        self.parent.params = value[0]
        for tag, prms in zip(self.tags, value[1:]):
            self.children[tag].params = prms

    def permute(self, perm):
        self.parent.permute(perm)
        for tag in self.tags:
            self.children[tag].permute(perm)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        for tag in self.tags:
            self.children[tag].params = copy.deepcopy(self.parent.params)

    def log_prior(self):
        lp = self.parent.log_prior()

        # Gaussian likelihood on each child param given parent param
        for tag in self.tags:
            for pprm, cprm in zip(self.parent.params, self.children[tag].params):
                lp += np.sum(norm.logpdf(cprm, pprm, self.lmbda))
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        # Optimize parent and child parameters at the same time with SGD
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints) \
                in zip(datas, inputs, masks, tags, expectations):

                if hasattr(self.children[tag], 'log_initial_state_distn'):
                    log_pi0 = self.children[tag].log_initial_state_distn(data, input, mask, tag)
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(self.children[tag], 'log_transition_matrices'):
                    log_Ps = self.children[tag].log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)

                if hasattr(self.children[tag], 'log_likelihoods'):
                    lls = self.children[tag].log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = \
            optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)

class HierarchicalObservations(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag, with_noise=with_noise)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(expectations, data, input, tag)
    
    
class LDS_noInputDynamics(LDS):
    def __init__(self, N, D, M=0):
        
        
        dynamics = HierarchicalObservations(AutoRegressiveNoInput,1,D)
        emissions = BernoulliEmission(N, 1, D, M=M)

        init_state_distn = InitialStateDistribution(1, D, M)
        transitions = StationaryTransitions(1, D, M)
        
        self.N, self.K, self.D, self.M = N, 1, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions


#%%
def initLDSandFitAR(inputDim, inputs, emissions,n_iters):
    stateDim = 1
    lds = LDS_noInputDynamics(1, 1, M = inputDim)
    
    AutoRegressiveNoInput.b = np.zeros(stateDim)
    AutoRegressiveNoInput.mu_init = np.zeros((stateDim,stateDim))
    AutoRegressiveNoInput.Sigmas_init = np.array([[[0.01]]])
    AutoRegressiveNoInput.Vs = np.array([[np.zeros(inputDim)]])
    
    #lds.dynamics.b = np.zeros(stateDim)                     # bias
    #lds.dynamics.mu_init = np.zeros((stateDim,stateDim))    # initial mu
    #lds.dynamics.Sigmas_init = np.array([[[0.01]]])         # initial sigma
    #lds.dynamics.Vs = np.array([[np.zeros(inputDim)]])      # input dynamics
    

    lds.emissions.Cs = np.array([[[1]]])
    lds.emissions.ds = np.array([[[0]]]) # if simulation is not effect coded ds = 0 is huge bias! should be -sens/2
    #lds.emissions.Fs = np.array([[np.zeros(inputDim)]])
    
    
    # if model shouldn't fit slow drift then: 
    # lds.dynamics.Sigmas_init = np.array([[[0]]])         # initial sigma
    # lds.dynamics.Sigmas = np.zeros(lds.dynamics.Sigmas.shape)


    elbos, q = lds.fit(emissions, inputs = inputs, method="laplace_em",
                        variational_posterior="structured_meanfield", 
                        continuous_optimizer='newton',
                        initialize=False, 
                        num_init_restarts=1,
                        num_iters=n_iters, 
                        alpha=0.1)

    # Get the posterior mean of the continuous states (drift)
    state_means = q.mean_continuous_states[0]
    estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]

    # Smooth the data under the variational posterior (to compute emissions)
    predEmissions = np.concatenate(lds.smooth(state_means, emissions, input = inputs))
       
    return predEmissions, estDrift, lds, q, elbos
