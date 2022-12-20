# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:49:01 2022

@author: u0141056
"""


# see https://github.com/lindermanlab/ssm/blob/master/notebooks/1b%20Simple%20Linear%20Dynamical%20System.ipynb
# see also https://github.com/lindermanlab/ssm/blob/master/ssm/lds.py#L825

# K = 1       # number of discrete states: 1, no switching
# D = 1       # number of latent dimensions: 1, AR model
# N = 1       # number of observed dimensions: only choices. Could add in RT/confidence?
# M = 3       # input dimensions

# To do:
# check n_iters
# random restarts
# fix parameters to certain values
# check update hierarchical prior
# allow different prior variance for Fs[0] (stimulus, perceptual sensitivity)?


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import simulate_choices
import ssm

ntrials = 50000 # original code: 40.000
nsubjects = 2
sens = 10
bias = 0 # is unbiased if effect coded simulations 0 is unbiased!
σd = 0.05


w_prevresp = 1
w_prevconf = 1
w_prevrespconf = 1
w_prevsignevi = 1
w_prevabsevi = 1
w_prevsignabsevi = 1


# a hierarchical model needs its input organised in a tuple, with subsets for each subject, and a separate variable indicating the tags
# structure derived from: https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/notebooks/8%20Hierarchical%20ARHMM%20Demo.ipynb
# see file test_ARHMM.py
# see if simulated data below can be estimated with HMM to rule out problem in simulated data structure: CHECK!


t_inputs = () # empty tuple
t_choices = ()
t_drift = ()
t_tags = ()

for sub in range(nsubjects):
    d_inputs, choices, drift = simulate_choices.simulateChoice_normalEvi_slowdriftConf(ntrials,
                                              estimateUpdating = True,
                                              fixedConfCriterion = False,
                                              postDecisionalEvi = True,
                                              σd = σd, 
                                              sens = sens, bias = bias, 
                                              w_prevresp = w_prevresp, 
                                              w_prevconf = w_prevconf,
                                              w_prevrespconf = w_prevrespconf,
                                              w_prevsignevi = w_prevsignevi,
                                              w_prevabsevi = w_prevabsevi,
                                              w_prevsignabsevi = w_prevsignabsevi,
                                              seed = sub) # if a constant then same drift for each person
    
    t_inputs = t_inputs + (d_inputs,)
    t_choices = t_choices + (choices,)
    t_drift = t_drift + (drift,)
    t_tags = t_tags + (sub,)
    
    
inputDim = np.shape(d_inputs)[1] # observed input dimensions 
stateDim = 1 # latent states
lags = 1
n_iters = 50



lds = ssm.LDS(1,1, M = inputDim, dynamics="hierarchical_ar",
                              dynamics_kwargs=dict(lags=lags, tags=t_tags),
                              emissions="hierarchical_bernoulli",
                              emission_kwargs=dict(tags=t_tags))

# lds outputs epoch_lps, inner_lps 
# see lds.py
# see https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/notebooks/8%20Hierarchical%20ARHMM%20Demo.ipynb how to use this for picking best model
# with multiple restarts
lps = lds.fit(datas = t_choices,
                    inputs = t_inputs, 
                    masks = None,
                    method = "laplace_em",
                    num_iters = n_iters,
                    initialize = False,
                    tags = t_tags)

# AR dynamics
tag = 0

lds.dynamics.global_ar_model.As
lds.dynamics.global_ar_model.Vs
lds.dynamics.global_ar_model.bs
lds.dynamics.global_ar_model.Sigmas

# per subject (tags)
lds.dynamics.per_group_ar_models[tag].As
lds.dynamics.per_group_ar_models[tag].Vs
lds.dynamics.per_group_ar_models[tag].bs
lds.dynamics.per_group_ar_models[tag].Sigmas



# Bernoulli emissions
lds.emissions.global_bernoulli_model.Cs
lds.emissions.global_bernoulli_model.Fs
lds.emissions.global_bernoulli_model.ds

# per subject (tags)
lds.emissions.per_group_bernoulli_models[tag].Cs
lds.emissions.per_group_bernoulli_models[tag].Fs
lds.emissions.per_group_bernoulli_models[tag].ds



# Variational Posterior allows estimation drift!!!!

# still need to fix this!
# see variational.py line 343
# try in a for 
from ssm.variational import SLDSStructuredMeanFieldVariationalPosterior

q_full = SLDSStructuredMeanFieldVariationalPosterior(
       lds,
       datas=[data for data in t_choices],
       inputs=[inpt for inpt in t_inputs],
       tags = t_tags)

drift_each_pp = q_full.mean_continuous_states


# Plot drift each subject

tag = 2
state_means = q_full.mean_continuous_states[tag]
estDrift = np.squeeze(lds.emissions.per_group_bernoulli_models[tag].Cs)*state_means[:]
estDrift = state_means[:]

fig, axs = plt.subplots(1, 1, figsize=(12,4))
axs.axhline(0, c = "k", ls = ":", lw =2)
axs.plot(t_drift[tag], "k", label = "Generative drift")
axs.plot(estDrift[:], c = 'firebrick', label = "Estimated drift")
axs.set(xlabel = "Trials", ylabel = "Decision criterion")
axs.legend(loc='upper center', bbox_to_anchor=(.5, 1.25),ncol=2, fancybox=True, shadow=True)





# #tags=[data['tag'] for data in full_datas]
# full_elbos = rslds.approximate_posterior(
#     q_full,
#     datas=[data['y'] for data in full_datas],
#     masks=[data['m'] for data in full_datas],
#     tags=[data['tag'] for data in full_datas],
#     num_iters=args.N_full_iter)



D = 1
lags = 1
K = 1
mean_A = None
bcast_and_repeat = lambda x, k: np.repeat(x[None, ...], k, axis=0)
if mean_A is None:
    mean_A = bcast_and_repeat(
        np.concatenate((np.zeros((D * (lags - 1), D)),
                        np.eye(D))), K)
mean_A.shape
lps[0][-1] 

# lds.approximate_posterior(datas = t_choices,inputs = t_inputs, tags = t_tags)


lds.dynamics.per_group_ar_models[0].As



lds.params
# estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]

# # Smooth the data under the variational posterior (to compute emissions)
#predEmissions = np.concatenate(lds.smooth(state_means, emissions, input = inputs))
# 
# lds.emissions.get_Cs(0)

# lds.
# N,D = 1,1
# mean_Cs = np.ones((1, N, D))
# mean_Cs.shape == (1, N, D)

# tags = t_tags
# x = dict([(tag, i) for i, tag in enumerate(tags)])



#TypeError: Cannot interpret '1' as a data type
# due to np.zeros((1, 1, 2)) with only one bracket

# import autograd.numpy.random as npr


# # original implementation
# N = 1
# D = 1
# M = 7

# # 3 or 2 dimensional
# Cs = npr.randn(1, N, D)
# Fs = npr.randn(1, N, M) 
# ds = npr.randn(1, N) 

# x = np.array([3], ndmin=3)
# spstats.norm.rvs(x, np.sqrt(1),10)
# x.shape




# lags = 1
# K = 1
# D = 1
# bcast_and_repeat = lambda x, k: np.repeat(x[None, ...], k, axis=0)

# mean_A = bcast_and_repeat(
#     np.concatenate((np.zeros((D * (lags - 1), D)),
#                     np.eye(D))), K)

# mean_A * np.ones((K, D, D * lags))

# mean_A.shape == (K, D, D * lags)
# # axis = 0 reduces dimension (eg 3 to 2)
# C = np.mean(Cs, axis=0)
# F = np.mean(Fs, axis=0)
# d = np.mean(ds, axis=0)

# # pseudoinv outputs 2 dimensional array
# C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

# import scipy.stats as spstats
# mean_Fs = np.zeros((N, D, M))
# x = np.array([3], ndmin=3)
# Cs = spstats.norm.rvs(mean_Cs, np.sqrt(variance_Cs))
# Fs = np.array([spstats.norm.rvs(mean_Fs, np.sqrt(variance_Fs),M)], ndmin=3) 
# ds = spstats.norm.rvs(mean_ds, np.sqrt(variance_ds))



# mean_Cs = npr.randn(1, N, D)
# spstats.norm.rvs(mean_Cs, np.sqrt(.01))





# C =  np.array([spstats.norm.rvs(1, np.sqrt(1))])
# C_pseudoinv = np.linalg.solve(np.array([C.T*C]), C.T).T
# C.T
# C.T.dot(C)


# import autograd.numpy.random as npr
# Fs = npr.randn(1, 1, 7)
# Fs.T
# np.mean(Fs, axis=0)

# x = spstats.norm.rvs(loc=0, scale=1, size=1, random_state=None)
# x[0]

# def fit_with_random_restarts(make_model,
#                              num_restarts=5,
#                              num_iters=1000,
#                              method="em",
#                              tolerance=1e-4,
#                              **kwargs):

#     all_models = []
#     all_lps = []

#     # Fit the model with a few random restarts
#     for r in range(num_restarts):
#         print("Restart ", r)
#         model = make_model()
#         lps = model.fit(datas, tags=tags,
#                         method=method, 
#                         num_iters=num_iters,
#                         tolerance=tolerance,
#                         **kwargs)
#         all_models.append(model)
#         all_lps.append(lps)
        
#     if isinstance(lps, tuple):
#         best_model_idx = np.argmax([lps[0][-1] for lps in all_lps])
#     else:
#         best_model_idx = np.argmax([lps[-1] for lps in all_lps])
        
#     best_model = all_models[idx]
    
#     return best_model, all_models, all_lps
# make_arhmm = lambda: ssm.HMM(num_states, obs_dim, 
#                              observations="ar",
#                              observation_kwargs=dict(lags=lags))

# reg_arhmm, all_reg_arhmms, all_reg_arhmm_lps = fit_with_random_restarts(make_arhmm)











#lds.initialize(y)
# lds.fit()
# elbos, q = lds.fit(t_choices, t_inputs, method="laplace_em",
#                     variational_posterior="structured_meanfield", 
#                     continuous_optimizer='newton',
#                     initialize=True, 
#                     num_init_restarts=1,
#                     num_iters=n_iters, 
#                     alpha=0.1, tags = t_tags)

# # Get the posterior mean of the continuous states (drift)
# state_means = q.mean_continuous_states[0]
# estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]






# Adjust hierarchical code Matt to Bernoulli distribution (using generic ssm.hierarchical code)
# Import these classes as dynamics and emissions in ssm.lds
# Change the __init__ from _BernoulliEmissionsMixin with parameters mean_Cs ... 
# Change


# D, K, lag = 1,1,2
# mean_Cs = None

# bcast_and_repeat = lambda x, k: np.repeat(x[None, ...], k, axis=0)
# if mean_Cs is None:
#     mean_Cs = bcast_and_repeat(
#         np.concatenate((np.zeros((D * (lag - 1), D)),
#                         np.eye(D))), K)
# assert mean_Cs.shape == (K, D, D)

# from ssm.util import random_rotation
# As = .80 * np.array(
#     [np.column_stack([random_rotation(D, theta=np.pi / 25),
#                       np.zeros((D, (lag - 1) * D))])
#      for _ in range(K)])
    