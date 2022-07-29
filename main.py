#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:15:06 2022
@author: urai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to build our model
import model_fit, simulate_choices #contains the fitting funcs
simulate = False

#%% simulate choice data
if simulate:
    ntrials = 500 # original code: 40.000
    sens = 10
    bias = -5 # -5 is unbiased
    σd = 0.1
    w_prevresp = 1
    w_prevconf = 0
    w_prevconfprevresp = 1
    
    inputs, choices, drift = simulate_choices.simulateChoiceRespConf(ntrials, σd = σd, 
                                              sens = sens, bias = bias, 
                                              w_prevresp = w_prevresp, 
                                              w_prevconf = w_prevconf,
                                              w_prevconfprevresp = w_prevconfprevresp)
    
                    
    '''
    inputs (array): ntrialsx3, first column is stimulus strength, second column is indicator for 
                post-correct trials (+1 right, -1 left, 0 error) and third column is indicator
                for post-error trials (+1 right, -1 left, 0 correct) 
    choices (array): ntrialsx1 choices made by the SDT oberver +1 for rightward choices,
                0 for leftward
    drift (array): ntrialsx1, mean centered 
    '''
else:  # alternative: load real data

    data = pd.read_csv('df_slowdrift_correction.csv')
    print(data.columns)
    
    
    #df = data[data.subj == 1] # pick one subject
    df = data
    
    evidence = df.evidence.to_numpy() # scaled between 0 and 1 (with 0 being left evidence, 1 right evidence, .5 would be ambigious no evidence trial)
    prevresp = df.prevresp.to_numpy()
    prevconf = df.prevconf.to_numpy()
    prevrespconf = df.prevrespconf.to_numpy()
    
    # this is the format that the fitting func needs - vstack as np arrays
    inputs = np.vstack([evidence, prevresp, prevconf, prevrespconf]).T
    choices = np.vstack(np.asarray(df.resp.tolist()).T)


#%% instead, spell out what's happening
# skip for now, doesnt work - AEU

# create a plain model
# see https://github.com/lindermanlab/ssm/blob/master/notebooks/1b%20Simple%20Linear%20Dynamical%20System.ipynb
# see also https://github.com/lindermanlab/ssm/blob/master/ssm/lds.py#L825

# Set the parameters of the HMM
# K = 1       # number of discrete states: 1, no switching
# D = 1       # number of latent dimensions: 1, AR model
# N = 1       # number of observed dimensions: only choices. Could add in RT/confidence?
# M = 3       # input dimensions

# N, D = 1, 1
# lds = ssm.LDS(1, 1)

# now: add input dynamics so that these can be given to the fit
# lds.dynamics         = model_fit.AutoRegressiveNoInput(1, D, M=inputDim) # import those
# lds.transitions      = ssm.transitions.StationaryTransitions(1, D, inputDim)
# lds.init_state_distn = ssm.init_state_distns.InitialStateDistribution(1, D, inputDim)
# lds.emissions        = ssm.emissions.BernoulliEmissions(N, 1, D, M=inputDim)

# # edit some of those (why? talk to Diksha)
# lds.dynamics.A       = np.ones((stateDim,stateDim))           # dynamics
# lds.dynamics.b       = np.zeros(stateDim)                     # bias
# lds.dynamics.mu_init = np.zeros((stateDim,stateDim))          # initial mu
# lds.dynamics.Sigmas_init = np.array([[[0.01]]])               # initial sigma
# lds.dynamics.Vs      = np.array([[np.zeros(inputDim)]])       # input dynamics

# note: everything that is not specified (in this case Cs, d, F) is fitted
# why not fit A instead of C?

# do the actual fit
# elbos, q = lds.fit(choices, inputs = inputs, 
#                     method = "laplace_em",
#                     variational_posterior = "structured_meanfield", 
#                     continuous_optimizer = 'newton',
#                     initialize = True, 
#                     num_init_restarts = 1,
#                     num_iters = 200, 
#                     alpha = 0.1)


#%% fit the LDS model
inputDim = np.shape(inputs)[1] # observed input dimensions 
stateDim = 1 # latent states
n_iters = 500
predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,
                                                        inputs, choices,n_iters)


#%%% Get the posterior mean of the continuous states (drift)

'''
You are totally right, sigma_d is the parameter that determines the degree to which the random 
noise is added to the signal. To infer the specific changes in slow drift from trial-to-trial, 
we need to estimate the conditional posterior density of the latent variable x_{t} aka drift 
given the observed choices y_{1...T}  - i.e. p(x_t | y_{1...T}). This operation is often called 
"smoothing" in latent variable models if T>t . 
Essentially, given the initial value of the random drift, the standard deviation specifies 
the distribution of values that the drift can take on the next trial. But, given that we 
observed the choice in the next trial, we can leverage that information to further sharpen 
the estimate of what the value of drift might have been. Macke 2015 is my go-to reference 
for reading about this, they consider a Poisson noise model with LDS, but essentially the 
same notions apply to a Bernoulli LDS (as in the manuscript).
'''

state_means = q.mean_continuous_states[0]
estDrift = np.squeeze(lds.emissions.Cs)*state_means[:] # flip around?

# Smooth the data under the variational posterior (to compute emissions aka choices)
predEmissions = np.concatenate(lds.smooth(state_means, choices, input = inputs))

#%% plot some things
# Plot the ELBOs
fig, axs = plt.subplots(1, 1, figsize=(4,4))
axs.plot(elbos[2:], label="Laplace-EM")
axs.set(xlabel="Iteration", ylabel="ELBO")
axs.legend()
sns.despine()

# does the simulated drift match the predicted drift?
fig, axs = plt.subplots(1, 1, figsize=(12,4))
axs.axhline(0, c = "k", ls = ":", lw =2)
axs.plot(drift[:], "k", label = "Generative drift")
axs.plot(estDrift[:], c = 'firebrick', label = "Estimated drift")
axs.set(xlabel = "Trials", ylabel = "Decision criterion")
axs.legend(loc='upper right')

mse = sum((drift - estDrift)**2) / ntrials

lds.dynamics.As
lds.dynamics.Vs
lds.dynamics.b
lds.dynamics.mu_init
lds.dynamics.Sigmas

lds.emissions.Cs
lds.emissions.Fs
lds.emissions.ds 
#%% save a clear and helpful fig







