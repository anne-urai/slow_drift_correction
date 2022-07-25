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
import ssm
import model_fit, simulate_choices #contains the fitting funcs
simulate = True

#%% simulate choice data
if simulate:
    ntrials = 40000 # original code: 40.000
    sens = 10
    bias = -0.5*sens
    σd = 0.05
    pc, pe = 1, 1
    
    inputs, choices, drift = simulate_choices.simulateChoice(ntrials, σd = σd, 
                                              sens = sens, bias = bias, 
                                              pc = pc, pe = pe)
    '''
    inputs (array): ntrialsx3, first column is stimulus strength, second column is indicator for 
                post-correct trials (+1 right, -1 left, 0 error) and third column is indicator
                for post-error trials (+1 right, -1 left, 0 correct) 
    emissions (array): ntrialsx1 choices made by the SDT oberver +1 for rightward choices,
                0 for leftward
    drift (array): ntrialsx1, mean centered 
    '''
else:  # alternative: load real data
    # from https://figshare.com/articles/dataset/Choice_history_biases_subsequent_evidence_accumulation/7268558/2?file=13593227
    data = pd.read_csv('https://figshare.com/ndownloader/files/13593227')
    # print(data.columns) # we'll use prevpupil as a stand-in for prevconfidence
    df = data[data.subj_idx == 1] # pick one subject
    
    # extract the trial-by-trial stimulus strength
    stim_strength = (df.coherence * df.stimulus).to_numpy() # or df.motionenergy
    # scale between 0 and 1, to match Gupta simulateChoice output
    stim_strength = (stim_strength-stim_strength.min())/(stim_strength.max()-stim_strength.min())

    # separate in post-correct trials
    prevchoice_correct = np.asarray([df.prevresp[t] if df.prevresp[t] == df.prevstim[t] 
                                     else 0 for t in range(len(df.prevresp))])
    prevchoice_error = np.asarray([df.prevresp[t] if df.prevresp[t] != df.prevstim[t] 
                                   else 0 for t in range(len(df.prevresp))])
    
    # this is the format that the fitting func needs - vstack as np arrays
    inputs = np.vstack([stim_strength, prevchoice_correct, prevchoice_error]).T
    choices = np.vstack(np.asarray(df.response.tolist()).T)

#%% fit the LDS model
inputDim = np.shape(inputs)[1]
stateDim = 1

predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,
                                                        inputs, choices)

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

N, D = 1, 1
lds = ssm.LDS(1, 1)

# now: add input dynamics so that these can be given to the fit
lds.dynamics         = model_fit.AutoRegressiveNoInput(1, D, M=inputDim) # import those
lds.transitions      = ssm.transitions.StationaryTransitions(1, D, inputDim)
lds.init_state_distn = ssm.init_state_distns.InitialStateDistribution(1, D, inputDim)
lds.emissions        = ssm.emissions.BernoulliEmissions(N, 1, D, M=inputDim)

# # edit some of those (why? talk to Diksha)
lds.dynamics.A       = np.ones((stateDim,stateDim))           # dynamics
lds.dynamics.b       = np.zeros(stateDim)                     # bias
lds.dynamics.mu_init = np.zeros((stateDim,stateDim))          # initial mu
lds.dynamics.Sigmas_init = np.array([[[0.01]]])               # initial sigma
lds.dynamics.Vs      = np.array([[np.zeros(inputDim)]])       # input dynamics

# note: everything that is not specified (in this case Cs, d, F) is fitted
# why not fit A instead of C?

# do the actual fit
elbos, q = lds.fit(choices, inputs = inputs, 
                    method = "laplace_em",
                    variational_posterior = "structured_meanfield", 
                    continuous_optimizer = 'newton',
                    initialize = True, 
                    num_init_restarts = 1,
                    num_iters = 200, 
                    alpha = 0.1)

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
axs.plot(drift, "k", label = "Generative drift")
axs.plot(estDrift, c = 'firebrick', label = "Estimated drift")
axs.set(xlabel = "Trials", ylabel = "Decision criterion")
axs.legend(loc='upper right')

#%% save a clear and helpful fig
