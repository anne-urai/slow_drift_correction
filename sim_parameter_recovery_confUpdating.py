# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:23:54 2022

@author: u0141056
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to build our model

import model_fit, simulate_choices #contains the fitting funcs

# 2x2x2x2
# simulated with slow drift, conf, fitting slow drift, conf
# 1111 means simulated with slow drift and conf updating, and both are also fitted

n_datasets = 15

ntrial      = 500
n_iter      = 50
sens        = 10
bias        = -5
sigma_d     = 0
w_prevresp  = 1
w_prevconf  = 0
w_prevconfprevresp = 1


ntrials                = []
n_iters                = []
sens_sim               = []
bias_sim               = []
sigma_sim              = []
prevresp_sim           = []
prevconf_sim           = []
prevconfprevresp_sim   = []


sens_fit               = []
bias_fit               = []
sigma_fit              = []
C_fit                  = []
prevresp_fit           = []
prevconf_fit           = []
prevconfprevresp_fit   = []
mse                    = []


inputDim = 4 # observed input dimensions
stateDim = 1 # latent states
count = 0

for dataset in range(n_datasets):
    
    count = count + 1
    
    print("number of datasets to go: " + str(n_datasets - dataset))
    
    
    # save true simulation parameters
    ntrials.append(ntrial)
    n_iters.append(n_iter)
    sens_sim.append(sens)
    bias_sim.append(bias)
    sigma_sim.append(sigma_d)
    prevresp_sim.append(w_prevresp)
    prevconf_sim.append(w_prevconf)
    prevconfprevresp_sim.append(w_prevconfprevresp)



    # simulate dataset
    inputs, choices, drift = simulate_choices.simulateChoiceRespConf(ntrials = ntrial,
                                                              sens = sens,
                                                              bias = bias,
                                                              σd = sigma_d,
                                                              w_prevresp = w_prevresp,
                                                              w_prevconf = w_prevconf,
                                                              w_prevconfprevresp = w_prevconfprevresp,
                                                              seed = count)

    # fit model
    predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,
                                                            inputs, choices, n_iters = n_iter)

    state_means = q.mean_continuous_states[0]
    estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]


    # Mean squared errors: as a reference: model with 40000 trials had .5717
    mean_sq_err = sum((drift - estDrift)**2) / ntrial
    mse.append(mean_sq_err[0])


    # compare predEmissions with psychfunc

    sens_fit.append(lds.emissions.Fs[0][0][0])
    bias_fit.append(lds.emissions.ds[0][0])
    prevresp_fit.append(lds.emissions.Fs[0][0][1])
    prevconf_fit.append(lds.emissions.Fs[0][0][2])
    prevconfprevresp_fit.append(lds.emissions.Fs[0][0][3])
    sigma_fit.append(lds.dynamics.Sigmas[0][0][0])
    C_fit.append(lds.emissions.Cs[0][0][0])


df_plot = pd.DataFrame(data={'ntrials' : ntrials,
                             'n_iters' : n_iters,
                             'mse' : mse,
                             'sens_sim' : sens_sim,
                             'sens_fit' : sens_fit,
                             'bias_sim' : bias_sim,
                             'bias_fit' : bias_fit,
                             'sigma_sim' : sigma_sim,
                             'sigma_fit' : sigma_fit,
                             'c_fit' : C_fit,
                             'prevresp_sim' : prevresp_sim,
                             'prevresp_fit' : prevresp_fit,
                             'prevconf_sim' : prevconf_sim,
                             'prevconf_fit' : prevconf_fit,
                             'prevconfprevresp_sim' : prevconfprevresp_sim,
                             'prevconfprevresp_fit' : prevconfprevresp_fit})



df_plot.to_csv("parameter_recoveryConf_0111.csv")





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



lds.dynamics.As
lds.dynamics.Vs
lds.dynamics.b
lds.dynamics.mu_init
lds.dynamics.Sigmas

lds.emissions.Cs
lds.emissions.Fs
lds.emissions.ds





