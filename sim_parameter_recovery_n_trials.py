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

# 200 n_iter with AR coef .9995

n_datasets = 20

ntrial      = [500,5000,15000,25000]
n_iter      = [100]
sens        = [10]
bias        = [-5]
sigma_d     = [0.1]
pc          = [1]
pe          = [-1]

total_iterations = len(ntrial) * len(sens) * len(bias) * len(sigma_d) * len(pc) * len(pe) * len(n_iter) * n_datasets
print('number of iterations: ' + str(total_iterations))



ntrials         = []
n_iters         = []
sens_sim        = []
bias_sim        = []
sigma_sim       = []
pc_sim          = []
pe_sim          = []

sens_fit        = []
bias_fit        = []
sigma_fit       = []
pc_fit          = []
pe_fit          = []
C_fit           = []

mse             = []


inputDim = 3 # observed input dimensions
stateDim = 1 # latent states
count = 0

for dataset in range(n_datasets):
    
    count = count + 1
    
    print("number of datasets to go: " + str(n_datasets - dataset))

    for sens_lvl in sens:

        for bias_lvl in bias:

            for sigma_lvl in sigma_d:

                for pc_lvl in pc:

                    for n_iters_lvl in n_iter:
                        
                        for pe_lvl in pe:
                            
                            for ntrials_lvl in ntrial:
    
                                
        
                                
        
        
                                # save true simulation parameters
                                ntrials.append(ntrials_lvl)
                                n_iters.append(n_iters_lvl)
                                sens_sim.append(sens_lvl)
                                bias_sim.append(bias_lvl)
                                sigma_sim.append(sigma_lvl)
                                pc_sim.append(pc_lvl)
                                pe_sim.append(pe_lvl)
    
        
        
                                # simulate dataset
                                inputs, choices, drift = simulate_choices.simulateChoice(ntrials = ntrials_lvl,
                                                                                          sens = sens_lvl,
                                                                                          bias = bias_lvl,
                                                                                          σd = sigma_lvl,
                                                                                          pc = pc_lvl,
                                                                                          pe = pe_lvl,
                                                                                          seed = count)
        
                                # fit model
                                predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,
                                                                                        inputs, choices, n_iters = n_iters_lvl)
        
                                state_means = q.mean_continuous_states[0]
                                estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]
        
        
                                # Mean squared errors: as a reference: model with 40000 trials had .5717
                                mean_sq_err = sum((drift - estDrift)**2) / ntrials_lvl
                                mse.append(mean_sq_err[0])
        
        
                                # compare predEmissions with psychfunc
        
                                sens_fit.append(lds.emissions.Fs[0][0][0])
                                pc_fit.append(lds.emissions.Fs[0][0][1])
                                pe_fit.append(lds.emissions.Fs[0][0][2])
                                sigma_fit.append(lds.dynamics.Sigmas[0][0][0])
                                bias_fit.append(lds.emissions.ds[0][0])
                                C_fit.append(lds.emissions.Cs[0][0][0])
                                

df_plot = pd.DataFrame(data={'ntrials' : ntrials,
                             'n_iters' : n_iters,
                             'mse' : mse,
                             'sens_sim' : sens_sim,
                             'bias_sim' : bias_sim,
                             'sigma_sim' : sigma_sim,
                             'pc_sim' : pc_sim,
                             'pe_sim' : pe_sim,
                             'sens_fit' : sens_fit,
                             'bias_fit' : bias_fit,
                             'sigma_fit' : sigma_fit,
                             'pc_fit' : pc_fit,
                             'pe_fit' : pe_fit,
                             'C_fit' : C_fit})



df_plot.to_csv("parameter_recovery_inflating_sigma.csv")





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







data = makeDataDict(inputs, emissions, predEmissions = predEmissions)

fig, axs = plt.subplots(1,2, figsize=(12,4), gridspec_kw={'width_ratios': [1.7,1]})
axs[0].axhline(0, c = "k", ls = ":", lw =2)
axs[0].plot(drift[:5000],  "k", lw = 2,label = "Generative drift")
axs[0].plot(np.squeeze(estDrift)[:5000], ls = "-", lw = 2, c = [0.5, 0.5, 0.5], label = "Estimated drift")
axs[0].set(xlabel = "Trials", ylabel = "Drift in choice bias")
axs[0].set_ylim(-50/sens, 40/sens)

axs[1].get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)
axs[1].spines['bottom'].set_edgecolor("white")
axs[1].spines['left'].set_edgecolor("white")
sns.despine()

fig, axs = plt.subplots(1,2, figsize=(12,4), subplot_kw=dict(box_aspect=1) )
plotCondPsych(data[data.Model == 0], True, axs[0], ls = '-')
plotCondPsych(data[data.Model == 0], False, axs[1], ls = '-')
plotCondPsych(data[data.Model == 1], True, axs[0], ls = (0, (3, 1, 1, 1)))
plotCondPsych(data[data.Model == 1], False, axs[1], ls = (0, (3, 1, 1, 1)))   
axs[1].set_ylabel("")
axs[1].get_xaxis().set_visible(False)
axs[0].get_xaxis().set_visible(False)
sns.despine()


# plot without drift (generative and model)
fig, axs = plt.subplots(1,2, figsize=(12,4), subplot_kw=dict(box_aspect=1) )
for i, ax in enumerate(axs):
    p = pc if i == 0 else pe
    col = colors_c if i == 0 else colors_e
    ax.plot(XVAL_DEF, sigmoid(sens*XVAL_DEF + bias + p), c = col[1], lw = 2)
    ax.plot(XVAL_DEF, sigmoid(sens*XVAL_DEF + bias - p), c = col[0], lw = 2)
    ax.plot(XVAL_DEF, sigmoid(lds.emissions.Fs[0,0,0]*XVAL_DEF + lds.emissions.ds + lds.emissions.Fs[0,0,i+1]), 
                        ls = (0, (3, 1, 1, 1)), c = col[1], lw = 3)
    ax.plot(XVAL_DEF, sigmoid(lds.emissions.Fs[0,0,0]*XVAL_DEF + lds.emissions.ds - lds.emissions.Fs[0,0,i+1]), 
                        ls = (0, (3, 1, 1, 1)), c = col[0], lw = 3)
    ax.set_ylabel("Fraction chose right")
    ax.set_xlabel("Stimulus strength")
    
axs[1].set_ylabel("")
sns.despine()
