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



ntrial     = [500,1000,2000]
sens        = [5,10]
bias        = [0,-2,-4]
σd          = [0.05,0.2,.5]
pc          = [-1,0,1]
pe          = [-1,0,1]

ntrials         = []
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


inputDim = 3 # observed input dimensions 
stateDim = 1 # latent states
                        
for ntrials_lvl in ntrial:
    
    for sens_lvl in sens:
        
        for bias_lvl in bias:
            
            for sigma_lvl in σd:
                
                for pc_lvl in pc:
                    
                    for pe_lvl in pe:
                        
                        # save true simulation parameters                        
                        ntrials.append(ntrials_lvl)
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
                                                                                  pe = pe_lvl)
                        
                        # fit model
                        predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,
                                                                                inputs, choices)

                        state_means = q.mean_continuous_states[0]
                        estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]
                        
                        sens_fit.append(lds.emissions.Fs[0][0][0])
                        pc_fit.append(lds.emissions.Fs[0][0][1])
                        pe_fit.append(lds.emissions.Fs[0][0][2])
                        sigma_fit.append(lds.dynamics.Sigmas[0][0][0])
                        bias_fit.append(lds.emissions.ds[0][0])
            
    

df_plot = pd.DataFrame(data={'ntrials' : ntrials,
                             'sens_sim' : sens_sim,
                             'bias_sim' : bias_sim,
                             'sigma_sim' : sigma_sim,
                             'pc_sim' : pc_sim,
                             'pe_sim' : pe_sim,
                             'sens_fit' : sens_fit,
                             'bias_fit' : bias_fit,
                             'sigma_fit' : sigma_fit,
                             'pc_fit' : pc_fit,
                             'pe_fit' : pe_fit})


df_plot.to_csv("parameter_recovery.csv")


g = sns.PairGrid(df_plot,
                 x_vars=["sens_fit", "bias_fit","sigma_fit","pc_fit","pe_fit"],
                 y_vars=["sens_sim", "bias_sim","sigma_sim","pc_sim","pe_sim"],
                 height=4,
                 hue = "ntrials",
                 palette="Set2")
g.map(sns.scatterplot)



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

vertices.append([x, y])



