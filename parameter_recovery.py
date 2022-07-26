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



ntrial      = [100, 500, 1000, 5000]
sens        = [1, 5, 10]
bias        = [-5, 0, 5]
sigma_d     = [0, 0.01, 0.05, 0.1]
pc          = [-1,0,1]
pe          = [-1,0,1]

print('number of iterations:')
print(len(ntrial) * len(sens) * len(bias) * len(sigma_d) * len(pc) * len(pe))

# add 200, 500, 1000 n_iter

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
count = 0
                        
for ntrials_lvl in ntrial:
    
    for sens_lvl in sens:
        
        for bias_lvl in bias:
            
            for sigma_lvl in sigma_d:
                
                for pc_lvl in pc:
                    
                    for pe_lvl in pe:
                        
                        # save true simulation parameters                        
                        ntrials.append(ntrials_lvl)
                        sens_sim.append(sens_lvl)
                        bias_sim.append(bias_lvl)
                        sigma_sim.append(sigma_lvl)
                        pc_sim.append(pc_lvl)
                        pe_sim.append(pe_lvl)
                        count =+ 1

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
                                                                                inputs, choices)

                        state_means = q.mean_continuous_states[0]
                        estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]
                        
                        # correlate estdrift with drift
                        
                        # compare predEmissions with psychfunc
                        
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


def myscatter(x, y):
    
    # axis equal
    # scatter
    # regression
    
    # see: https://github.com/anne-urai/2022_Urai_choicehistory_MEG/blob/main/hddmnn_funcs.py#L17

g = sns.PairGrid(df_plot,
                 x_vars=["sens_fit", "bias_fit","sigma_fit","pc_fit","pe_fit"],
                 y_vars=["sens_sim", "bias_sim","sigma_sim","pc_sim","pe_sim"],
                 height=4,
                 hue = "ntrials",
                 palette="Set2")
g.map(myscatter)



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



