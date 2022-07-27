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

ntrial      = [100, 500, 1000, 5000]
sens        = [1, 5, 10]
bias        = [-5, 0, 5]
sigma_d     = [0, 0.01, 0.05, 0.1]
pc          = [-1,0,1]
pe          = [-1,0,1]

total_iterations = len(ntrial) * len(sens) * len(bias) * len(sigma_d) * len(pc) * len(pe)
print('number of iterations: ' + str(total_iterations))



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

mse             = []


inputDim = 3 # observed input dimensions 
stateDim = 1 # latent states
count = 0

                        
for ntrials_lvl in ntrial:
    
    for sens_lvl in sens:
        
        for bias_lvl in bias:
            
            for sigma_lvl in sigma_d:
                
                for pc_lvl in pc:
                    
                    for pe_lvl in pe:
                        
                        count = count + 1
                        
                        print("number of iterations to go: " + str(total_iterations - count))
                        

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
                                                                                  pe = pe_lvl, 
                                                                                  seed = count)
                        
                        # fit model
                        predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,
                                                                                inputs, choices)

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
            

df_plot = pd.DataFrame(data={'ntrials' : ntrials,
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
                             'pe_fit' : pe_fit})


df_plot.to_csv("parameter_recovery500_AR1.csv")



fig, ax = plt.subplots(1,5)

plt.subplots_adjust(left=-.5,
                    bottom=0.1, 
                    right=1, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

def scatter(x,y,axis):
    p = sns.scatterplot(data=df_plot, x=x, y=y,hue = "ntrials",legend = False,ax=ax[axis])
    
    
    low = min(df_plot[x] + df_plot[y])
    high = max(df_plot[x] + df_plot[y])
    
    llow = low - (abs(high - low)/2)
    hhigh = high + (abs(high - low)/2)
    p.set_ylim(llow, hhigh)
    p.set_xlim(llow, hhigh)
    
    p.set_aspect('equal', adjustable='box')

    # Draw a line of x=y 
    
    lims = [llow, hhigh]
    p.plot(lims, lims, '-r')

scatter("sens_sim","sens_fit",0)
scatter("bias_sim","bias_fit",1)
scatter("sigma_sim","sigma_fit",2)
scatter("pc_sim","pc_fit",3)
scatter("pe_sim","pe_fit",4)

     
fig.show()





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




