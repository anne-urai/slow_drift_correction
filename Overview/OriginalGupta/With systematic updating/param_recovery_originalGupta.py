# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:23:54 2022

@author: u0141056
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf


import sys #needed to load model_fit and simulate_choices since there in a different folder
sys.path.append('C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/slow_drift_correction')   

import model_fit, simulate_choices #contains the fitting funcs

# AR coef .9995 with drift mean- corrected 

n_datasets = 20

ntrial      = [500,10000,50000]
n_iter      = 200
sens        = 10
bias        = -5
sigma_d     = .05

pc = 1
pe = -1

total = n_datasets * len(ntrial)


ntrials         = []
n_iterations    = [n_iter] * total
sens_sim        = [sens] * total
bias_sim        = [bias] * total
sigma_sim       = [sigma_d] * total
pc_sim          = [pc] * total
pe_sim          = [pe] * total



sens_fit        = []
bias_fit        = []
sigma_fit       = []
pc_fit          = []
pe_fit          = []
C_fit           = []



inputDim = 3 # observed input dimensions
stateDim = 1 # latent states
count = 1

pdf = matplotlib.backends.backend_pdf.PdfPages("param_recovery_originalGupta_estimated_slowdrifts.pdf")


for dataset in range(n_datasets):
    
    count = count + 1
    print("number of datasets to go: " + str(n_datasets - dataset))
                      
    for ntrials_lvl in ntrial:
        ntrials.append(ntrials_lvl)

        # simulate dataset
        inputs, choices, drift = simulate_choices.simulateChoice(ntrials = ntrials_lvl,
                                                                 sens = sens,
                                                                 bias = bias,
                                                                 σd = sigma_d,
                                                                 pc = pc,
                                                                 pe = pe,
                                                                 seed = count)

        # fit model
        predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,
                                                                inputs, choices, n_iters = n_iter)

        state_means = q.mean_continuous_states[0]
        estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]


        sens_fit.append(lds.emissions.Fs[0][0][0])
        pc_fit.append(lds.emissions.Fs[0][0][1])
        pe_fit.append(lds.emissions.Fs[0][0][2])
        sigma_fit.append(lds.dynamics.Sigmas[0][0][0])
        bias_fit.append(lds.emissions.ds[0][0])
        C_fit.append(lds.emissions.Cs[0][0][0])


        
        if ntrials_lvl == 500: #first save plot for n = 500
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12,10))
            plt.subplots_adjust(hspace=1)
            
            ax1.axhline(0, c = "k", ls = ":", lw =2)
            ax1.plot(drift[:500], "k", label = "Generative drift")
            ax1.plot(estDrift[:500], c = 'firebrick', label = "Estimated drift")
            ax1.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax1.legend(loc='upper right')
            ax1.title.set_text('Slow drift (n = 500)')
            

            
        elif ntrials_lvl == 10000:
            
            ax2.axhline(0, c = "k", ls = ":", lw =2)
            ax2.plot(drift[:500], "k", label = "Generative drift")
            ax2.plot(estDrift[:500], c = 'firebrick', label = "Estimated drift")
            ax2.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax2.legend(loc='upper right')
            ax2.title.set_text('Slow drift, zoomed in on first 500 (n = 10000)')
            
            ax4.axhline(0, c = "k", ls = ":", lw =2)
            ax4.plot(drift[:], "k", label = "Generative drift")
            ax4.plot(estDrift[:], c = 'firebrick', label = "Estimated drift")
            ax4.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax4.legend(loc='upper right')
            ax4.title.set_text('Slow drift (n = 10000)')
            
            
        elif ntrials_lvl == 50000:
            
            ax3.axhline(0, c = "k", ls = ":", lw =2)
            ax3.plot(drift[:500], "k", label = "Generative drift")
            ax3.plot(estDrift[:500], c = 'firebrick', label = "Estimated drift")
            ax3.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax3.legend(loc='upper right')
            ax3.title.set_text('Slow drift, zoomed in on first 500 (n = 50000)')
            
            # add second plot with zoom in on first 500 trials when ntrials = 10000
            ax5.axhline(0, c = "k", ls = ":", lw =2)
            ax5.plot(drift[:], "k", label = "Generative drift")
            ax5.plot(estDrift[:], c = 'firebrick', label = "Estimated drift")
            ax5.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax5.legend(loc='upper right')
            ax5.title.set_text('Slow drift (n = 50000)')



            pdf.savefig(fig)
                                
                                
pdf.close()

df_plot = pd.DataFrame(data={'ntrials' : ntrials,
                             'n_iters' : n_iterations,
                             
                             'sens_sim' : sens_sim,
                             'bias_sim' : bias_sim,
                             'sigma_sim' : sigma_sim,
                             
                             'pc_sim' : pc_sim,
                             'pe_sim' : pe_sim,

                             'sens_fit' : sens_fit,
                             'bias_fit' : bias_fit,
                             'sigma_fit' : sigma_fit,

                             'C_fit' : C_fit,

                             'pc_fit' : pc_fit,
                             'pe_fit' : pe_fit})


df_plot.to_csv("param_recovery_originalGupta.csv")









