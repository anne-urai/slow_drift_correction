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

import model_fitAR, simulate_choices #contains the fitting funcs

# AR coef .9995 with drift NOT mean- corrected 
# Intercept is added to estimated drift

n_datasets = 20

ntrial      = [10000,50000]
n_iter      = 1000
sens        = 10
bias        = 0
sigma_d     = .1



total = n_datasets * len(ntrial)


w_prevresp  = 0
w_prevconf  = 0
w_prevrespconf = 0
w_prevsignevi = 0
w_prevabsevi = 0
w_prevsignabsevi = 0

total = n_datasets * len(ntrial)


ntrials         = []
n_iterations    = [n_iter] * total
sens_sim        = [sens] * total
bias_sim        = [bias] * total
AR_sim        = [.9995] * total
sigma_sim       = [sigma_d] * total
prevresp_sim  = [w_prevresp] * total
prevconf_sim  = [w_prevconf] * total
prevrespconf_sim  = [w_prevrespconf] * total
prevsignevi_sim  = [w_prevsignevi] * total
prevabsevi_sim  = [w_prevabsevi] * total
prevsignabsevi_sim  = [w_prevsignabsevi] * total



sens_fit        = []
bias_fit        = [0] * total
sigma_fit       = []
prevresp_fit  = []
prevconf_fit  = []
prevrespconf_fit  = [] 
prevsignevi_fit  = []
prevabsevi_fit  = []
prevsignabsevi_fit  = []
AR_fit           = []



inputDim = 7 # observed input dimensions
stateDim = 1 # latent states
count = 1

pdf = matplotlib.backends.backend_pdf.PdfPages("param_recovery_AR_prevrespconf_noSys_9995_estimated_slowdrifts.pdf")


for dataset in range(n_datasets):
    
    count = count + 1
    print("number of datasets to go: " + str(n_datasets - dataset))
                      
    for ntrials_lvl in ntrial:
        ntrials.append(ntrials_lvl)

        # simulate dataset
        inputs, choices, drift = simulate_choices.simulateChoice_normalEvi_slowdriftConf(estimateUpdating = True,
                                                                 fixedConfCriterion = False,
                                                                 postDecisionalEvi = True,
                                                                 ntrials = ntrials_lvl,
                                                                 sens = sens,
                                                                 bias = bias,
                                                                 σd = sigma_d,
                                                                 w_prevresp = w_prevresp,
                                                                 w_prevconf = w_prevconf,
                                                                 w_prevrespconf = w_prevrespconf,
                                                                 w_prevsignevi = w_prevsignevi,
                                                                 w_prevabsevi = w_prevabsevi,
                                                                 w_prevsignabsevi = w_prevsignabsevi,
                                                                 seed = count)

        # fit model
        predEmissions, estDrift, lds, q, elbos = model_fitAR.initLDSandFitAR(inputDim,
                                                                inputs, choices, n_iters = n_iter)

        state_means = q.mean_continuous_states[0]
        estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]


        sens_fit.append(lds.emissions.Fs[0][0][0])
        prevresp_fit.append(lds.emissions.Fs[0][0][1])
        prevconf_fit.append(lds.emissions.Fs[0][0][2])
        prevrespconf_fit.append(lds.emissions.Fs[0][0][3])
        prevsignevi_fit.append(lds.emissions.Fs[0][0][4])
        prevabsevi_fit.append(lds.emissions.Fs[0][0][5])
        prevsignabsevi_fit.append(lds.emissions.Fs[0][0][6])
        sigma_fit.append(lds.dynamics.Sigmas[0][0][0])
        AR_fit.append(lds.dynamics.As[0][0][0])



            
        if ntrials_lvl == 10000:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,10))
            plt.subplots_adjust(hspace=1)
            
            ax1.axhline(0, c = "k", ls = ":", lw =2)
            ax1.plot(drift[:1000], "k", label = "Generative drift")
            ax1.plot(estDrift[:1000], c = 'firebrick', label = "Estimated drift")
            ax1.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax1.legend(loc='upper right')
            ax1.title.set_text('Slow drift, zoomed in on first 1000 (n = 10000)')
            
            ax3.axhline(0, c = "k", ls = ":", lw =2)
            ax3.plot(drift[:], "k", label = "Generative drift")
            ax3.plot(estDrift[:], c = 'firebrick', label = "Estimated drift")
            ax3.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax3.legend(loc='upper right')
            ax3.title.set_text('Slow drift (n = 10000)')
            
            
        elif ntrials_lvl == 50000:
            
            ax2.axhline(0, c = "k", ls = ":", lw =2)
            ax2.plot(drift[:1000], "k", label = "Generative drift")
            ax2.plot(estDrift[:1000], c = 'firebrick', label = "Estimated drift")
            ax2.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax2.legend(loc='upper right')
            ax2.title.set_text('Slow drift, zoomed in on first 1000 (n = 50000)')
            
            # add second plot with zoom in on first 500 trials when ntrials = 10000
            ax4.axhline(0, c = "k", ls = ":", lw =2)
            ax4.plot(drift[:], "k", label = "Generative drift")
            ax4.plot(estDrift[:], c = 'firebrick', label = "Estimated drift")
            ax4.set(xlabel = "Trials", ylabel = "Decision criterion")
            ax4.legend(loc='upper right')
            ax4.title.set_text('Slow drift (n = 50000)')



            pdf.savefig(fig)
                                
                                
pdf.close()

df_plot = pd.DataFrame(data={'ntrials' : ntrials,
                             'n_iters' : n_iterations,
                             
                             'sens_sim' : sens_sim,
                             'bias_sim' : bias_sim,
                             'sigma_sim' : sigma_sim,
                             
                             'AR_sim' : AR_sim,
                             
                             'prevresp_sim' : prevresp_sim,
                             'prevconf_sim' : prevconf_sim,
                             'prevrespconf_sim' : prevrespconf_sim,
                             'prevsignevi_sim' : prevsignevi_sim,
                             'prevabsevi_sim' : prevabsevi_sim,
                             'prevsignabsevi_sim' : prevsignabsevi_sim,

                             
                             'sens_fit' : sens_fit,
                             'bias_fit' : bias_fit,
                             'sigma_fit' : sigma_fit,

                             'AR_fit' : AR_fit,

                             'prevresp_fit' : prevresp_fit,
                             'prevconf_fit' : prevconf_fit,
                             'prevrespconf_fit' : prevrespconf_fit,
                             'prevsignevi_fit' : prevsignevi_fit,
                             'prevabsevi_fit' : prevabsevi_fit,
                             'prevsignabsevi_fit' : prevsignabsevi_fit})


df_plot.to_csv("param_recovery_prevrespconf_AR_noSys_9995.csv")









