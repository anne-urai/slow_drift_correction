# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:23:54 2022

@author: u0141056
"""

import sys #needed to load model_fit since there in a different folder
sys.path.append('C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/slow_drift_correction')   

# ds fixed to 0
import pandas as pd
import numpy as np
import model_fit #contains the fitting funcs
import simulate_data


n_iters = 200 # number of iterations the E-M procedure takes
n_trials = 10000
n_datasets = 20

dataset = []
drift_type = []
drift_value = []
trial = []


sens        = 10
bias        = 0
sigma_d     = .05
w_prevresp  = .5
w_prevconf  = 0
w_prevrespconf = 1
w_prevsignevi = -1
w_prevabsevi = 0
w_prevsignabsevi = -1

count = 0





for iteration in range(n_datasets):
    
    count = count + 1
    print("iteration: " + str(iteration))
    
    

    # Estimate drift while also estimating the systematic updating
    inputs, choices, drift = simulate_data.simulate_estimationWithEffects(ntrials = n_trials,
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
    

    # fit the model
    inputDim = np.shape(inputs)[1] # observed input dimensions 
    stateDim = 1 # latent states

    predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim, inputs, choices, n_iters = n_iters)


    state_means = q.mean_continuous_states[0]
    estDrift_with = np.squeeze(lds.emissions.Cs)*state_means[:] + lds.emissions.ds


    
    # Estimate drift while also estimating the systematic updating
    
    inputs, choices, drift = simulate_data.simulate_estimationWithoutEffects(ntrials = n_trials,
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
    

    # fit the model
    inputDim = np.shape(inputs)[1] # observed input dimensions 
    stateDim = 1 # latent states

    predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim, inputs, choices, n_iters = n_iters)


    state_means = q.mean_continuous_states[0]
    estDrift_without = np.squeeze(lds.emissions.Cs)*state_means[:] + lds.emissions.ds
    
    
    
    
    drift = np.squeeze(drift)
    estDrift_without = np.squeeze(estDrift_without)
    estDrift_with = np.squeeze(estDrift_with)

    # Messy solution to add values to list, otherwise we get list of lists
    for i in range(len(drift)):
        drift_value.append(drift[i])
        drift_type.append("generative")
        dataset.append(iteration)
        trial.append(i)
        
    for i in range(len(estDrift_with)):
        drift_value.append(estDrift_with[i])
        drift_type.append("withSys")
        dataset.append(iteration)
        trial.append(i)
        
    for i in range(len(estDrift_without)):
        drift_value.append(estDrift_without[i])
        drift_type.append("withoutSys")
        dataset.append(iteration)
        trial.append(i)


estimated_drifts = pd.DataFrame(data={'trial' : trial,
                                        'dataset' : dataset,
                                        'drift_type' : drift_type,
                                        'drift_value' : drift_value})






estimated_drifts.to_csv("estimation_slowdrifts_withSys.csv")

