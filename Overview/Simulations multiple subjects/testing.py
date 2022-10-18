# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:43:28 2022

@author: u0141056
"""

import sys #needed to load model_fit and simulate_choices since there in a different folder
sys.path.append('C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/slow_drift_correction')   

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import simulate_choices, model_fit #contains the fitting funcs



sigmoid = lambda x: 1 / (1 + np.exp(-x))



n_subjects = 90
df_inputs = []
df_choices = []
df_drift = []

np.random.seed(seed=1234)

df_prevresp = 0 + 1*np.random.randn(n_subjects)
np.mean(df_prevresp)

df_prevrespconf = 0 + 1*np.random.randn(n_subjects)
np.mean(df_prevrespconf)

df_prevsignevi = 0 + .5*np.random.randn(n_subjects)
np.mean(df_prevsignevi)

df_prevsignabsevi = 0 + .5*np.random.randn(n_subjects)
np.mean(df_prevsignabsevi)
for sub in range(n_subjects):
    
    inputs, choices, drift = simulate_choices.simulateChoice_normalEvi_slowdriftConf(ntrials = 500, 
                     estimateUpdating = True,
                     fixedConfCriterion = False,
                     postDecisionalEvi = True,
                     sens = 10.,       
                     bias = 0.,       
                     σd = 0.1,
                     sigma_evidence = 1.,        
                     dprime = 1.,            
                     w_prevresp = df_prevresp[sub],
                     w_prevconf = 0.,
                     w_prevrespconf = df_prevrespconf[sub],
                     w_prevsignevi = 0, #df_prevsignevi[sub],
                     w_prevabsevi = 0.,
                     w_prevsignabsevi = 0,# df_prevsignabsevi[sub],
                     seed = sub) #if not changing, every person has same drift
    if sub == 0:
        df_inputs = inputs
        df_choices = choices
    else:
        df_inputs = np.vstack([df_inputs,inputs])
        df_choices = np.vstack([df_choices,choices])

    df_drift = np.append(df_drift,drift)


inputDim = np.shape(df_inputs)[1] # observed input dimensions 
stateDim = 1 # latent states
n_iters = 50
predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,df_inputs, df_choices,n_iters)


fig, axs = plt.subplots(1, 1, figsize=(4,4))
axs.plot(elbos[2:], label="Laplace-EM")
axs.set(xlabel="Iteration", ylabel="ELBO")
axs.legend()
sns.despine()


fig, axs = plt.subplots(1, 1, figsize=(12,4))
axs.axhline(0, c = "k", ls = ":", lw =2)
axs.plot(estDrift[:2000], c = 'firebrick', label = "Estimated drift")
axs.plot(df_drift[:2000], "k", label = "Generative drift")
axs.set(xlabel = "Trials", ylabel = "Decision criterion")
axs.legend(loc='upper center', bbox_to_anchor=(.5, 1.25),ncol=2, fancybox=True, shadow=True)


lds.dynamics.As
lds.dynamics.Vs
lds.dynamics.b
lds.dynamics.Sigmas_init
lds.dynamics.Sigmas


lds.emissions.Cs
tt = lds.emissions.Fs
lds.emissions.ds # positive bias indicates an overall bias towards right responses



# df_input = pd.DataFrame(df_inputs)
# df_input.to_csv("df_input.csv")

# df_choice = pd.DataFrame(df_choices)
# df_choice.to_csv("df_choices.csv")

# df_drifts = pd.DataFrame(df_drift)
# df_drifts.to_csv("df_drift.csv")

# df_estdrift = pd.DataFrame(estDrift)
# df_estdrift.to_csv("df_estdrift.csv")
