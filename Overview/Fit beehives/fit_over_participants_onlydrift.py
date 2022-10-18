# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:23:54 2022

@author: u0141056
"""
import sys #needed to load model_fit since there in a different folder
sys.path.append('C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/CHOICE HISTORY BIAS/Correction slow drifts/slow_drift_correction')   

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import model_fit, simulate_choices #contains the fitting funcs



n_iters = 200 # number of iterations the E-M procedure takes
iterations_pp = 1 #don't change

for i in range(iterations_pp): 
    locals()["slowdrifts"+str(i)] = [] #slowdrifts0, slowdrifts1, ...
    


data = pd.read_csv('df_slowdrift_correction.csv')
#data = pd.read_csv('df_squircles_slowdrift.csv')

print(data.columns)



As, Vs, b, Sigmas_init, Sigmas = [],[],[],[],[]
Cs, sens, ds = [],[],[]
prev_resp, prev_conf, prev_resp_conf = [],[],[]
prev_sign_evi, prev_abs_evi, prev_sign_abs_evi = [],[],[]







    
df = data # pick one subject

# restructure dataframe
evidence = df.evidence.to_numpy() # scaled between -1 and 1 (with -1 being left evidence, 1 right evidence, 0 would be ambigious no evidence trial)
# prevresp = df.prevresp.to_numpy() # -1 left, 1 right
# prevconf = df.prevconf.to_numpy() # continuous scale between -1 and 1, -1 sure error, 1 sure correct, 0 guess
# prevrespconf = df.prevrespconf.to_numpy() # interaction between prevreso and prevconf
# prevsignevi = df.prevsignevi.to_numpy() # -1 left, 1 right
# prevabsevi = df.prevabsevi.to_numpy() # continuous scale between 0 and 1 with 0 being no evidence and 1 maximal evidence
# prevsignabsevi = df.prevsignabsevi.to_numpy() # interaction prevsign and prevabsevi

# this is the format that the fitting func needs - vstack as np arrays
#inputs = np.vstack([evidence, prevresp, prevconf, prevrespconf, prevsignevi, prevabsevi, prevsignabsevi]).T
inputs = np.vstack([evidence]).T

choices = np.vstack(np.asarray(df.resp.tolist()).T)


# fit the model
inputDim = np.shape(inputs)[1] # observed input dimensions 
stateDim = 1 # latent states


for iteration in range(iterations_pp):
    
    print("iteration: " + str(iteration))
    
    predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim, inputs, choices, n_iters)

    # convergence
    # fig, axs = plt.subplots(1, 1, figsize=(4,4))
    # axs.plot(elbos[2:], label="Laplace-EM")
    # axs.set(xlabel="Iteration", ylabel="ELBO")
    # axs.legend()
    # sns.despine()

    # slow drift
    state_means = q.mean_continuous_states[0]
    estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]
    
    
    
    drifts = np.squeeze(estDrift)
    drifts_intercept = np.squeeze(estDrift + lds.emissions.ds) 

    for i in range(len(drifts)):
        locals()["slowdrifts"+str(iteration)].append(drifts[i])
    

    fig, axs = plt.subplots(1, 1, figsize=(12,4))
    axs.axhline(0, c = "k", ls = ":", lw =2)
    axs.plot(estDrift[:], c = 'firebrick', label = "Estimated drift")
    axs.set(xlabel = "Trials", ylabel = "Decision criterion")
    axs.legend(loc='upper center', bbox_to_anchor=(.5, 1.25),ncol=2, fancybox=True, shadow=True)
    
    
    lds.dynamics.As
    lds.dynamics.Vs
    lds.dynamics.b
    lds.dynamics.Sigmas_init
    lds.dynamics.Sigmas


    lds.emissions.Cs
    lds.emissions.Fs
    lds.emissions.ds
    
    
    As.append(lds.dynamics.As[0][0][0])
    Vs.append(lds.dynamics.Vs)
    b.append(lds.dynamics.b[0])
    Sigmas_init.append(lds.dynamics.Sigmas_init[0][0][0])
    Sigmas.append(lds.dynamics.Sigmas[0][0][0])
    Cs.append(lds.emissions.Cs[0][0][0])
    sens.append(lds.emissions.Fs[0][0][0])
    # prev_resp.append(lds.emissions.Fs[0][0][1])
    # prev_conf.append(lds.emissions.Fs[0][0][2])
    # prev_resp_conf.append(lds.emissions.Fs[0][0][3])
    # prev_sign_evi.append(lds.emissions.Fs[0][0][4])
    # prev_abs_evi.append(lds.emissions.Fs[0][0][5])
    # prev_sign_abs_evi.append(lds.emissions.Fs[0][0][6])
    ds.append(lds.emissions.ds[0][0])
    
# ignore error 
# drifts_over_subjects = pd.DataFrame(data={'slowdrift0' : slowdrifts0,
#                                         'slowdrift1' : slowdrifts1,
#                                         'slowdrift2' : slowdrifts2,
#                                         'slowdrift3' : slowdrifts3,
#                                         'slowdrift4' : slowdrifts4,
#                                         'slowdrift5' : slowdrifts5,
#                                         'slowdrift6' : slowdrifts6,
#                                         'slowdrift7' : slowdrifts7,
#                                         'slowdrift8' : slowdrifts8,
#                                         'slowdrift9' : slowdrifts9})


drifts_over_subjects = pd.DataFrame(data={'slowdrift0' : slowdrifts0})


params_fits_over_subjects_nosystem = pd.DataFrame(data={
                                         'As' : As,
                                         'Vs' : Vs,
                                         'b' : b,
                                         'Sigmas_init' : Sigmas_init,
                                         'Sigmas' : Sigmas,
                                         'Cs' : Cs,
                                         'sens' : sens,
                                         # 'prev_resp' : prev_resp,
                                         # 'prev_conf' : prev_conf,
                                         # 'prev_resp_conf' : prev_resp_conf,
                                         # 'prev_sign_evi' : prev_sign_evi,
                                         # 'prev_abs_evi' : prev_abs_evi,
                                         # 'prev_sign_abs_evi' : prev_sign_abs_evi,
                                         'ds' : ds})






drifts_over_subjects.to_csv("drifts_over_subjects_nosystemupdating.csv")
params_fits_over_subjects_nosystem.to_csv("params_fits_over_subjects_nosystem.csv")


