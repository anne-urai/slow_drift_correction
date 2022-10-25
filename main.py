#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:15:06 2022
@author: urai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to build our model
import model_fit, model_fitAR_hierarchical, simulate_choices #contains the fitting funcs


simulate = True
fitAR = True # fit model where AR coef for slow drifts is estimated, else original Gupta

# see https://github.com/lindermanlab/ssm/blob/master/notebooks/1b%20Simple%20Linear%20Dynamical%20System.ipynb
# see also https://github.com/lindermanlab/ssm/blob/master/ssm/lds.py#L825

# K = 1       # number of discrete states: 1, no switching
# D = 1       # number of latent dimensions: 1, AR model
# N = 1       # number of observed dimensions: only choices. Could add in RT/confidence?
# M = 3       # input dimensions


# is it possible that when fitting on real data, the systematic updating doesn't change a lot
# because confidence itself is influenced by the slow drifts, so the influence of slow drift remains 
# and is not corrected by estimating the slow drifts?

# fit model without slow drifts, compare with mixed models -> check, perfect match
# simulate with post decisional evidence accumulation
# plotting!
# simulate with more realistic values (not 1)
# compare three models on real data (1 participant/whole dataset)
# simulate different observers, merge into one df, and see how model behaves 
# is it an option to implement a hierarchical version? fitted on all participants simulateneously, but allowing individual parameters
# see ssm.hierarchical


# try to estimate effect prev resp on latent model? -> not possible given how the model is coded

# would effect coding play a role when fitting the data?
# what are realistic simulation values?


# What if we simulate data with only slow drifts (no systematic updating)
# eg 50 datasets, then fit model, and see what the range of estimates is
# for example: is a beta for prev resp of .1 normal to find without any simulated effects

# EM:
    # E step: calculate expected log likelihood with previous parameters
    # M step: improve parameters based on E step -> sort of gradient descent
    #         to find parameters with higher likelihood?
    
#%% Attempt to fix C to 1 and estimate A freely

# see file ssm.emissions
## class _LinearEmissions line 132 and 136: change definition params because this is used in m-step (line 86)
## This seems to work, C is not fitted anymore, but we don't have control over it
## Sometimes it becomes -1 or 1, and if we change the intial value to eg 2, then it doesn't become 2
## On line 178 we see initialize with PCA, if we comment out line 200 and 204
## then we are able to change the value of C to what we want (eg 2)
## With C = 1 we get bad estimates for A (eg .52 which is way too low, should be around .99)
## But Diksha removed to estimation code for A to keep it fixed at 1
## So maybe this .52 is just the result of a tradeoff with other parameters,
## which emerges now because it's not fixed, and isn't actually estimated at all
## So in model_fit.py class AutoRegressiveNoInput def m_step we comment out all the code
## for Vs and Bs. But horrible fits (huge sigma and A super low)
## But maybe the model needs these parameters to solve the linear equation
## see np.linalg.solve in def m_step
## So we include all the code for Vs and Bs, but we just don't update it
## such that it remains 0, like we initialized it
## And indeed, the A and sigma estimates are spot on!!!
## Note: more iterations might be needed (500)
## Note: if you simulate with AR coef = .99 then for some reason
## the simulated sigma has to be higher (eg .1 or higher) in order to get
## good fit. Maybe a low sigma with AR coef = .99 falls out the parameter range
## that makes sense?


# if we simulate drift but we then substract the mean(drift) from it
# we actually have generated a sequence with a certain AR coef but also with an intercept,
# namely -mean(drift). This is problematic because we don't estimate an intercept in our AR model
# so what's the result, NEGATIVE AR1 coef! 
# how can we avoid this? estimate an intercept in our AR model, and fix the intercept in our emissions to unbiased
# test if this solution works with effect coded data!
# because using this method on our real data, still weird negative AR coefs
# so maybe change data to dummy coding? stimulus from 0 to 1 ...
# make sure that lds.emissions.ds = np.array([[[-5]]]) is correct!!!

# simulation with parameters below: if we change to 10000 trials then it doesn't work anymore,
# even though the parameters are the same, look whats wrong here. maybe something with intercept?
#%% simulate choice data
if simulate:
    ntrials = 10000 # original code: 40.000
    sens = 10
    bias = 0 #is unbiased # if effect coded simulations 0 is unbiased!
    σd = 0.1
    
    
    w_prevresp = 1
    w_prevconf = 0
    w_prevrespconf = 1
    w_prevsignevi = 0
    w_prevabsevi = 0
    w_prevsignabsevi = 0
    
    
    
    inputs, choices, drift = simulate_choices.simulateChoice_normalEvi_slowdriftConf(ntrials,
                                              estimateUpdating = False,
                                              fixedConfCriterion = True,
                                              postDecisionalEvi = True,
                                              σd = σd, 
                                              sens = sens, bias = bias, 
                                              w_prevresp = w_prevresp, 
                                              w_prevconf = w_prevconf,
                                              w_prevrespconf = w_prevrespconf,
                                              w_prevsignevi = w_prevsignevi,
                                              w_prevabsevi = w_prevabsevi,
                                              w_prevsignabsevi = w_prevsignabsevi,
                                              seed = 5)
    # pc = 0
    # pe = 0

    # inputs, choices, drift = simulate_choices.simulateChoice(ntrials, σd = σd, 
    #                                           sens = sens, bias = bias,
    #                                           pc = pc, pe = pe)
    
    # inputs, choices, drift = simulate_choices.simulateChoiceEffectCoded(ntrials, σd = σd, 
    #                                           sens = sens, bias = bias,
    #                                           pc = pc, pe = pe)
                    
    '''
    inputs (array): ntrialsx3, first column is stimulus strength, second column is indicator for 
                post-correct trials (+1 right, -1 left, 0 error) and third column is indicator
                for post-error trials (+1 right, -1 left, 0 correct) 
    choices (array): ntrialsx1 choices made by the SDT oberver +1 for rightward choices,
                0 for leftward
    drift (array): ntrialsx1, mean centered 
    '''
    
else:  # load real data

    data = pd.read_csv('df_slowdrift_correction.csv')
    #data = pd.read_csv('df_squircles_slowdrift.csv')
    #data = pd.read_csv('simData_slowdriftR.csv') #from R slow drift script -> slow drift, conf influenced by slow drift, no systematic updating
    #data = pd.read_csv('simData_slowdriftR_fixedconf.csv')
    
    
    print(data.columns)
    
    # choose whether fitting whole dataset or a specific subject
    #df = data
    df = data[data.subj == 1] # pick one subject
    
    # in case of NA on first line
    #df = df.iloc[1: , :] #remove first trial
    
        
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



#%% fit the LDS model
inputDim = np.shape(inputs)[1] # observed input dimensions 
stateDim = 1 # latent states
n_iters = 200


if fitAR:
    predEmissions, estDrift, lds, q, elbos = model_fitAR_hierarchical.initLDSandFitAR(inputDim,inputs, choices,n_iters)

else:
    predEmissions, estDrift, lds, q, elbos = model_fit.initLDSandFit(inputDim,inputs, choices,n_iters)

# check autocorrelation of input variables
s = inputs[:,0]
x = pd.plotting.autocorrelation_plot(s)
x.set_xlim([0, 50])
x.set_ylim([-.25, .25])
x.plot()
plt.show()

#%%% Get the posterior mean of the continuous states (drift)

'''
Gupta: sigma_d is the parameter that determines the degree to which the random 
noise is added to the signal. To infer the specific changes in slow drift from trial-to-trial, 
we need to estimate the conditional posterior density of the latent variable x_{t} aka drift 
given the observed choices y_{1...T}  - i.e. p(x_t | y_{1...T}). This operation is often called 
"smoothing" in latent variable models if T>t . 
Essentially, given the initial value of the random drift, the standard deviation specifies 
the distribution of values that the drift can take on the next trial. But, given that we 
observed the choice in the next trial, we can leverage that information to further sharpen 
the estimate of what the value of drift might have been. Macke 2015 is my go-to reference 
for reading about this, they consider a Poisson noise model with LDS, but essentially the 
same notions apply to a Bernoulli LDS (as in the manuscript).
'''

state_means = q.mean_continuous_states[0]

# If drift was simulated without mean-correction then adding intercept (lds.emissions.ds)
# to the estimated drift will result in a better overlap with the generative drift
# It seems that lds.emissions.ds captures the mean of the slow drift

estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]
#estDrift = np.squeeze(lds.emissions.Cs)*state_means[:] + lds.emissions.ds

# Smooth the data under the variational posterior (to compute emissions aka choices)
predEmissions = np.concatenate(lds.smooth(state_means, choices, input = inputs))

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
axs.legend(loc='upper center', bbox_to_anchor=(.5, 1.25),ncol=2, fancybox=True, shadow=True)

# mse as a metric for distance between simulated and estimated slow drift
# (as proxy for goodness of fit without looking at the actual plot)
#mse = sum((drift - estDrift)**2) / ntrials

lds.dynamics.As
lds.dynamics.Vs
lds.dynamics.b
lds.dynamics.Sigmas_init
lds.dynamics.Sigmas


lds.emissions.Cs
t = lds.emissions.Fs
lds.emissions.ds # positive bias indicates an overall bias towards right responses



# estimated_drift = pd.DataFrame(np.squeeze(estDrift))
# estimated_drift.to_csv("estimateddrifts_beehives_allsubjectsAR_sys_intercept.csv")



# df_input = pd.DataFrame(inputs)
# df_input.to_csv("df_input.csv")

# df_choices = pd.DataFrame(choices)
# df_choices.to_csv("df_choices.csv")

# df_drift = pd.DataFrame(drift)
# df_drift.to_csv("df_drift.csv")

# df_estdrift = pd.DataFrame(estDrift)
# df_estdrift.to_csv("df_estdrift.csv")






# for plotting psychometric function in PC/PE scenario

# import numpy as np
# import seaborn as sns
# from plotting import plotCondPsych,makeDataDict

# XVAL_DEF = np.linspace(0, 1, 101).reshape(-1,1) # xvals for computing psychometrics
# sigmoid = lambda x: 1 / (1 + np.exp(-x))

# col_pc = sns.color_palette('Greens', n_colors = 1)[0]
# col_pe = sns.color_palette('Reds', n_colors = 1)[0]

# colors_c = sns.color_palette('Greens', n_colors = 2)
# colors_e = sns.color_palette('Reds', n_colors = 2)




# data = makeDataDict(inputs, choices, predEmissions = predEmissions)

# # psychometric systematic updates and slow drifts (generative and model)
# fig, axs = plt.subplots(1,2, figsize=(12,4), subplot_kw=dict(box_aspect=1) )
# plotCondPsych(data[data.Model == 0], True, axs[0], ls = '-')
# plotCondPsych(data[data.Model == 0], False, axs[1], ls = '-')
# plotCondPsych(data[data.Model == 1], True, axs[0], ls = (0, (3, 1, 1, 1)))
# plotCondPsych(data[data.Model == 1], False, axs[1], ls = (0, (3, 1, 1, 1)))   
# axs[1].set_ylabel("")
# axs[1].get_xaxis().set_visible(True)
# axs[0].get_xaxis().set_visible(True)
# sns.despine()



# # psychometric after removing slow drift (generative and model)
# fig, axs = plt.subplots(1,2, figsize=(12,4), subplot_kw=dict(box_aspect=1) )
# for i, ax in enumerate(axs):
#     p = pc if i == 0 else pe
#     col = colors_c if i == 0 else colors_e
#     ax.plot(XVAL_DEF, sigmoid(sens*XVAL_DEF + bias + p), c = col[1], lw = 2)
#     ax.plot(XVAL_DEF, sigmoid(sens*XVAL_DEF + bias - p), c = col[0], lw = 2)
#     ax.plot(XVAL_DEF, sigmoid(lds.emissions.Fs[0,0,0]*XVAL_DEF + lds.emissions.ds + lds.emissions.Fs[0,0,i+1]), 
#                         ls = (0, (3, 1, 1, 1)), c = col[1], lw = 3)
#     ax.plot(XVAL_DEF, sigmoid(lds.emissions.Fs[0,0,0]*XVAL_DEF + lds.emissions.ds - lds.emissions.Fs[0,0,i+1]), 
#                         ls = (0, (3, 1, 1, 1)), c = col[0], lw = 3)
#     ax.set_ylabel("Fraction chose right")
#     ax.set_xlabel("Stimulus strength")
    
# axs[1].set_ylabel("")
# sns.despine()
