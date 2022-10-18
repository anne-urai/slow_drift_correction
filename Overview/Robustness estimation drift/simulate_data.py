# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:30:05 2022

@author: u0141056
"""


import numpy as np
import pandas as pd

sigmoid = lambda x: 1 / (1 + np.exp(-x)) # from config.py

#%%

def simDrift(σd, ntrials):
    '''function to simulate random drift, discrete-time OU process with a fixed 
    small tendency (lambda = 0.0005) to decay to zero 
    Args:
        σd (float): standard deviation of the gaussian noise
        ntrials (int): number of trials 
    Returns:
        drift (array): ntrialsx1, mean centered 
    '''
    drift = np.zeros((ntrials,1))
    for i in range(ntrials):
        drift[i] = σd*np.random.randn() + .9995 * drift[i-1]  if i>0 else 0
        
    #return drift - np.mean(drift)  
    
    return drift

#%%
def simulate_estimationWithEffects(ntrials,    
                 sens = 10.,       
                 bias = 0.,       
                 σd = 0.1,
                 sigma_evidence = 1.,        
                 dprime = 1.,            
                 w_prevresp = 0.,
                 w_prevconf = 0.,
                 w_prevrespconf = 0.,
                 w_prevsignevi = 0.,
                 w_prevabsevi = 0.,
                 w_prevsignabsevi = 0.,
                 seed = 1):

    np.random.seed(seed)
    
    
    emissions = []    
    inputs = np.ones((ntrials, 7)) #evidence, prevresp, prevconf, prevrespconf, prevsignevi, prevabsevi, prevsignabsevi
    drift = simDrift(σd, ntrials)
    
    
    # generate two evidence lists: evidence1 for response, evidence2 for calculation confidence
    evidence1 = list(-dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2))) + list(dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2)))
    evidence2 = list(-dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2))) + list(dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2)))
    rewSide = list(np.repeat(0, ntrials/2)) + list(np.repeat(1, ntrials/2))
    
    df_evi = pd.DataFrame(data={'evidence1' : evidence1, 'evidence2' : evidence2, 'rewSide': rewSide})
    
    df = df_evi.sample(frac=1).reset_index(drop=True) #shuffle order rows
    
    
    prevresp, prevconf, prevrespconf, prevsignevi, prevabsevi, prevsignabsevi = 0, 0, 0, 0, 0, 0 # initialize for first trial
    for i in range(ntrials):
        inputs[i,0] = df.evidence1[i]
        inputs[i,1] = prevresp
        inputs[i,2] = prevconf
        inputs[i,3] = prevrespconf
        inputs[i,4] = prevsignevi
        inputs[i,5] = prevabsevi
        inputs[i,6] = prevsignabsevi
    
        # compare evidence with criterion
        # what is criterion here? all terms except the actual evidence
        # so slow drift, bias, prev resp and prev conf
        
        crit = bias + w_prevresp*prevresp + w_prevconf*prevconf + w_prevrespconf*prevrespconf + w_prevsignevi*prevsignevi + w_prevabsevi*prevabsevi + w_prevsignabsevi*prevsignabsevi+ drift[i]
        
        
        # response 
        pR = sigmoid(sens*df.evidence1[i] + crit)
        choice = np.random.rand() < pR # draw from bernoulli with probability right response
        
        # calculation confidence (prev confidence)
        sample1 = sigmoid(sens*df.evidence1[i] + crit) - .5
        sample2 = sigmoid(sens*df.evidence2[i] + crit) - .5
        prevconf = (sample1 + sample2) if choice[0] else -(sample1 + sample2)
        
        # previous response
        prevresp = (2*choice - 1) # +1 right, -1 left
        
        # prev resp * prev conf
        prevrespconf = prevresp * prevconf # interaction between prevresp and prevconf
        
        prevsignevi = -1 if df.evidence1[i] < 0 else 1
        prevabsevi = np.abs(df.evidence1[i])
        prevsignabsevi = prevsignevi * prevabsevi
        
        
        emissions.append(choice*1)   
        
    return inputs, np.array(emissions), drift 
    


#%%


# Same as above, but systematic effects are not saved in inputs and therefore also not estimated

def simulate_estimationWithoutEffects(ntrials,    
                 sens = 10.,       
                 bias = 0.,       
                 σd = 0.1,
                 sigma_evidence = 1.,        
                 dprime = 1.,            
                 w_prevresp = 0.,
                 w_prevconf = 0.,
                 w_prevrespconf = 0.,
                 w_prevsignevi = 0.,
                 w_prevabsevi = 0.,
                 w_prevsignabsevi = 0.,
                 seed = 1):

    np.random.seed(seed)
    
    
    emissions = []    
    inputs = np.ones((ntrials, 1)) #evidence, prevresp, prevconf, prevrespconf, prevsignevi, prevabsevi, prevsignabsevi
    drift = simDrift(σd, ntrials)
    
    
    # generate two evidence lists: evidence1 for response, evidence2 for calculation confidence
    evidence1 = list(-dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2))) + list(dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2)))
    evidence2 = list(-dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2))) + list(dprime/2 + sigma_evidence*np.random.randn(int(ntrials/2)))
    rewSide = list(np.repeat(0, ntrials/2)) + list(np.repeat(1, ntrials/2))
    
    df_evi = pd.DataFrame(data={'evidence1' : evidence1, 'evidence2' : evidence2, 'rewSide': rewSide})
    
    df = df_evi.sample(frac=1).reset_index(drop=True) #shuffle order rows
    
    
    prevresp, prevconf, prevrespconf, prevsignevi, prevabsevi, prevsignabsevi = 0, 0, 0, 0, 0, 0 # initialize for first trial
    for i in range(ntrials):
        inputs[i,0] = df.evidence1[i]
        # inputs[i,1] = prevresp
        # inputs[i,2] = prevconf
        # inputs[i,3] = prevrespconf
        # inputs[i,4] = prevsignevi
        # inputs[i,5] = prevabsevi
        # inputs[i,6] = prevsignabsevi
    
        # compare evidence with criterion
        # what is criterion here? all terms except the actual evidence
        # so slow drift, bias, prev resp and prev conf
        
        crit = bias + w_prevresp*prevresp + w_prevconf*prevconf + w_prevrespconf*prevrespconf + w_prevsignevi*prevsignevi + w_prevabsevi*prevabsevi + w_prevsignabsevi*prevsignabsevi+ drift[i]
        
        
        # response 
        pR = sigmoid(sens*df.evidence1[i] + crit)
        choice = np.random.rand() < pR # draw from bernoulli with probability right response
        
        # calculation confidence (prev confidence)
        sample1 = sigmoid(sens*df.evidence1[i] + crit) - .5
        sample2 = sigmoid(sens*df.evidence2[i] + crit) - .5
        prevconf = (sample1 + sample2) if choice[0] else -(sample1 + sample2)
        
        # previous response
        prevresp = (2*choice - 1) # +1 right, -1 left
        
        # prev resp * prev conf
        prevrespconf = prevresp * prevconf # interaction between prevresp and prevconf
        
        prevsignevi = -1 if df.evidence1[i] < 0 else 1
        prevabsevi = np.abs(df.evidence1[i])
        prevsignabsevi = prevsignevi * prevabsevi
        
        
        emissions.append(choice*1)   
        
    return inputs, np.array(emissions), drift 
    


#%%