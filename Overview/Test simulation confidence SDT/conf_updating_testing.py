# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:43:28 2022

@author: u0141056
"""


import numpy as np
import pandas as pd

sigmoid = lambda x: 1 / (1 + np.exp(-x))

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
        
    return drift - np.mean(drift)  
    #return drift
    
def simulateChoice_normalEvi_slowdriftConf(ntrials,    
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



# def simulateChoice_normalEvi_slowdriftConf(ntrials,    
#                 sens = 10.,       
#                 bias = 0.,       
#                 σd = 0.1,        
#                 pc=1.,            
#                 pe=-1.,
#                 seed = 1):

    
# In SDT confidence is calculated by comparing evidence to criterion.
# However, the lowest that confidence can go is zero (being guess correct)
# So we present second evidence sample, and compare this with criterion
# Combining these two can then result in stronger confidence, if the second confirms the first,
# or lower (and even negative confidence, perceived errors),  when the second disproves the first
# cf. post-decisional evidence accumulation 
# we make our decision based on first evidence sample, but there is still some evidence in the pipeline
# when giving our confidence judgements, this additional evidence is then incorporated in cj
# cf. psychophysicial kernels beehives

ntrials = 1000 # has to be even!
sens = 10  
bias = 0  
σd = 0.1   
sigma_evidence = 1
dprime = 1

w_prevresp = 0
w_prevconf = 0
w_prevrespconf = 0
w_prevsignevi = 0
w_prevabsevi = 0
w_prevsignabsevi = 0

seed = 1



np.random.seed(seed)

emissions = []    

inputs = np.ones((ntrials, 9)) #evidence, prevresp, prevconf, prevrespconf, used in the model
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
    
    
    inputs[i,7] = choice*1
    inputs[i,8] = df.rewSide[i]
    emissions.append(choice*1)   
    
#return inputs, np.array(emissions), drift 

inputs = pd.DataFrame(inputs)

inputs.to_csv("checkconf.csv")
