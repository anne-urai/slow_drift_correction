#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:00:22 2022

@author: urai
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
        
    return drift - np.mean(drift)  # original Gupta
    
    #return drift # more realistic when drift is not mean corrected
 
#%%
def simulateChoice(ntrials,    
                sens = 10.,       
                bias = -5.,       
                σd = 0.01,        
                pc=1.,            
                pe=-1.,
                seed = 1234):
    '''simulates choices from a SDT observer with logisitic noise and trial-history effects
    Args:
        ntrials (int): number of trials
        sens (float): sensitivity of the agent
        bias (float): fixed component of the decision criterion
        σd (float, positive): standard deviation of the drifting component of decision criterion
        pc (float): bias in decision criterion induced by correct choices, +pc following rightward 
                    correct choices, and -pc following leftward
        pe (float): bias in decision criterion induced by incorrect choices, +pe following rightward 
                    incorrect choices, and -pe following leftward
        seed (int): seed for random number generation
    Returns:
        inputs (array): ntrialsx3, first column is stimulus strength, second column is indicator for 
                    post-correct trials (+1 right, -1 left, 0 error) and third column is indicator
                    for post-error trials (+1 right, -1 left, 0 correct) 
        emissions (array): ntrialsx1 choices made by the SDT oberver +1 for rightward choices,
                    0 for leftward
        drift (array): ntrialsx1, mean centered 
    '''
    np.random.seed(seed)

    emissions = []    
    inputs = np.ones((ntrials, 3)) 
    drift = simDrift(σd, ntrials)
    inpt =  np.round(np.random.rand(ntrials), decimals = 2)
    rewSide = [True if i > 0.5 else (np.random.rand() < 0.5) if i == 0.5 else False for i in inpt] 
    
    c, e = 1, 0
    for i in range(ntrials):
        inputs[i,0] = inpt[i]
        inputs[i,1] = c
        inputs[i,2] = e

        pR = sigmoid(sens*inputs[i,0] + pc*c + pe*e + bias + drift[i])
        choice = np.random.rand() < pR
        
        c = (2*choice - 1)*(choice == rewSide[i])
        e = (2*choice - 1)*(choice != rewSide[i])
        
        emissions.append(choice*1)   
            
    return inputs, np.array(emissions), drift 


#%%

'''
Same as Simulatechoice but now evidence is drawn from uniform distribution with range [-1,1] instead of [0,1]
As a consequence, the intercept d will be unbiased when 0.
When evidence is between [0,1], the intercept will be unbiased when equal to -sensitivity/2
with sensitivity being the slope of the stimulus.
'''

def simulateChoiceEffectCoded(ntrials,    
                sens = 10.,       
                bias = 0.,       
                σd = 0.01,        
                pc=1.,            
                pe=-1.,
                seed = 1):

    np.random.seed(seed)

    emissions = []    
    inputs = np.ones((ntrials, 3)) 
    drift = simDrift(σd, ntrials)
    inpt =  np.round(np.random.uniform(low=-1, high=1, size=ntrials), decimals = 2)
    rewSide = [True if i > 0 else (np.random.rand() < 0.5) if i == 0 else False for i in inpt] 
    
    c, e = 1, 0
    for i in range(ntrials):
        inputs[i,0] = inpt[i]
        inputs[i,1] = c
        inputs[i,2] = e

        pR = sigmoid(sens*inputs[i,0] + pc*c + pe*e + bias + drift[i])

        choice = np.random.rand() < pR
        
        c = (2*choice - 1)*(choice == rewSide[i])
        e = (2*choice - 1)*(choice != rewSide[i])
        
        emissions.append(choice*1)   
            
    return inputs, np.array(emissions), drift 



#%%

'''
- Evidence is drawn from 2 normal distributions (cf. signal detection theory)
- Confidence is calculated by comparing evidence to criterion (which is influenced by slow drifts and systematic updating)
  and allowing post-decisional evidence.
- Argument estimateUpdating is added to allow the estimation of the systematic updating or not

In SDT confidence is calculated by comparing evidence to criterion. However, the lowest that confidence can go is zero (being guess correct)
So we present second evidence sample, and compare this with criterion. Combining these two can then result in stronger confidence, 
if the second confirms the first, or lower (and even negative confidence, perceived errors),  when the second disproves the first
cf. post-decisional evidence accumulation 
we make our decision based on first evidence sample, but there is still some evidence in the pipeline (participants are processing most recently seen information)
when giving our confidence judgements, this additional evidence is then incorporated in cj
cf. psychophysicial kernels beehives


'''

def simulateChoice_normalEvi_slowdriftConf(ntrials, 
                 estimateUpdating = True,
                 fixedConfCriterion = False,
                 postDecisionalEvi = True,
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
    
    
    # model will estimate a slope for each variable given in inputs
    # so if systematic updating (prev resp, prev conf...) shouldn't be estimated, then don't include it in input
    inputs = np.ones((ntrials, 7)) if estimateUpdating else np.ones((ntrials, 1)) #evidence, prevresp, prevconf, prevrespconf, prevsignevi, prevabsevi, prevsignabsevi
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
        
        if estimateUpdating: # save systematic updating in input only when it has to be estimated in model
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
        # sens governs the steepness of the psychometric function
        # in SDT this would affect the width of the stimulus distribution,
        # the higher sens, the smaller the distribution
        pR = sigmoid(sens*df.evidence1[i] + crit)
        
        choice = np.random.rand() < pR # draw from bernoulli with probability right response

        # calculation confidence (prev confidence)
        # we want confidence between -1 (sure error) and 1 (sure correct), with 0 being guess
        # if sens*evidence is equal to crit then sigmoid will result in .5
        # so we subtract -.5 to get this to 0 (confidence when guessing)
        
        if fixedConfCriterion: 
            crit = 0
            
        sample1 = sigmoid(sens*df.evidence1[i] + crit) - .5
        sample2 = sigmoid(sens*df.evidence2[i] + crit) - .5
        # if choice[0] (right choice) then add up because two samples will be positive when correct
        # else (left choice), add up and change sign because sum will be negative if correct choice
        
        if postDecisionalEvi:
            prevconf = (sample1 + sample2) if choice[0] else -(sample1 + sample2)
            
        else:
            prevconf = sample1 if choice[0] else -sample1
            
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

'''
Old definition used to generate figure with prev resp, prev conf... on Github repo
Not realistic because response is made based on biased criterion (due to slow drifts)
But confidence is calculated by comparing evidence to unbiased criterion.
Why would an observer use biased criterion for making a response, but have access to an unbiased criterion
to calculate confidence? 
'''
def simulateChoiceRespConfEvi(ntrials,    
                        sens = 10.,     
                        bias = -5.,    
                        σd = 0.01,     
                        w_prevresp = 0., # if positive: tendency to repeat, if negative: tendency to alternate
                        w_prevconf = 0.,
                        w_prevconfprevresp = 0.,
                        w_prevsignevi = 0.,
                        w_prevabsevi = 0.,
                        w_prevsignevi_prevabsevi = 0.,
                        seed = 1):

    
    np.random.seed(seed)
    
    emissions = [] #responses
    inputs = np.ones((ntrials, 7)) #are used as observed variables to fit the model
    drift = simDrift(σd, ntrials) 
    inpt =  np.round(np.random.rand(ntrials), decimals = 2) #stimulus
    rewSide = [True if i > 0.5 else (np.random.rand() < 0.5) if i == 0.5 else False for i in inpt] #determine which response would be correct
    
    prevresp, prevconf, prevresp_prevconf, prevsignevi, prevabsevi, prevsignevi_prevabsevi = 0, 0, 0, 0, 0, 0
    for i in range(ntrials):
        inputs[i,0] = inpt[i]
        inputs[i,1] = prevresp 
        inputs[i,2] = prevconf
        inputs[i,3] = prevresp_prevconf
        inputs[i,4] = prevsignevi
        inputs[i,5] = prevabsevi
        inputs[i,6] = prevsignevi_prevabsevi
    
        # chance of right response
        pR = sigmoid(bias +
                     sens*inputs[i,0] + 
                     w_prevresp*prevresp + # if previous response is -1, and the weight is positive (attractive effect)
                                           # then this will cause a lower chance of a right response
                     w_prevconf*prevconf + 
                     w_prevconfprevresp*prevresp_prevconf +
                     w_prevsignevi * prevsignevi +
                     w_prevabsevi * prevabsevi +
                     w_prevsignevi_prevabsevi * prevsignevi_prevabsevi + 
                     drift[i])
        
        # draw from bernoulli with probability right response
        choice = np.random.rand() < pR
        
    
        prevresp = (2*choice - 1) # +1 right, -1 left
        prevconf = (2 * np.abs(pR - .5)) if choice == rewSide[i] else (-2 * np.abs(pR - .5)) # between -1 and 1 with 0 being guess level
        prevresp_prevconf = prevresp * prevconf # interaction between prevresp and prevconf
        prevsignevi = -1 if inpt[i] < .50 else 1
        prevabsevi = np.abs(inpt[i] - .50)
        prevsignevi_prevabsevi = prevsignevi * prevabsevi
        
        emissions.append(choice*1)   
        
    return inputs, np.array(emissions), drift   
