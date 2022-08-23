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
        drift[i] = σd*np.random.randn() + 0.9995*drift[i-1]  if i>0 else σd*np.random.randn()

    return drift - np.mean(drift)  
#%%
def simulateChoice(ntrials,    
                sens = 10.,       
                bias = -5.,       
                σd = 0.01,        
                pc=1.,            
                pe=-1.,
                seed = 1):
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

def simulateChoiceConf(ntrials,    
                sens = 10.,       
                bias = -5.,       
                σd = 0.01,        
                pc_conf=0.,            
                pe_conf=0.,
                seed = 1):
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
    
    c, e, conf_c, conf_e = 1, 0, 1, 0   
    for i in range(ntrials):
        inputs[i,0] = inpt[i]
        inputs[i,1] = conf_c
        inputs[i,2] = conf_e

        
        pR = sigmoid(sens*inputs[i,0] + pc_conf*conf_c + pe_conf*conf_e + bias + drift[i])
        choice = np.random.rand() < pR
        
        c = (2*choice - 1)*(choice == rewSide[i]) # +1 right, -1 left, 0 error
        e = (2*choice - 1)*(choice != rewSide[i]) # +1 right, -1 left, 0 correct
        
        conf_c = c * (2 * np.abs(pR - .5))
        conf_e = e * (2 * np.abs(pR - .5))
        
        emissions.append(choice*1)   
            
    return inputs, np.array(emissions), drift 




#%%
def simulateChoiceRespConf(ntrials,    
                sens = 10.,       
                bias = -5.,       
                σd = 0.01,        
                w_prevresp=0.,        # if positive: tendency to repeat, if negative: tendency to alternate     
                w_prevconf=0.,
                w_prevconfprevresp = 0.,
                seed = 1):
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

    
    emissions = [] #responses
    inputs = np.ones((ntrials, 4))
    drift = simDrift(σd, ntrials)
    inpt =  np.round(np.random.rand(ntrials), decimals = 2) #stimulus
    rewSide = [True if i > 0.5 else (np.random.rand() < 0.5) if i == 0.5 else False for i in inpt] #determine which response would be correct
    
    prevresp, prevconf, prevresp_prevconf = 0, 0, 0   
    for i in range(ntrials):
        inputs[i,0] = inpt[i]
        inputs[i,1] = prevresp 
        inputs[i,2] = prevconf
        inputs[i,3] = prevresp_prevconf
    
    
        pR = sigmoid(sens*inputs[i,0] + w_prevresp*prevresp + w_prevconf*prevconf + w_prevconfprevresp*prevresp_prevconf + bias + drift[i])
        choice = np.random.rand() < pR
        
        prevresp = (2*choice - 1) # +1 right, -1 left
        
        prevconf = (2 * np.abs(pR - .5)) if choice == rewSide[i] else (-2 * np.abs(pR - .5)) # between -1 and 1 with 0 being guess level
        
        prevresp_prevconf = prevresp * prevconf # interaction between prevresp and prevconf
        
        
        emissions.append(choice*1)  
        
    return inputs, np.array(emissions), drift   


#%%

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