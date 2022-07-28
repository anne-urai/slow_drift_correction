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
    


def simulateChoice(ntrials,    
                sens = 10.,       
                bias = -5.,       
                σd = 0.01,        
                pc=0.,            
                pe=0.,
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
    inputs = np.ones((ntrials, 5)) 
    drift = simDrift(σd, ntrials)
    inpt =  np.round(np.random.rand(ntrials), decimals = 2)
    rewSide = [True if i > 0.5 else (np.random.rand() < 0.5) if i == 0.5 else False for i in inpt] 
    
    c, e, conf_c, conf_e = 1, 0, 1, 0   
    for i in range(ntrials):
        inputs[i,0] = inpt[i]
        inputs[i,1] = c
        inputs[i,2] = e
        inputs[i,3] = conf_c
        inputs[i,4] = conf_e
        
        pR = sigmoid(sens*inputs[i,0] + pc*c + pe*e + pc_conf*conf_c + pe_conf*conf_e + bias + drift[i])
        choice = np.random.rand() < pR
        
        c = (2*choice - 1)*(choice == rewSide[i])
        e = (2*choice - 1)*(choice != rewSide[i])
        
        conf_c = c * (2 * np.abs(pR - .5))
        conf_e = e * (2 * np.abs(pR - .5))
        
        emissions.append(choice*1)   
            
    return inputs, np.array(emissions), drift 