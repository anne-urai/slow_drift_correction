# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:44:51 2022

@author: u0141056
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:15:06 2022
@author: urai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from ssm.lds import LDS
from ssm.observations import AutoRegressiveObservations
from ssm.emissions import BernoulliEmissions
from ssm.transitions import StationaryTransitions
from ssm.init_state_distns import InitialStateDistribution

#%% LOAD SOME RANDOM DATA
# from https://figshare.com/articles/dataset/Choice_history_biases_subsequent_evidence_accumulation/7268558/2?file=13593227

#data = pd.read_csv('visual_motion_2afc_rt.csv')
#print(data.columns) # we'll use prevpupil as a stand-in for prevconfidence

# ToDo: reorder so it can be used as inputs to ssm

#%% fit with SSM package
# copied from https://codeocean.com/capsule/0617593/tree/v1




class AutoRegressiveNoInput(AutoRegressiveObservations):
    
    # AutoRegressive observation model with Gaussian noise.
    # (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)
    # which is equal to x_t = A_k x_{t-1} + b_k + V_k u_t + E_k with E_k ~ N(0,S_k)
    # default lag is 1
    
    
    # sigma is updated analytically
    def m_step(self, expectations, datas, inputs, masks, tags,
                   continuous_expectations=None, **kwargs):
        
        K, D, M, lags = self.K, self.D, self.M, self.lags

        # Collect sufficient statistics
        if continuous_expectations is None:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._get_sufficient_statistics(expectations, 
                                                                          datas, 
                                                                          inputs)
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = \
                self._extend_given_sufficient_statistics(expectations, 
                                                         continuous_expectations, 
                                                         inputs)

        Sigmas = np.zeros((K, D, D))
        for k in range(K):
            Wk = np.linalg.solve(ExuxuTs[k] + self.J0[k], ExuyTs[k] + self.h0[k]).T

            # Solve for the MAP estimate of the covariance
            EWxyT =  Wk @ ExuyTs[k]
            sqerr = EyyTs[k] - EWxyT.T - EWxyT + Wk @ ExuxuTs[k] @ Wk.T
            nu = self.nu0 + Ens[k]
            Sigmas[k] = (sqerr + self.Psi0) / (nu + D + 1)

        # If any states are unused, set their parameters to a perturbation of a used state
        unused = np.where(Ens < 1)[0]
        used = np.where(Ens > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = np.choice(used) #npr was standing here first but shouldn't this be np?
                Sigmas[k] = Sigmas[i]

        # Update parameters via their setter
        self.Sigmas = Sigmas
        

# ToDo: merge below?
class LDS_noInputDynamics(LDS):
    def __init__(self, N, D, M=0): #M is input dimensions?

        print('M is:')
        print(M)
        dynamics = AutoRegressiveNoInput(1, D, M=M)
        emissions = BernoulliEmissions(N, 1, D, M=M)

        init_state_distn = InitialStateDistribution(1, D, M)
        transitions = StationaryTransitions(1, D, M)
        
        self.N, self.K, self.D, self.M = N, 1, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions



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

sigmoid = lambda x: 1 / (1 + np.exp(-x)) #function needed below

def simulateChoice(ntrials,    
                sens = 10.,       
                bias = -5.,       
                σd = 0.01,        
                pc=0.,            
                pe=0.,
                seed = 5):
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
    
    c, e = 1, 0  # we can change this to simulate behavior that differs in updating strategies (now win stay, lose-nothing)  
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



def makeDataDict(inputs, emissions, predEmissions = None):  
    ''' returns a dataframe with columns relevant for plotting and analysis 
    Args:
        inputs (array): (ntrialsx3 where columns are stimulus strength, prev correct, prev error) 
                    output of simulateChoice
        emissions (array): (ntrialsx1) choices made by the SDT oberver +1 for rightward choices,
                    0 for leftward.  output of simulateChoice 
        predEmissions (array): (ntrialsx1) another set of choices for the same trials, say from a 
                    model. appended to the dataframe with `Model` column set to 1
    Returns:
        data (DataFrame): dataframe with columns pokedR, Model, StimStrength, PreviousChoice,
                    PreviousOutcome, rc (right correct), lc, re and le
    '''
    data = {}
    if predEmissions is None:
        tileMe = lambda x: x
        data["pokedR"] = np.ravel(emissions)
        data["Model"] = np.zeros(np.shape(inputs)[0])
    else:
        # if fit was given make duplicates with a model indicator
        tileMe = lambda x: np.tile(x,2)
        predpokedR  = 1*(predEmissions > np.random.rand(np.size(predEmissions)))
        data["pokedR"] = np.concatenate((np.ravel(emissions), predpokedR))
        data["Model"] = np.concatenate((np.zeros(np.shape(inputs)[0]), np.ones(np.shape(inputs)[0])))

    data["StimStrength"] = tileMe(inputs[:,0])
    data["PreviousChoice"] = tileMe((inputs[:,1] == 1) | (inputs[:,2] == 1))
    data["PreviousOutcome"] = tileMe((inputs[:,1] == 1) | (inputs[:,1] == -1))
    data["rc"] = tileMe(inputs[:,1] == 1)
    data["lc"] = tileMe(inputs[:,1] == -1) 
    data["re"] = tileMe(inputs[:,2] == 1)
    data["le"] = tileMe(inputs[:,2] == -1)  
        
    return pd.DataFrame(data)


#%% Simulate some data

seed = 200

inputs, emissions, drift = simulateChoice(ntrials = 50000, 
                                            σd = 0.05,
                                            sens = 10,
                                            bias = 0,
                                            pc = 0,
                                            pe = 0, 
                                            seed = seed)

data = makeDataDict(inputs, emissions)

#%% run the fitting func
stateDim = 1 #latent states
inputDim = np.shape(inputs)[1] #observed variables

lds = LDS_noInputDynamics(1, 1, M = inputDim)

# Parameters specified on forehand are not estimated
lds.dynamics.A = np.ones((stateDim,stateDim))   # slow drift dynamics (A is autoregressive coefficient A x_{t-1}) : why is this fixed at 1? rather strong assumption
lds.dynamics.b = np.zeros(stateDim)              # bias
lds.dynamics.mu_init = np.zeros((stateDim,stateDim))    # initial mu
lds.dynamics.Sigmas_init = np.array([[[0.05]]])     # initial sigma
lds.dynamics.Vs = np.array([[np.zeros(inputDim)]])               # input dynamics
#lds.dynamics.Cs = np.ones((stateDim,stateDim))

# elbos: evidence lower bound (lower bound on the log-likelihood of the data)
# used to monitor convergence of the EM algorithm
elbos, q = lds.fit(emissions, inputs = inputs, 
                   method="laplace_em",
                    variational_posterior="structured_meanfield", 
                    continuous_optimizer='newton',
                    initialize=True, 
                    num_init_restarts=1,
                    num_iters=20, 
                    alpha=0.1)


plt.plot(elbos, label="Laplace-EM")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.legend()


# Get the posterior mean of the continuous states (drift)
state_means = q.mean_continuous_states[0]
estDrift = np.squeeze(lds.emissions.Cs)*state_means[:] ###################

# Smooth the data under the variational posterior (to compute emissions)
predEmissions = np.concatenate(lds.smooth(state_means, emissions, input = inputs))



# Plot true and generative drift

plt.figure(figsize=(8,4))
for d in range(stateDim):
    plt.plot(estDrift[:1000], '--', label="Estimated States" if d==0 else None)
    plt.plot(drift[:1000],  "k", lw = 2,label = "Generative drift")
plt.ylabel("$x$")
plt.xlabel("time")
plt.legend(loc='upper right')
plt.title("True and Estimated States")
plt.show()



A_est = lds.dynamics.A
b_est = lds.dynamics.b


lds.emissions.Cs
lds.emissions.Fs
lds.emissions.ds


lds.emissions.D

lds.emissions.K
lds.emissions.D









