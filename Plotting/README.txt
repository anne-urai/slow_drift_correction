4 number code: 1111
first number: data simulated with slow drifts (1 yes, 0 no)
second number: data simulated with systematic confidence learning (1 yes, 0 no)
third number: model estimates slow drifts (1 yes, 0 no)
fourth number: model estimates systematic confidence updating (1 yes, 0 no)


note: 
- if simulate without slow drifts: just set sigma_d to 0 in simulation script
- if model shouldn't estimate slow drifts: 
	- comment out code under K,D,M in model_fit.py ; class AutoregressiveNoInput; def m_step
	- def initLDSandFit ; lds.dynamics.Sigmas_init = np.array([[[0]]]), lds.dynamics.Sigmas = np.zeros(lds.dynamics.Sigmas.shape)