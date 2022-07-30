# slow_drift_correction
Fitting the model from Gupta &amp; Brody to human behavioral data. Based on https://codeocean.com/capsule/0617593/tree/v1.

By Anne Urai, 2022

---

#### Create a clean conda env
```
conda create -n ssm_env
conda activate ssm_env
conda install cython
conda install seaborn
git clone https://github.com/lindermanlab/ssm.git
cd ssm
pip install -e .
```


### Open questions
- what is the minimum number of trials needed to retrieve slow drift + history weights?
    - How does this depend on the size of the noise sigma?
- is there a numerical determinant of ELBO convergence? -> no, have to inspect visually
- how many `num_iter` do we need?
- how do `predEmissions` compare to binary choices?
- nice to have: printing function (with explanation of parameters)?
- does it matter that some trials are deleted due to timed-out responses?
- what if simulated drift has a AR coef lower than .9995, can the model still account for this given that a random walk being imposed?

### Questions for Diksha
- why is `A = 1`, rather than estimated (while fitting `C`)? Because now a random walk is imposed
- how is the analytical computation of sigma done? why analytical 
(estimated sigma seems to explode with higher number of trials -  insert image)? 
- how to interpret `C` (sometimes negative) and `sigma`? 
- why is `estDrift` computed by multiplying `Cs`? Sort of scaling to compensate for A = 1?
- would analysis would with only 500 trial per participant? or would a switching LDS where the drifting criterion jumps between observers be better?
- why all these functions with `noInput` and not use the default function of the package?
- why is drift mean centered in data simulation?

### Plan van aanpak
1. [ ] answer: with our trial counts, can we even expect to get good fits?
     - [x] try simulations with data-ish parameters
2. [ ] add confidence-scaling (use previous R simulations and fit with ntrials = 500)
3. [ ] fit to real data from Beehives or Squircles task
    - [ ] correlate single-subject beta's with GLM weights
4. [ ] compare confidence-betas with R output, and with and without fixing sigma at 0

### Next ideas
- concatenate all trials across participants, then fit with a switching LDS where the drifting criterion jumps between observers
- what if the frequency of the slow drift wave changes over time? eg faster oscillations over time, problem because we only fit one AR coef?
- how does the fitting behave if sigma is not estimated analytically?

### To do
- write summary of this week's work
    - describe model
    - describe code structure
    - discuss simulation findings
    - overview open questions
- design matrix predictors
- plotting psychometric functions


### Summary

$x_{t} = Ax_{t-1} 