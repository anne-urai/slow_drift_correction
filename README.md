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
How does this depend on the size of the noise sigma?
- is there a numerical determinant of ELBO convergence?
- how many `num_iter` do we need?
- how do `predEmissions` compare to binary choices?
- nice to have: printing function (with explanation)?

### Questions for Diksha
- why is `A = 1`?
- how is the analytical computation of sigma done?
- why is `estDrift` computed by multiplying `Cs`?

### Plan van aanpak 
1. answer: with our trial counts, can we even expect to get good fits?
     - try simulations with data-ish parameters from Beehives or Squircles task
2. fit to real data
    - correlate single-subject beta's with GLM weights
3. add confidence-scaling (simulate and fit)
     
### Crazy ideas
- concatenate all trials across participants, then fit with a switching LDS where the drifting criterion jumps between observers

