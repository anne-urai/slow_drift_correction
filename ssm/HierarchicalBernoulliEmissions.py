

import autograd.numpy as np

from scipy.stats import norm
from autograd import grad
from ssm.emissions import Emissions, BernoulliEmission
from ssm.optimizers import convex_combination
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs, convex_combination



# Adjusted code from Matt and Scott:
# https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/ssm/hierarchical.py



class HierarchicalBernoulliEmissions(BernoulliEmission): # or (Emissions)?
    
    
    def __init__(self, N, K, D, M=0, *args,
                 cond_variance_Cs=.000001,
                 cond_variance_Fs=.1,
                 cond_variance_ds=.1,
                 tags=(None,), **kwargs):

        super(HierarchicalBernoulliEmissions, self).__init__(N,K,D,M,*args,**kwargs)

        # First figure out how many tags/groups
        self.tags = tags
        self.tags_to_indices = dict([(tag, i) for i, tag in enumerate(tags)])
        self.G = len(tags)
        assert self.G > 0

        # Set the hierarchical prior hyperparameters
        self.cond_variance_Cs = cond_variance_Cs
        self.cond_variance_Fs = cond_variance_Fs
        self.cond_variance_ds = cond_variance_ds

        # Create a group-level AR model
        self.global_bernoulli_model = \
            BernoulliEmission(N, 1, D, M=M, initialize = "global")

        # Create AR objects for each tag
        self.per_group_bernoulli_models = [
            BernoulliEmission(N, 1, D, M=M,
                                       mean_Cs=self.global_bernoulli_model.Cs,
                                       variance_Cs=cond_variance_Cs,
                                       mean_Fs=self.global_bernoulli_model.Fs,
                                       variance_Fs=cond_variance_Fs,
                                       mean_ds=self.global_bernoulli_model.ds,
                                       variance_ds=cond_variance_ds,
                                       initialize = "group")
            for _ in range(self.G)
        ]
        
        
    def get_Cs(self, tag):
        return self.per_group_bernoulli_models[self.tags_to_indices[tag]].Cs

    def get_Fs(self, tag):
        return self.per_group_bernoulli_models[self.tags_to_indices[tag]].Fs

    def get_ds(self, tag):
        return self.per_group_bernoulli_models[self.tags_to_indices[tag]].ds

    @property
    def params(self):
        raise Exception("Don't try to get these parameters")

    @params.setter
    def params(self, value):
        raise Exception("Don't try to set these parameters")

    def permute(self, perm):
        self.global_bernoulli_model.permute(perm)
        for bern in self.per_group_bernoulli_models:
            bern.permute(perm)

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        
        # Initialize with linear regressions
        self.global_bernoulli_model.initialize(datas, inputs, masks, tags)
        self._update_hierarchical_prior()
        

        # Copy global parameters to per-group models
        for bern in self.per_group_bernoulli_models:
            bern.Cs = self.global_bernoulli_model.Cs.copy()
            bern.Fs = self.global_bernoulli_model.Fs.copy()
            bern.ds = self.global_bernoulli_model.ds.copy()


            bern.Cs = norm.rvs(self.global_bernoulli_model.Cs, np.sqrt(self.cond_variance_Cs))
            bern.Fs = norm.rvs(self.global_bernoulli_model.Fs, np.sqrt(self.cond_variance_Fs))
            bern.ds = norm.rvs(self.global_bernoulli_model.ds, np.sqrt(self.cond_variance_ds))
            
            
    def log_prior(self):
        lp = 0
        for bern in self.per_group_bernoulli_models:
            lp += bern.log_prior()
        return lp

    # original code below but then we get following error:
    # File "c:\users\u0141056\ssm\ssm\lds.py", line 812, in estimate_expected_log_joint
    # log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

    # TypeError: log_likelihoods() takes 5 positional arguments but 6 were given
    
    # def log_likelihoods(self, data, input, mask, tag):
    #     return self.per_group_bernoulli_models[self.tags_to_indices[tag]].\
    #         log_likelihoods(data, input, mask, tag)
    
    # So added x as argument
    def log_likelihoods(self, data, input, mask, tag, x):
        return self.per_group_bernoulli_models[self.tags_to_indices[tag]].\
            log_likelihoods(data, input, mask, tag, x)
    
    
    def _m_step_global(self):
        # Note: we could explore smarter averaging techniques for estimating
        #       the global parameters.  E.g. using uncertainty estimages for
        #       the per-group parameters in a hierarchical Bayesian fashion.

        #self.global_bernoulli_model.Cs = np.mean([bern.Cs for bern in self.per_group_bernoulli_models], axis=0)
        self.global_bernoulli_model.Cs = np.array([[[1]]])

        self.global_bernoulli_model.Fs = np.mean([bern.Fs for bern in self.per_group_bernoulli_models], axis=0)
        self.global_bernoulli_model.ds = np.mean([bern.ds for bern in self.per_group_bernoulli_models], axis=0)
    

    def _update_hierarchical_prior(self):
        # Update the per-group AR objects to have the global AR model
        # parameters as their prior.
        for bern in self.per_group_bernoulli_models:
            bern.set_prior(self.global_bernoulli_model.Cs, self.cond_variance_Cs,
                         self.global_bernoulli_model.Fs, self.cond_variance_Fs,
                         self.global_bernoulli_model.ds, self.cond_variance_ds)
    
    
    # Is this correct???
    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, **kwargs):
        """
        If M-step in Laplace-EM cannot be done in closed form for the emissions, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # Update the per-group weights
        for bern_model in self.per_group_bernoulli_models:
                bern_model.m_step(discrete_expectations, continuous_expectations, datas, inputs, masks, tags)
                
        # Update the shared weights
        self._m_step_global()
        self._update_hierarchical_prior()
    

    
    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.per_group_bernoulli_models[self.tags_to_indices[tag]]. \
            sample_x(z, xhist, input, tag, with_noise)
    
    def smooth(self, expectations, data, input, tag):
        return self.per_group_bernoulli_models[self.tags_to_indices[tag]]. \
            smooth(expectations, data, input, tag)



