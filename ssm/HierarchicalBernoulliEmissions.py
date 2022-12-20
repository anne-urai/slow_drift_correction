

import autograd.numpy as np

from scipy.stats import norm
from autograd import grad
from ssm.emissions import Emissions, BernoulliEmission
from ssm.optimizers import convex_combination



# Now the generic code from ssm.hierarchical is adjusted to the code from Matt:
# https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/ssm/hierarchical.py
# A second option would be exactly copy Matt's code and change the distribution to Bernoulli
# This could work because if you look at his code, the actual m-steps (which I am most afraid of requires custom code)
# are actually called from the base class (AutoRegressiveObservations)
# Maybe we need to copy the stochastic m_step from ssm.observations


class HierarchicalBernoulliEmissions(BernoulliEmission): # or Emissions?
    
    
    def __init__(self, N, K, D, M=0, *args,
                 cond_variance_Cs=0.001,
                 cond_variance_Fs=0.001,
                 cond_variance_ds=0.001,
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



    def log_likelihoods(self, data, input, mask, tag):
        return self.per_group_bernoulli_models[self.tags_to_indices[tag]].\
            log_likelihoods(data, input, mask, tag)
    
    def compute_sample_size(self, datas, inputs, masks, tags):
        sample_sizes = np.zeros(self.G)
        for g, (bern_model, tag) in enumerate(zip(self.per_group_bernoulli_models, self.tags)):
            if any([t == tag for t in tags]):
                # Pull out the subset of data that corresponds to this tag
                tdatas = [d for d, t in zip(datas, tags)        if t == tag]
                tinpts = [i for i, t in zip(inputs, tags)       if t == tag]
                tmasks = [m for m, t in zip(masks, tags)        if t == tag]
                ttags  = [t for t    in tags                    if t == tag]
            # TODO: check if compute_sample_size has to be changed in emissions
                # Compute total sample size for this tag
                sample_sizes[g] = bern_model.compute_sample_size(tdatas, tinpts, tmasks, ttags)
    
        return sample_sizes
    
    def expected_sufficient_stats(self, expectations, datas, inputs, masks, tags):
        # assumes that each input is a list of length 1
        stats = []
        for bern_model, tag in zip(self.per_group_bernoulli_models, self.tags):
    
            if any([t == tag for t in tags]):
                # Pull out the subset of data that corresponds to this tag
                texpts = [e for e, t in zip(expectations, tags) if t == tag]
                tdatas = [d for d, t in zip(datas, tags)        if t == tag]
                tinpts = [i for i, t in zip(inputs, tags)       if t == tag]
                tmasks = [m for m, t in zip(masks, tags)        if t == tag]
                ttags  = [t for t    in tags                    if t == tag]
            
                # Compute expected sufficient stats for this subset of data
                these_stats = bern_model.expected_sufficient_stats(texpts, 
                                                                  tdatas, 
                                                                  tinpts, 
                                                                  tmasks, 
                                                                  ttags)

                stats.append(these_stats)
            else:
                stats.append(None)
    
        return stats

    def _m_step_global(self):
        # Note: we could explore smarter averaging techniques for estimating
        #       the global parameters.  E.g. using uncertainty estimages for
        #       the per-group parameters in a hierarchical Bayesian fashion.
        self.global_bernoulli_model.Cs = np.mean([bern.Cs for bern in self.per_group_bernoulli_models], axis=0)
        self.global_bernoulli_model.Fs = np.mean([bern.Fs for bern in self.per_group_bernoulli_models], axis=0)
        self.global_bernoulli_model.ds = np.mean([bern.ds for bern in self.per_group_bernoulli_models], axis=0)
    

    def _update_hierarchical_prior(self):
        # Update the per-group AR objects to have the global AR model
        # parameters as their prior.
        for bern in self.per_group_bernoulli_models:
            bern.set_prior(self.global_bernoulli_model.Cs, self.cond_variance_Cs,
                         self.global_bernoulli_model.Fs, self.cond_variance_Fs,
                         self.global_bernoulli_model.ds, self.cond_variance_ds)
    
    def m_step(self, expectations, datas, inputs, masks, tags,
               sufficient_stats=None, 
               **kwargs):
    
        # Collect sufficient statistics for each group
        if sufficient_stats is None:
            sufficient_stats = \
                self.expected_sufficient_stats(expectations,
                                               datas,
                                               inputs,
                                               masks,
                                               tags)
        else:
            assert isinstance(sufficient_stats, list) and \
                   len(sufficient_stats) == self.G
    
        # Update the per-group weights
        for bern_model, stats in zip(self.per_group_bernoulli_models, sufficient_stats):
            
            # Note: this is going to perform M-steps even for groups that 
            #       are not present in this minibatch.  Hopefully this isn't
            #       too much extra overhead.
            if stats is not None:
                # TODO: shouldn't we change this to m_step_stochastic?
                bern_model.m_step(None, None, None, None, None,
                                sufficient_stats=stats)

        # Update the shared weights
        self._m_step_global()
        self._update_hierarchical_prior()
    
    def stochastic_m_step(self, 
                          optimizer_state,
                          total_sample_size,
                          expectations,
                          datas,
                          inputs,
                          masks,
                          tags,
                          step_size=0.5):
        """
        """
        # Get the expected sufficient statistics for this minibatch
        # Note: this is an array of length num_groups (self.G)
        #       and entries in the array are None if there is no
        #       data with the corresponding tag in this minibatch.
        stats = self.expected_sufficient_stats(expectations,
                                               datas,
                                               inputs,
                                               masks,
                                               tags)
    
        # Scale the statistics by the total sample size on a per-group basis
        this_sample_size = self.compute_sample_size(datas, inputs, masks, tags)
        for g in range(self.G):
            if stats[g] is not None:
                stats[g] = tuple(map(lambda x: x * total_sample_size[g] / this_sample_size[g], stats[g]))
    
        # Combine them with the running average sufficient stats
        if optimizer_state is not None:
            # we've seen some data before, but not necessarily from all groups
            for g in range(self.G):
                if optimizer_state[g] is not None and stats[g] is not None:
                    # we've seen this group before and we have data for it. 
                    # update its stats.
                    stats[g] = convex_combination(optimizer_state[g], stats[g], step_size)
                elif optimizer_state[g] is not None and stats[g] is None:
                    # we've seen this group before but not in this minibatch.
                    # pass existing stats through.
                    stats[g] = optimizer_state[g]
        else:
            # first time we're seeing any data.  return this minibatch's stats.
            pass
    
        # Call the regular m-step with these sufficient statistics
        self.m_step(None, None, None, None, None, sufficient_stats=stats)
    
        # Return the update state (i.e. the new stats)
        return stats
    
    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.per_group_ar_models[self.tags_to_indices[tag]]. \
            sample_x(z, xhist, input, tag, with_noise)
    
    def smooth(self, expectations, data, input, tag):
        return self.per_group_ar_models[self.tags_to_indices[tag]]. \
            smooth(expectations, data, input, tag)




#### Old: generic ssm.hierarchical adjusted to Matt's code
    # def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
    #     for tag in tags:
    #         if not tag in self.tags:
    #             raise Exception("Invalid tag: ".format(tag))

    #     # Optimize parent and child parameters at the same time with SGD
    #     optimizer = dict(sgd=sgd, adam=adam)[optimizer]

    #     # expected log joint
    #     def _expected_log_joint(expectations):
    #         elbo = self.log_prior()

    #         for data, input, mask, tag, (expected_states, expected_joints) \
    #             in zip(datas, inputs, masks, tags, expectations):

    #             if hasattr(self.per_group_bernoulli_models[self.tags_to_indices[tag]], 'log_initial_state_distn'):
    #                 log_pi0 = self.per_group_bernoulli_models[self.tags_to_indices[tag]].log_initial_state_distn(data, input, mask, tag)
    #                 elbo += np.sum(expected_states[0] * log_pi0)

    #             if hasattr(self.per_group_bernoulli_models[self.tags_to_indices[tag]], 'log_transition_matrices'):
    #                 log_Ps = self.per_group_bernoulli_models[self.tags_to_indices[tag]].log_transition_matrices(data, input, mask, tag)
    #                 elbo += np.sum(expected_joints * log_Ps)

    #             if hasattr(self.per_group_bernoulli_models[self.tags_to_indices[tag]], 'log_likelihoods'):
    #                 lls = self.per_group_bernoulli_models[self.tags_to_indices[tag]].log_likelihoods(data, input, mask, tag)
    #                 elbo += np.sum(expected_states * lls)

    #         return elbo

    #     # define optimization target
    #     T = sum([data.shape[0] for data in datas])
    #     def _objective(params, itr):
    #         self.params = params
    #         obj = _expected_log_joint(expectations)
    #         return -obj / T

    #     self.params = \
    #         optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)
            
            
            
    # def log_likelihoods(self, data, input, mask, tag):
    #     return self.per_group_bernoulli_models[self.tags_to_indices[tag]].log_likelihoods(data, input, mask, tag)
                
    # def sample_y(self, z, x, input=None, tag=None):
    #     return self.per_group_bernoulli_models[self.tags_to_indices[tag]].sample_y(z, x, input=input, tag=tag)

    # def initialize_variational_params(self, data, input, mask, tag):
    #     return self.per_group_bernoulli_models[self.tags_to_indices[tag]].initialize_variational_params(data, input, mask, tag)

    # def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
    #     return self.per_group_bernoulli_models[self.tags_to_indices[tag]].smooth(expected_states, variational_mean, data, input, mask, tag)

