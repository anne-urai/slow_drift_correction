import copy
import warnings
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad, grad

from ssm.optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, \
    convex_combination, newtons_method_block_tridiag_hessian, geometric_learning_rate
from ssm.primitives import hmm_normalizer
from ssm.messages import hmm_expected_states, viterbi
from ssm.util import ensure_args_are_lists, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, ssm_pbar

import ssm.observations as obs
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.emissions as emssn
import ssm.hmm as hmm
import ssm.variational as varinf

import ssm.HierarchicalAutoRegressiveObservations as hierAR
import ssm.HierarchicalBernoulliEmissions as hierBern


__all__ = ['SLDS', 'LDS']


class SLDS(object):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, K, D, *, M=0,
                 init_state_distn=None,
                 transitions="standard",
                 transition_kwargs=None,
                 dynamics="gaussian",
                 dynamics_kwargs=None,
                 emissions="gaussian_orthog",
                 emission_kwargs=None,
                 single_subspace=True,
                 **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        assert isinstance(init_state_distn, isd.InitialStateDistribution)

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            sticky=trans.StickyTransitions,
            inputdriven=trans.InputDrivenTransitions,
            recurrent=trans.RecurrentTransitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            rbf_recurrent=trans.RBFRecurrentTransitions,
            nn_recurrent=trans.NeuralNetworkRecurrentTransitions
            )

        if isinstance(transitions, str):
            transitions = transitions.lower()
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # Make the dynamics distn
        dynamics_classes = dict(
            none=obs.GaussianObservations,
            gaussian=obs.AutoRegressiveObservations,
            diagonal_gaussian=obs.AutoRegressiveDiagonalNoiseObservations,
            t=obs.RobustAutoRegressiveObservations,
            studentst=obs.RobustAutoRegressiveObservations,
            diagonal_t=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_studentst=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            hierarchical_ar = hierAR.HierarchicalAutoRegressiveObservations)

        if isinstance(dynamics, str):
            dynamics = dynamics.lower()
            if dynamics not in dynamics_classes:
                raise Exception("Invalid dynamics model: {}. Must be one of {}".
                    format(dynamics, list(dynamics_classes.keys())))

            dynamics_kwargs = dynamics_kwargs or {}
            dynamics = dynamics_classes[dynamics](K, D, M=M, **dynamics_kwargs)
        if not isinstance(dynamics, obs.Observations):
            raise TypeError("'dynamics' must be a subclass of"
                            " ssm.observations.Observations")

        # Make the emission distn
        emission_classes = dict(
            gaussian=emssn.GaussianEmissions,
            gaussian_orthog=emssn.GaussianOrthogonalEmissions,
            gaussian_id=emssn.GaussianIdentityEmissions,
            gaussian_nn=emssn.GaussianNeuralNetworkEmissions,
            studentst=emssn.StudentsTEmissions,
            studentst_orthog=emssn.StudentsTOrthogonalEmissions,
            studentst_id=emssn.StudentsTIdentityEmissions,
            studentst_nn=emssn.StudentsTNeuralNetworkEmissions,
            t=emssn.StudentsTEmissions,
            t_orthog=emssn.StudentsTOrthogonalEmissions,
            t_id=emssn.StudentsTIdentityEmissions,
            t_nn=emssn.StudentsTNeuralNetworkEmissions,
            poisson=emssn.PoissonEmissions,
            poisson_orthog=emssn.PoissonOrthogonalEmissions,
            poisson_id=emssn.PoissonIdentityEmissions,
            poisson_nn=emssn.PoissonNeuralNetworkEmissions,
            bernoulli=emssn.BernoulliEmissions,
            bernoulli_orthog=emssn.BernoulliOrthogonalEmissions,
            bernoulli_id=emssn.BernoulliIdentityEmissions,
            bernoulli_nn=emssn.BernoulliNeuralNetworkEmissions,
            ar=emssn.AutoRegressiveEmissions,
            ar_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            ar_id=emssn.AutoRegressiveIdentityEmissions,
            ar_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
            autoregressive=emssn.AutoRegressiveEmissions,
            autoregressive_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            autoregressive_id=emssn.AutoRegressiveIdentityEmissions,
            autoregressive_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
            hierarchical_bernoulli = hierBern.HierarchicalBernoulliEmissions
            )

        if isinstance(emissions, str):
            emissions = emissions.lower()
            if emissions not in emission_classes:
                raise Exception("Invalid emission model: {}. Must be one of {}".
                    format(emissions, list(emission_classes.keys())))

            emission_kwargs = emission_kwargs or {}
            emissions = emission_classes[emissions](N, K, D, M=M,
                single_subspace=single_subspace, **emission_kwargs)
        if not isinstance(emissions, emssn.Emissions):
            raise TypeError("'emissions' must be a subclass of"
                            " ssm.emissions.Emissions")

        self.N, self.K, self.D, self.M = N, K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.dynamics.params, \
               self.emissions.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.dynamics.params = value[2]
        self.emissions.params = value[3]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, verbose = 2, num_iters=25):
        
        self.emissions.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.init_state_distn.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.transitions.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.dynamics.initialize(datas, inputs=inputs, masks=masks, tags=tags)

        # First initialize the observation model
        # Get the initialized variational mean for the data
        # xs = [self.emissions.invert(data, input, mask, tag)
        #       for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        # xmasks = [np.ones_like(x, dtype=bool) for x in xs]

        # # Now run a few iterations of EM on a ARHMM with the variational mean
        # if verbose > 0:
        #   print("Initializing with an ARHMM using {} steps of EM.".format(num_iters))
        # arhmm = hmm.HMM(self.K, self.D, M=self.M,
        #                 init_state_distn=copy.deepcopy(self.init_state_distn),
        #                 transitions=copy.deepcopy(self.transitions),
        #                 observations=copy.deepcopy(self.dynamics))
        # arhmm.fit(xs, inputs=inputs, masks=xmasks, tags=tags, verbose = verbose,
        #           method="em", num_iters=num_iters)

        # self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
        # self.transitions = copy.deepcopy(arhmm.transitions)
        # self.dynamics = copy.deepcopy(arhmm.observations)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.dynamics.permute(perm)
        self.emissions.permute(perm)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.dynamics.log_prior() + \
               self.emissions.log_prior()

    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        N = self.N
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)

        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = np.zeros(T+1, dtype=int)
            x = np.zeros((T+1,) + D)
            # input = np.zeros((T+1,) + M) if input is None else input
            input = np.zeros((T+1,) + M) if input is None else np.concatenate((np.zeros((1,) + M), input))
            xmask = np.ones((T+1,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            x[0] = self.dynamics.sample_x(z[0], x[:0], tag=tag, with_noise=with_noise)

        else:
            zhist, xhist, yhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            assert xhist.shape == (pad, D)
            assert yhist.shape == (pad, N)

            z = np.concatenate((zhist, np.zeros(T, dtype=int)))
            x = np.concatenate((xhist, np.zeros((T,) + D)))
            # input = np.zeros((T+pad,) + M) if input is None else input
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            xmask = np.ones((T+pad,) + D, dtype=bool)

        # Sample z and x
        for t in range(pad, T+pad):
            Pt = np.exp(self.transitions.log_transition_matrices(x[t-1:t+1], input[t-1:t+1], mask=xmask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_x(z[t], x[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Sample observations given latent states
        # TODO: sample in the loop above?
        y = self.emissions.sample(z, x, input=input, tag=tag)
        return z[pad:], x[pad:], y[pad:]

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        x_mask = np.ones_like(variational_mean, dtype=bool)
        pi0 = self.init_state_distn.log_initial_state_distn
        Ps = self.transitions.transition_matrices(variational_mean, input, x_mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, x_mask, tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return hmm_expected_states(pi0, Ps, log_likes)

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(variational_mean, input, mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, np.ones_like(variational_mean, dtype=bool), tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return viterbi(pi0, Ps, log_likes)

    @ensure_slds_args_not_none
    def smooth(self, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(variational_mean, data, input, mask, tag)
        return self.emissions.smooth(Ez, variational_mean, data, input, tag)


    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Cannot compute exact marginal log probability for the SLDS.")
        return np.nan
    
    
    # Copied from https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/ssm/hmm.py#L302
    # Not sure if this is correct!
    # @ensure_args_are_lists
    # def log_probability(self, datas, inputs=None, masks=None, tags=None):
    #     return self.log_likelihood(datas, inputs, masks, tags) + self.log_prior()

    

    # # added log_likes += self.emissions.log_likelihoods(data, input, mask, tag) in m-step
    # def _fit_stochastic_em(self, optimizer, datas, inputs, masks, tags, verbose = 2, num_epochs=100, **kwargs):
    #         """
    #         Replace the M-step of EM with a stochastic gradient update using the ELBO computed
    #         on a minibatch of data.
    #         """
    #         M = len(datas)
    #         T = sum([data.shape[0] for data in datas])
    
    #         # A helper to grab a minibatch of data
    #         perm = [np.random.permutation(M) for _ in range(num_epochs)]
    #         def _get_minibatch(itr):
    #             epoch = itr // M
    #             m = itr % M
    #             i = perm[epoch][m]
    #             return datas[i], inputs[i], masks[i], tags[i]
    
    #         # Define the objective (negative ELBO)
    #         def _objective(params, itr):
    #             # Grab a minibatch of data
    #             data, input, mask, tag = _get_minibatch(itr)
    #             Ti = data.shape[0]
    
    #             # E step: compute expected latent states with current parameters
    #             Ez, Ezzp1, _ = self.expected_states(data, input, mask, tag)
    
    #             # M step: set the parameter and compute the (normalized) objective function
    #             self.params = params
    #             pi0 = self.init_state_distn.initial_state_distn
    #             log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
    #             log_likes = self.dynamics.log_likelihoods(data, input, mask, tag)
    #             log_likes += self.emissions.log_likelihoods(data, input, mask, tag)
    
    #             # Compute the expected log probability
    #             # (Scale by number of length of this minibatch.)
    #             obj = self.log_prior()
    #             obj += np.sum(Ez[0] * np.log(pi0)) * M
    #             obj += np.sum(Ezzp1 * log_Ps) * (T - M) / (Ti - 1)
    #             obj += np.sum(Ez * log_likes) * T / Ti
    #             assert np.isfinite(obj)
    
    #             return -obj / T
    
    #         # Set up the progress bar
    #         lls  = [-_objective(self.params, 0) * T]
    #         pbar = ssm_pbar(num_epochs * M, verbose, "Epoch {} Itr {} LP: {:.1f}", [0, 0, lls[-1]])
    
    #         # Run the optimizer
    #         step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
    #         state = None
    #         for itr in pbar:
    #             self.params, val, g, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
    #             epoch = itr // M
    #             m = itr % M
    #             lls.append(-val * T)
    #             if verbose == 2:
    #               pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(epoch, m, lls[-1]))
    #               pbar.update(1)
    
    #         return lls

    # # copied from https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/ssm/hmm.py#L552
    # # line 338: added self.emissions, and changed self.observations to self.emissions
    # # in def fit (see below) added stochastic_em_conj to fitting dict
    # def _fit_stochastic_em_conjugate(self, datas, inputs, masks, tags,
    #                                      verbose=2,
    #                                      num_epochs=100,
    #                                      tolerance=0,
    #                                      learning_rate=geometric_learning_rate,
    #                                      **kwargs):
    #         """
    #         Fit the parameters with stochastic EM, assuming that the observations
    #         and transitions are exponential family distributions with closed-form
    #         sufficient statistics.
    #         Initialize:
    #             - Initialize the running average sufficient statistics.  E.g. set
    #               them to the stats from the prior if you have one, or to zero o.w.
    #         E-step:
    #             - grab a minibatch of data (e.g. one of your data arrays)
    #             - compute E[z_t] and E[z_t, z_{t+1}] for that minibatch
    #         M-step:
    #             - compute expected sufficient statistics of the transition and
    #               observation models from this minibatch. Note: these are summed
    #               over datapoints in the minibatch.
    #             - take a convex combination of ESS from this minibatch and your
    #               running average.
    #             - then set parameters to maximize the likelihood using these
    #               averaged sufficent statistics.
    #         References:
    #             - Cappé, Olivier, and Eric Moulines. "On‐line expectation–maximization
    #               algorithm for latent data models." Journal of the Royal Statistical
    #               Society: Series B (Statistical Methodology) 71.3 (2009): 593-613.
    #             - Hoffman, Matthew D., et al. "Stochastic variational inference."
    #               The Journal of Machine Learning Research 14.1 (2013): 1303-1347.
    #         TODO:
    #             - Specify learning rates
    #             - Control how log likelihoods are reported
    #         """
    #         # Store the log likelihoods (per minibatch)
    #         epoch_lps = []
    #         inner_lps = []
    
    #         # Initialize the progress bars
    #         num_datas = len(datas)
    #         T_total = sum([data.shape[0] for data in datas])
    #         epoch_pbar = ssm_pbar(num_epochs, verbose, "Initializing...", [])
    #         inner_pbar = ssm_pbar(num_datas, verbose, "Initializing...", [])
    
    #         # Initialize the sufficient statistics by calling without data
    #         distributions = [self.init_state_distn, self.transitions, self.dynamics, self.emissions]
    #         m_step_states = [None, None, None, None]
    #         total_sample_sizes = [distn.compute_sample_size(datas, inputs, masks, tags) 
    #                               for distn in distributions]
                
    #         # Iterate over minibatches
    #         for epoch in epoch_pbar:
    #             perm = npr.permutation(num_datas)
    #             inner_pbar.reset()
    
    #             for i in inner_pbar:
    #                 # Grab a minibatch of data
    #                 j = perm[i]
    #                 data, input, mask, tag = datas[j], inputs[j], masks[j], tags[j]
    
    #                 # E step: compute expected latent states with current parameters
    #                 #         _for this particular data array_.
    #                 expectations = self.expected_states(data, input, mask, tag)
    
    #                 # Compute noisy estimate of the total log prob using this minibatch
    #                 inner_lps.append(self.log_prior() + expectations[2] * T_total / data.shape[0])
    #                 inner_pbar.set_description("LP: {:.1f}".format(inner_lps[-1]))
    
    #                 # M step: Get expected sufficient statistics for this data
    #                 #         and combine them with the running average.
    
    #                 # convex combo is computed as
    #                 # (1 - step_size) * run_avg_stats + step_size * stats_from_this_minibatch
    #                 step_size = learning_rate(epoch * num_datas + i)
    
    #                 # M step: update the parameters with those stats.
    #                 for i, distn in enumerate(distributions):
    #                     m_step_states[i] = distn.stochastic_m_step(m_step_states[i], 
    #                                                                total_sample_sizes[i],
    #                                                                [expectations], 
    #                                                                [data], 
    #                                                                [input], 
    #                                                                [mask],
    #                                                                [tag],
    #                                                                step_size=step_size)
                    
    #             # Compute the log probability of the full dataset
    #             if verbose == 2:
    #                 epoch_lps.append(self.log_probability(datas, inputs, masks, tags))
    #                 epoch_pbar.set_description("LP: {:.1f}".format(epoch_lps[-1]))
    
    #         return epoch_lps, inner_lps




    @ensure_variational_args_are_lists
    def _bbvi_elbo(self, variational_posterior, datas, inputs=None, masks=None, tags=None,  n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta)
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for sample in range(n_samples):
            # Sample x from the variational posterior
            xs = variational_posterior.sample()

            # log p(theta)
            elbo += self.log_prior()

            # log p(x, y | theta) = log \sum_z p(x, y, z | theta)
            for x, data, input, mask, tag in zip(xs, datas, inputs, masks, tags):

                # The "mask" for x is all ones
                x_mask = np.ones_like(x, dtype=bool)

                pi0 = self.init_state_distn.initial_state_distn
                Ps = self.transitions.transition_matrices(x, input, x_mask, tag)
                log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
                log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)
                elbo += hmm_normalizer(pi0, Ps, log_likes)

            # -log q(x)
            elbo -= variational_posterior.log_density(xs)
            assert np.isfinite(elbo)

        return elbo / n_samples

    def _fit_bbvi(self, variational_posterior, datas, inputs, masks, tags, verbose = 2,
                  learning=True, optimizer="adam", num_iters=100, **kwargs):
        """
        Fit with black box variational inference using a
        Gaussian approximation for the latent states x_{1:T}.
        """
        # Define the objective (negative ELBO)
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            if learning:
                self.params, variational_posterior.params = params
            else:
                variational_posterior.params = params

            obj = self._bbvi_elbo(variational_posterior, datas, inputs, masks, tags)
            return -obj / T

        # Initialize the parameters
        if learning:
            params = (self.params, variational_posterior.params)
        else:
            params = variational_posterior.params

        # Set up the progress bar
        elbos = [-_objective(params, 0) * T]
        pbar  = ssm_pbar(num_iters, verbose, "LP: {:.1f}", [elbos[0]])

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            params, val, g, state = step(value_and_grad(_objective), params, itr, state)
            elbos.append(-val * T)

            # TODO: Check for convergence -- early stopping

            # Update progress bar
            if verbose == 2:
              pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
              pbar.update()

        # Save the final parameters
        if learning:
            self.params, variational_posterior.params = params
        else:
            variational_posterior.params = params

        return elbos

    def _fit_laplace_em_discrete_state_update(
        self, variational_posterior, datas,
        inputs, masks, tags,
        num_samples):

        # 0. Draw samples of q(x) for Monte Carlo approximating expectations
        x_sampless = [variational_posterior.sample_continuous_states() for _ in range(num_samples)]
        # Convert this to a list of length len(datas) where each entry
        # is a tuple of length num_samples
        x_sampless = list(zip(*x_sampless))

        # 1. Update the variational posterior on q(z) for fixed q(x)
        #    - Monte Carlo approximate the log transition matrices
        #    - Compute the expected log likelihoods (i.e. log dynamics probs)
        #    - If emissions depend on z, compute expected emission likelihoods
        discrete_state_params = []
        for x_samples, data, input, mask, tag in \
            zip(x_sampless, datas, inputs, masks, tags):

            # Make a mask for the continuous states
            x_mask = np.ones_like(x_samples[0], dtype=bool)

            # Compute expected log initial distribution, transition matrices, and likelihoods
            pi0 = np.mean(
                [self.init_state_distn.initial_state_distn
                 for x in x_samples], axis=0)

            Ps = np.mean(
                [self.transitions.transition_matrices(x, input, x_mask, tag)
                 for x in x_samples], axis=0)

            log_likes = np.mean(
                [self.dynamics.log_likelihoods(x, input, x_mask, tag)
                 for x in x_samples], axis=0)

            if not self.emissions.single_subspace:
                log_likes += np.mean(
                    [self.emissions.log_likelihoods(data, input, mask, tag, x)
                     for x in x_samples], axis=0)

            discrete_state_params.append(dict(pi0=pi0,
                                              Ps=Ps,
                                              log_likes=log_likes))

        # Update the variational parameters
        variational_posterior.discrete_state_params = discrete_state_params

    # Compute the expected log joint
    def _laplace_neg_expected_log_joint(self,
                                        data,
                                        input,
                                        mask,
                                        tag,
                                        x,
                                        Ez,
                                        Ezzp1,
                                        scale=1):
        # The "mask" for x is all ones
        x_mask = np.ones_like(x, dtype=bool)
        log_pi0 = self.init_state_distn.log_initial_state_distn
        log_Ps = self.transitions.\
            log_transition_matrices(x, input, x_mask, tag)
        log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
        log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

        # Compute the expected log probability
        elp = np.sum(Ez[0] * log_pi0)
        elp += np.sum(Ezzp1 * log_Ps)
        elp += np.sum(Ez * log_likes)
        # assert np.all(np.isfinite(elp))
        return -1 * elp / scale

    # We also need the hessian of the of the expected log joint
    def _laplace_neg_hessian_params(self, data, input, mask, tag, x, Ez, Ezzp1):
        T, D = np.shape(x)
        x_mask = np.ones((T, D), dtype=bool)

        J_ini, J_dyn_11, J_dyn_21, J_dyn_22 = self.dynamics.\
            neg_hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)
        J_transitions = self.transitions.\
            neg_hessian_expected_log_trans_prob(x, input, x_mask, tag, Ezzp1)
        J_dyn_11 += J_transitions

        J_obs = self.emissions.\
            neg_hessian_log_emissions_prob(data, input, mask, tag, x, Ez)

        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs

    def _laplace_hessian_neg_expected_log_joint(self, data, input, mask, tag, x, Ez, Ezzp1, scale=1):
        J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs = \
            self._laplace_neg_hessian_params(data, input, mask, tag, x, Ez, Ezzp1)

        hessian_diag = np.zeros_like(J_obs)
        hessian_diag[:] += J_obs
        hessian_diag[0] += J_ini
        hessian_diag[:-1] += J_dyn_11
        hessian_diag[1:] += J_dyn_22
        hessian_lower_diag = J_dyn_21

        # Return the scaled negative hessian, which is positive definite
        return hessian_diag / scale, hessian_lower_diag / scale

    def _laplace_neg_hessian_params_to_hs(self,
                                          x,
                                          J_ini,
                                          J_dyn_11,
                                          J_dyn_21,
                                          J_dyn_22,
                                          J_obs):
        h_ini = J_ini @ x[0]

        h_dyn_1 = (J_dyn_11 @ x[:-1][:, :, None])[:, :, 0]
        h_dyn_1 += (np.swapaxes(J_dyn_21, -1, -2) @ x[1:][:, :, None])[:, :, 0]

        h_dyn_2 = (J_dyn_22 @ x[1:][:, :, None])[:, :, 0]
        h_dyn_2 += (J_dyn_21 @ x[:-1][:, :, None])[:, :, 0]

        h_obs = (J_obs @ x[:, :, None])[:, :, 0]
        return h_ini, h_dyn_1, h_dyn_2, h_obs

    def _fit_laplace_em_continuous_state_update(self,
                                                variational_posterior,
                                                datas,
                                                inputs,
                                                masks,
                                                tags,
                                                continuous_optimizer,
                                                continuous_tolerance,
                                                continuous_maxiter):

        # 2. Update the variational posterior q(x) for fixed q(z)
        #    - Use Newton's method or LBFGS to find the argmax of the expected
        #      log joint
        #       - Compute the gradient g(x) and block tridiagonal Hessian J(x)
        #       - Newton update: x' = x + J(x)^{-1} g(x)
        #       - Check for convergence of x'
        #    - Evaluate the J(x*) at the optimal x*

        # Optimize the expected log joint for each data array to find the mode
        # and the curvature around the mode.  This gives a  Laplace approximation
        # for q(x).
        continuous_state_params = []
        x0s = variational_posterior.mean_continuous_states
        for (Ez, Ezzp1, _), x0, data, input, mask, tag in \
            zip(variational_posterior.discrete_expectations,
                x0s, datas, inputs, masks, tags):

            # Use Newton's method or LBFGS to find the argmax of the expected log joint
            scale = x0.size
            kwargs = dict(data=data, input=input, mask=mask, tag=tag, Ez=Ez, Ezzp1=Ezzp1, scale=scale)

            def _objective(x, iter): return self._laplace_neg_expected_log_joint(x=x, **kwargs)
            def _grad_obj(x): return grad(self._laplace_neg_expected_log_joint, argnum=4)(data, input, mask, tag, x, Ez, Ezzp1, scale)
            def _hess_obj(x): return self._laplace_hessian_neg_expected_log_joint(x=x, **kwargs)

            if continuous_optimizer == "newton":
                x = newtons_method_block_tridiag_hessian(
                    x0, lambda x: _objective(x, None), _grad_obj, _hess_obj,
                    tolerance=continuous_tolerance, maxiter=continuous_maxiter)

            elif continuous_optimizer  == "lbfgs":
                x = lbfgs(_objective, x0, num_iters=continuous_maxiter,
                          tol=continuous_tolerance)

            else:
                raise Exception("Invalid continuous_optimizer: {}".format(continuous_optimizer ))

            # Evaluate the Hessian at the mode
            assert np.all(np.isfinite(_objective(x, -1)))

            J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs = self.\
                _laplace_neg_hessian_params(data, input, mask, tag, x, Ez, Ezzp1)
            h_ini, h_dyn_1, h_dyn_2, h_obs = \
                self._laplace_neg_hessian_params_to_hs(x, J_ini, J_dyn_11,
                                              J_dyn_21, J_dyn_22, J_obs)

            continuous_state_params.append(dict(J_ini=J_ini,
                                                J_dyn_11=J_dyn_11,
                                                J_dyn_21=J_dyn_21,
                                                J_dyn_22=J_dyn_22,
                                                J_obs=J_obs,
                                                h_ini=h_ini,
                                                h_dyn_1=h_dyn_1,
                                                h_dyn_2=h_dyn_2,
                                                h_obs=h_obs))

        # Update the variational posterior params
        variational_posterior.continuous_state_params = continuous_state_params

    def _fit_laplace_em_params_update(self,
                                      variational_posterior,
                                      datas,
                                      inputs,
                                      masks,
                                      tags,
                                      emission_optimizer,
                                      emission_optimizer_maxiter,
                                      step_size):

        # Compute necessary expectations either analytically or via samples
        continuous_samples = variational_posterior.sample_continuous_states()
        discrete_expectations = variational_posterior.discrete_expectations

        # Approximate update of initial distribution  and transition params.
        # Replace the expectation wrt x with sample from q(x). The parameter
        # update is partial and depends on alpha.
        xmasks = [np.ones_like(x, dtype=bool) for x in continuous_samples]
        for distn in [self.init_state_distn, self.transitions]:
            curr_prms = copy.deepcopy(distn.params)
            if curr_prms == tuple(): continue
            distn.m_step(discrete_expectations, continuous_samples, inputs, xmasks, tags)
            distn.params = convex_combination(curr_prms, distn.params, step_size)

        kwargs = dict(expectations=discrete_expectations,
                      datas=continuous_samples,
                      inputs=inputs,
                      masks=xmasks,
                      tags=tags
        )
        # TODO: should we add hierarchicalAutoRegressiveObservations? R
        exact_m_step_dynamics = [
           obs.AutoRegressiveObservations,
           obs.AutoRegressiveObservationsNoInput,
           obs.AutoRegressiveDiagonalNoiseObservations, 
           hierAR.HierarchicalAutoRegressiveObservations,
        ]
        if type(self.dynamics) in exact_m_step_dynamics and self.dynamics.lags == 1:
            # In this case, we can do an exact M-step on the dynamics by passing
            # in the true sufficient statistics for the continuous state.
            kwargs["continuous_expectations"] = variational_posterior.continuous_expectations
            self.dynamics.m_step(**kwargs)
        else:
            # Otherwise, do an approximate m-step by sampling.
            curr_prms = copy.deepcopy(self.dynamics.params)
            self.dynamics.m_step(**kwargs)
            self.dynamics.params = convex_combination(curr_prms, self.dynamics.params, step_size)

        # Update emissions params. This is always approximate (at least for now).
        #curr_prms = copy.deepcopy(self.emissions.params)
        self.emissions.m_step(discrete_expectations, continuous_samples,
                              datas, inputs, masks, tags,
                              optimizer=emission_optimizer,
                              maxiter=emission_optimizer_maxiter)
        #self.emissions.params = convex_combination(curr_prms, self.emissions.params, step_size)

    def _laplace_em_elbo(self,
                         variational_posterior,
                         datas,
                         inputs,
                         masks,
                         tags,
                         n_samples=1):

        def estimate_expected_log_joint(n_samples):
            exp_log_joint = 0.0
            for sample in range(n_samples):

                # sample continuous states
                continuous_samples = variational_posterior.sample_continuous_states()
                discrete_expectations = variational_posterior.discrete_expectations

                # log p(theta)
                exp_log_joint += self.log_prior()

                for x, (Ez, Ezzp1, _), data, input, mask, tag in \
                    zip(continuous_samples, discrete_expectations, datas, inputs, masks, tags):

                    # The "mask" for x is all ones
                    x_mask = np.ones_like(x, dtype=bool)
                    log_pi0 = self.init_state_distn.log_initial_state_distn
                    log_Ps = self.transitions.log_transition_matrices(x, input, x_mask, tag)
                    log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
                    log_likes += self.emissions.log_likelihoods(data, input, mask, tag, x)

                    # Compute the expected log probability
                    exp_log_joint += np.sum(Ez[0] * log_pi0)
                    exp_log_joint += np.sum(Ezzp1 * log_Ps)
                    exp_log_joint += np.sum(Ez * log_likes)
            return exp_log_joint / n_samples

        return estimate_expected_log_joint(n_samples) + variational_posterior.entropy()

    def _fit_laplace_em(self, variational_posterior, datas,
                        inputs=None, masks=None, tags=None,
                        verbose = 2,
                        num_iters=100,
                        num_samples=1,
                        continuous_optimizer="newton",
                        continuous_tolerance=1e-4,
                        continuous_maxiter=100,
                        emission_optimizer="lbfgs",
                        emission_optimizer_maxiter=100,
                        step_size=0.5,
                        learning=True):
        """
        Fit an approximate posterior p(z, x | y) \approx q(z) q(x).
        Perform block coordinate ascent on q(z) followed by q(x).
        Assume q(x) is a Gaussian with a block tridiagonal precision matrix,
        and that we update q(x) via Laplace approximation.
        Assume q(z) is a chain-structured discrete graphical model.
        """
        elbos = [self._laplace_em_elbo(variational_posterior, datas, inputs, masks, tags)]
       
        pbar = ssm_pbar(num_iters, verbose, "ELBO: {:.1f}", [elbos[-1]])

        for itr in pbar:
            # 1. Update the discrete state posterior q(z) if K>1
            if self.K > 1:
                self._fit_laplace_em_discrete_state_update(
                    variational_posterior, datas, inputs, masks, tags, num_samples)

            # 2. Update the continuous state posterior q(x)
            self._fit_laplace_em_continuous_state_update(
                variational_posterior, datas, inputs, masks, tags,
                continuous_optimizer, continuous_tolerance, continuous_maxiter)

            # Update parameters
            if learning:
                self._fit_laplace_em_params_update(
                    variational_posterior, datas, inputs, masks, tags,
                    emission_optimizer, emission_optimizer_maxiter, step_size)

            elbos.append(self._laplace_em_elbo(
                variational_posterior, datas, inputs, masks, tags))
            if verbose == 2:
              pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))

        return elbos

    def _make_variational_posterior(self, variational_posterior, datas, inputs, masks, tags, method, **variational_posterior_kwargs):
        # Initialize the variational posterior
        if isinstance(variational_posterior, str):
            # Make a posterior of the specified type
            _var_posteriors = dict(
                meanfield=varinf.SLDSMeanFieldVariationalPosterior,
                mf=varinf.SLDSMeanFieldVariationalPosterior,
                lds=varinf.SLDSTriDiagVariationalPosterior,
                tridiag=varinf.SLDSTriDiagVariationalPosterior,
                structured_meanfield=varinf.SLDSStructuredMeanFieldVariationalPosterior
                )

            if variational_posterior not in _var_posteriors:
                raise Exception("Invalid posterior: {}. Options are {}.".\
                                format(variational_posterior, _var_posteriors.keys()))
            posterior = _var_posteriors[variational_posterior](self, datas, inputs, masks, tags, **variational_posterior_kwargs)

        else:
            # Check validity of given posterior
            posterior = variational_posterior
            assert isinstance(posterior, varinf.VariationalPosterior), \
            "Given posterior must be an instance of ssm.variational.VariationalPosterior"

        # Check that the posterior type works with the fitting method
        if method in ["svi", "bbvi"]:
            assert isinstance(posterior,
                (varinf.SLDSMeanFieldVariationalPosterior, varinf.SLDSTriDiagVariationalPosterior)),\
            "BBVI only supports 'meanfield' or 'lds' posteriors."

        elif method in ["laplace_em"]: # originally only laplace_em
            assert isinstance(posterior, varinf.SLDSStructuredMeanFieldVariationalPosterior),\
            "Laplace EM only supports 'structured' posterior."

        return posterior




    # from https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/ssm/hmm.py#L700
    # @ensure_args_are_lists
    # def fit(self, datas, inputs, masks=None, tags=None, verbose=2,
    #         method="em", initialize=True, num_iters = 10, **kwargs):
      
    #     _fitting_methods = \
    #         dict(stochastic_em_conj=self._fit_stochastic_em_conjugate)

    #     if method not in _fitting_methods:
    #         raise Exception("Invalid method: {}. Options are {}".
    #                         format(method, _fitting_methods.keys()))

    #     # Initialize the model parameters
    #     if initialize:
    #         self.initialize(datas, inputs, masks, tags, verbose = verbose, num_iters=num_iters)

    #     # TODO: Move this!
    #     if isinstance(self.transitions,
    #                   trans.ConstrainedStationaryTransitions):
    #         if method != "em":
    #             raise Exception("Only EM is implemented "
    #                             "for Constrained transitions.")

    #     return _fitting_methods[method](
    #         datas, inputs=inputs, masks=masks, tags=tags, verbose=verbose, **kwargs)
    
    
    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None, verbose = 2,
            method="laplace_em", variational_posterior="structured_meanfield",
            variational_posterior_kwargs=None,
            initialize=True, num_init_iters=25,
            **kwargs):

        """
        There are many possible algorithms one could run.  We have only implemented
        two here:
            - Laplace variational EM, i.e. a structured mean field algorithm where
              we approximate the posterior on continuous states with a Gaussian
              using the mode of the expected log likelihood and the curvature around
              the mode.  This seems to work well for a variety of nonconjugate models,
              and it has the advantage of relaxing to exact EM for the case of
              Gaussian linear dynamical systems.

            - Black box variational inference (BBVI) with mean field or structured
              mean field variational posteriors.  This doesn't seem like a very
              effective fitting algorithm, but it is quite general.
        """
        # Specify fitting methods
        _fitting_methods = dict(laplace_em=self._fit_laplace_em,
                                bbvi=self._fit_bbvi)
                                #stochastic_em_conj=self._fit_stochastic_em_conjugate)

        # Deprecate "svi" as a method
        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        # Initialize the model parameters
        if initialize:
            self.initialize(datas, inputs, masks, tags, verbose = verbose, num_iters=num_init_iters)

        # Initialize the variational posterior
        # Variational posterior allows to approximate the true posterior distribution
        # It is defined by a set of parameters that we can adjust
        # By adjusting these parameters, we can try to make the variational distribution as similar as possible to the true posterior distribution.
        # ~ variational inference: replacing the true posterior distribution with a simpler, approximated distribution

        variational_posterior_kwargs = variational_posterior_kwargs or {}
        posterior = self._make_variational_posterior(variational_posterior, datas, inputs, masks, tags, method, **variational_posterior_kwargs)
        elbos = _fitting_methods[method](posterior, datas, inputs, masks, tags, verbose, learning=True, **kwargs)
        return elbos, posterior

    @ensure_args_are_lists
    def approximate_posterior(self, datas, inputs=None, masks=None, tags=None,
                              method="laplace_em", variational_posterior="structured_meanfield",
                              **kwargs):
        """
        Fit an approximate posterior to data, without updating model params.

        This function computes the posterior over discrete and continuous states
        with model parameters fixed. This can be thought of as extending the Kalman
        Smoother (which computes the state distribution in an LDS) to the SLDS case.
        If the model is an LDS, and the laplace-em method is used with a structured_meanfield
        posterior, this function will be equivalent to running a Kalman Smoother.

        Returns: (elbos, posterior)
                 posterior is a variational posterior object as defined in variational.py
        """
        # Specify fitting methods
        _fitting_methods = dict(bbvi=self._fit_bbvi,
                                laplace_em=self._fit_laplace_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        # Initialize the variational posterior
        posterior = self._make_variational_posterior(variational_posterior, datas, inputs, masks, tags, method)
        elbos = _fitting_methods[method](posterior, datas, inputs, masks, tags, learning=False, **kwargs)
        return elbos, posterior


class LDS(SLDS):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, N, D, *, M=0,
            dynamics="gaussian",
            dynamics_kwargs=None,
            emissions="gaussian_orthog",
            emission_kwargs=None,
            **kwargs):

        # Make the dynamics distn
        dynamics_classes = dict(
            none=obs.GaussianObservations,
            gaussian=obs.AutoRegressiveObservations,
            diagonal_gaussian=obs.AutoRegressiveDiagonalNoiseObservations,
            t=obs.RobustAutoRegressiveObservations,
            studentst=obs.RobustAutoRegressiveObservations,
            diagonal_t=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_studentst=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            hierarchical_ar = hierAR.HierarchicalAutoRegressiveObservations)

        if isinstance(dynamics, str):
            dynamics = dynamics.lower()
            if dynamics not in dynamics_classes:
                raise Exception("Invalid dynamics model: {}. Must be one of {}".
                    format(dynamics, list(dynamics_classes.keys())))

            dynamics_kwargs = dynamics_kwargs or {}
            dynamics = dynamics_classes[dynamics](1, D, M=M, **dynamics_kwargs)
        if not isinstance(dynamics, obs.Observations):
            raise TypeError("'dynamics' must be a subclass of"
                            " ssm.observations.Observations")

        # Make the emission distn
        emission_classes = dict(
            gaussian=emssn.GaussianEmissions,
            gaussian_orthog=emssn.GaussianOrthogonalEmissions,
            gaussian_id=emssn.GaussianIdentityEmissions,
            gaussian_nn=emssn.GaussianNeuralNetworkEmissions,
            studentst=emssn.StudentsTEmissions,
            studentst_orthog=emssn.StudentsTOrthogonalEmissions,
            studentst_id=emssn.StudentsTIdentityEmissions,
            studentst_nn=emssn.StudentsTNeuralNetworkEmissions,
            t=emssn.StudentsTEmissions,
            t_orthog=emssn.StudentsTOrthogonalEmissions,
            t_id=emssn.StudentsTIdentityEmissions,
            t_nn=emssn.StudentsTNeuralNetworkEmissions,
            poisson=emssn.PoissonEmissions,
            poisson_orthog=emssn.PoissonOrthogonalEmissions,
            poisson_id=emssn.PoissonIdentityEmissions,
            poisson_nn=emssn.PoissonNeuralNetworkEmissions,
            bernoulli=emssn.BernoulliEmissions,
            bernoulli_orthog=emssn.BernoulliOrthogonalEmissions,
            bernoulli_id=emssn.BernoulliIdentityEmissions,
            bernoulli_nn=emssn.BernoulliNeuralNetworkEmissions,
            ar=emssn.AutoRegressiveEmissions,
            ar_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            ar_id=emssn.AutoRegressiveIdentityEmissions,
            ar_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
            autoregressive=emssn.AutoRegressiveEmissions,
            autoregressive_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            autoregressive_id=emssn.AutoRegressiveIdentityEmissions,
            autoregressive_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
            hierarchical_bernoulli = hierBern.HierarchicalBernoulliEmissions)

        if isinstance(emissions, str):
            emissions = emissions.lower()
            if emissions not in emission_classes:
                raise Exception("Invalid emission model: {}. Must be one of {}".
                    format(emissions, list(emission_classes.keys())))

            emission_kwargs = emission_kwargs or {}
            emissions = emission_classes[emissions](N, 1, D, M=M,
                single_subspace=True, **emission_kwargs)
        if not isinstance(emissions, emssn.Emissions):
            raise TypeError("'emissions' must be a subclass of"
                            " ssm.emissions.Emissions")

        init_state_distn = isd.InitialStateDistribution(1, D, M)
        transitions = trans.StationaryTransitions(1, D, M)
        super().__init__(N, 1, D, M=M,
                         init_state_distn=init_state_distn,
                         transitions=transitions,
                         dynamics=dynamics,
                         emissions=emissions)

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        return np.ones((variational_mean.shape[0], 1)), \
               np.ones((variational_mean.shape[0], 1, 1)), \
               0

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def log_prior(self):
        return self.dynamics.log_prior() + self.emissions.log_prior()

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        warnings.warn("Log probability of LDS is not yet implemented.")
        return np.nan
    
    # Copied from https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/ssm/hmm.py#L302
    # Not sure if this is correct!
    # @ensure_args_are_lists
    # def log_probability(self, datas, inputs=None, masks=None, tags=None):
    #     return self.log_prior() + self.log_likelihood(datas, inputs, masks, tags)


    #@ensure_args_are_lists
    # Copied from https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/ssm/hmm.py#L305
    # def log_likelihood(self, datas, inputs=None, masks=None, tags=None):
    #     """
    #     Compute the log probability of the data under the current
    #     model parameters.
       
    #     :param datas: single array or list of arrays of data.
    #     :return total log probability of the data.
    #     """
    #     ll = 0
    #     for data, input, mask, tag in zip(datas, inputs, masks, tags):
    #         pi0 = self.init_state_distn.initial_state_distn
    #         Ps = self.transitions.transition_matrices(data, input, mask, tag)
    #         log_likes = self.dynamics.log_likelihoods(data, input, mask, tag)

    #         ll += hmm_normalizer(pi0, Ps, log_likes)
    #         assert np.isfinite(ll)
    #     return ll
 
    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        (_, x, y) = super().sample(T, input=input, tag=tag, prefix=prefix, with_noise=with_noise)
        return (x, y)