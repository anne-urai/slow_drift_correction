from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd import hessian

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus, random_rotation
from ssm.preprocessing import interpolate_data, pca_with_imputation
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs, convex_combination
from ssm.stats import independent_studentst_logpdf, bernoulli_logpdf
import scipy.stats as spstats

# Observation models for SLDS
class Emissions(object):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        self.N, self.K, self.D, self.M, self.single_subspace = \
            N, K, D, M, single_subspace
            
    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag, x):
        raise NotImplementedError

    def forward(self, x, input=None, tag=None):
        raise NotImplementedError

    def invert(self, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def sample(self, z, x, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        warn("Analytical Hessian is not implemented for this Emissions class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        obj = lambda xt, datat, inputt, maskt: \
            self.log_likelihoods(datat[None,:], inputt[None,:], maskt[None,:], tag, xt[None,:])[0, 0]
        hess = hessian(obj)
        terms = np.array([np.squeeze(hess(xt, datat, inputt, maskt))
                          for xt, datat, inputt, maskt in zip(x, data, input, mask)])
        return -1 * terms

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, **kwargs):
        """
        If M-step in Laplace-EM cannot be done in closed form for the emissions, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log likelihood
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = 0
            obj += self.log_prior()
            for data, input, mask, tag, x, (Ez, _, _) in \
                zip(datas, inputs, masks, tags, continuous_expectations, discrete_expectations):
                obj += np.sum(Ez * self.log_likelihoods(data, input, mask, tag, x))
            return -obj / T

        # Optimize emissions log-likelihood
        self.params = optimizer(_objective, self.params,
                                num_iters=maxiter,
                                suppress_warnings=True,
                                **kwargs)
        
  




class BernoulliEmission(Emissions):
    def __init__(self, N, K, D, M=0, 
                 mean_Cs=None,
                 variance_Cs=1,
                 mean_Fs=None,
                 variance_Fs=1,
                 mean_ds=None,
                 variance_ds=1,
                 initialize="global",
                 single_subspace=True, link="logit", **kwargs):
        
        super(BernoulliEmission, self).__init__(N, K, D, M=M, single_subspace=single_subspace)


        # if np.zeros then we get a singular matrix so should differ from 0!
        # TODO: fix this such that we also draw a value for Cs in global model
        # is this correct? see HierarchicalBernoulliEmissions _update_hierarchical_prior (line 164)
        # aren't we resetting the prior to 1 all the time then?
        mean_Cs = np.ones((1, N, D))
        mean_Fs = np.ones((1, N, M))
        mean_ds = np.ones((1, N))
 
    
        self.set_prior(mean_Cs, variance_Cs,
                       mean_Fs, variance_Fs,
                       mean_ds, variance_ds)
 
    
        # Initialize the dynamics 
        # For global model parameter should just be mean (in this case just zero)
        if initialize.lower() == "global":
            
            self.Cs = mean_Cs
            self.Fs = mean_Fs
            self.ds = mean_ds
 
        # For group model parameter draw for each tag from normal distribution with pre-specified mean and variance
        elif initialize.lower() == "group":
        
            self.Cs = np.array([spstats.norm.rvs(mean_Cs, np.sqrt(variance_Cs))], ndmin=3) 
            self.Fs = np.array([spstats.norm.rvs(mean_Fs, np.sqrt(variance_Fs), M)], ndmin=3) 
            self.ds = np.array([spstats.norm.rvs(mean_ds, np.sqrt(variance_ds))], ndmin=2) 
 
        else:
            raise Exception("Invalid initialization method: {}".format(initialize))
             
        assert self.Cs.shape == (1, N, D)
        assert self.Fs.shape == (1, N, M)
        assert self.ds.shape == (1, N)  
            
        
        # comes from BernoulliMixing
        self.link_name = link
        mean_functions = dict(
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]



    # TODO: check if this is correct? See original ssm.emissions
    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        self._Cs = value
        
        
    @property
    def C(self):
        return self.Cs[0]

    @C.setter
    def C(self, value):
        assert value.shape == self.Cs[0].shape
        self.Cs[0] = value
        
        

    @property
    def params(self):
        return self.Cs, self.Fs, self.ds
    
    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value
    
    # TODO: change dimension params that belong to observations (K,D,D) to emission params (N,D,M), see above   
    def set_prior(self, mean_Cs, variance_Cs,
                        mean_Fs, variance_Fs,
                        mean_ds, variance_ds):
        """
        :param E_A:          (K, D, D * lags) array
        :param variance_A:        positive scalar
        :param E_V:          (K, D, M) array
        :param variance_V:        positive scalar
        :param E_b:          (K, D) array
        :param variance_b:        positive scalar
        :param E_Sigma:      (K, D, D) array of psd matrices
        :param var_Sigma:    positive scalar
        """
        N, D, M = self.N, self.D, self.M

        # Check inputs
        assert all([(np.isscalar(x) and x > 0) for x in [variance_Cs, variance_Fs, variance_ds]])
        assert mean_Cs.shape == (1, N, D)
        assert mean_Fs.shape == (1, N, M)
        assert mean_ds.shape == (1, N)

        # Store the mean parameters
        self.mean_Cs = mean_Cs
        self.variance_Cs = variance_Cs
        self.mean_Fs = mean_Fs
        self.variance_Fs = variance_Fs
        self.mean_ds = mean_ds
        self.variance_ds = variance_ds


        # Compute natural parameters of Gaussian prior on (C, F, d) weight matrix
        # given these means and variances
        J0_diag = np.concatenate((1 / variance_Cs * np.ones(D * 1),
                                  1 / variance_Fs * np.ones(M),
                                  1 / variance_ds * np.ones(1)))
        self.J0 = np.tile(np.diag(J0_diag)[None, :, :], (1, 1, 1))

        # h0: (K, D, D * lags + M + 1)
        self.h0 = np.concatenate([mean_Cs / variance_Cs,
                                  mean_Fs / variance_Fs,
                                  mean_ds[..., None] / variance_ds], axis=2)

        # TODO: (KEEP AN EYE ON WHETHER THIS IS IMPORTANT) Note:  we have to transpose h0 per convention in other AR classes
        self.h0 = np.swapaxes(self.h0, -1, -2)

    
    # added from observations
    def compute_sample_size(self, datas, inputs, masks, tags):
        """
        Count the number of data points in the given data arrays.
        This is necessary for the stochastic_m_step function.
        """
        return sum([data.shape[0] for data in datas])
    
    def log_prior(self):

        lp = np.sum(spstats.norm.logpdf(self.Cs, self.mean_Cs,
                                        np.sqrt(self.variance_Cs)))

        lp += np.sum(spstats.norm.logpdf(self.Fs, self.mean_Fs,
                                         np.sqrt(self.variance_Fs)))

        lp += np.sum(spstats.norm.logpdf(self.ds, self.mean_ds,
                                         np.sqrt(self.variance_ds)))

        return lp
    
    
    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]
    
    
    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        # Invert with the average emission parameters
        
        
        # C = np.mean(self.Cs, axis=0)
        # F = np.mean(self.Fs, axis=0)
        # d = np.mean(self.ds, axis=0)
        
        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)
    
    
    def forward(self, x, input, tag):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
            + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
            + self.ds
    
    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K
    
        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))
    
        # shouldn't everything be concatenated?
        #data = np.concatenate(datas) 
        #input = np.concatenate(inputs)
        
        
        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]
    
        # Run PCA to get a linear embedding of the data
        pca, xs, ll = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)
    
        self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
        self.ds = np.tile(pca.mean_[None, :], (Keff, 1))
    
        return pca
    
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)
    
    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == bool or (data.dtype == int and data.min() >= 0 and data.max() <= 1)
        assert self.link_name == "logit", "Log likelihood is only implemented for logit link."
        logit_ps = self.forward(x, input, tag)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return bernoulli_logpdf(data[:, None, :], logit_ps, mask=mask[:, None, :])

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, .9))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.forward(x, input, tag))
        return (npr.rand(T, self.N) < ps[np.arange(T), z,:]).astype(int)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.forward(variational_mean, input, tag))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)
    


    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess



    # added from observations
    # TODO: change K,D,M?
    def expected_sufficient_stats(self, expectations, datas, inputs, masks, tags):
        K, D, M, lags = self.K, self.D, self.M, 1
        D_in = D * lags + M + 1
    
        # Initialize the outputs
        ExuxuTs = np.zeros((K, D_in, D_in))
        ExuyTs = np.zeros((K, D_in, D))
        EyyTs = np.zeros((K, D, D))
        Ens = np.zeros(K)
    
        # Return early if no data is given
        if len(expectations) == 0:
            return ExuxuTs, ExuyTs, EyyTs, Ens
    
        # Iterate over data arrays and discrete states
        for (Ez, _, _), data, input in zip(expectations, datas, inputs):
            u = input[lags:]
            y = data[lags:]
            for k in range(K):
                w = Ez[lags:, k]
    
                # ExuxuTs[k]
                for l1 in range(lags):
                    x1 = data[lags-l1-1:-l1-1]
                    # Cross terms between lagged data and other lags
                    for l2 in range(l1, lags):
                        x2 = data[lags - l2 - 1:-l2 - 1]
                        ExuxuTs[k, l1*D:(l1+1)*D, l2*D:(l2+1)*D] += np.einsum('t,ti,tj->ij', w, x1, x2)
    
                    # Cross terms between lagged data and inputs and bias
                    ExuxuTs[k, l1*D:(l1+1)*D, D*lags:D*lags+M] += np.einsum('t,ti,tj->ij', w, x1, u)
                    ExuxuTs[k, l1*D:(l1+1)*D, -1] += np.einsum('t,ti->i', w, x1)
    
                ExuxuTs[k, D*lags:D*lags+M, D*lags:D*lags+M] += np.einsum('t,ti,tj->ij', w, u, u)
                ExuxuTs[k, D*lags:D*lags+M, -1] += np.einsum('t,ti->i', w, u)
                ExuxuTs[k, -1, -1] += np.sum(w)
    
                # ExuyTs[k]
                for l1 in range(lags):
                    x1 = data[lags - l1 - 1:-l1 - 1]
                    ExuyTs[k, l1*D:(l1+1)*D, :] += np.einsum('t,ti,tj->ij', w, x1, y)
                ExuyTs[k, D*lags:D*lags+M, :] += np.einsum('t,ti,tj->ij', w, u, y)
                ExuyTs[k, -1, :] += np.einsum('t,ti->i', w, y)
    
                # EyyTs[k] and Ens[k]
                EyyTs[k] += np.einsum('t,ti,tj->ij',w,y,y)
                Ens[k] += np.sum(w)
    
        # Symmetrize the expectations
        for k in range(K):
            for l1 in range(lags):
                for l2 in range(l1, lags):
                    ExuxuTs[k, l2*D:(l2+1)*D, l1*D:(l1+1)* D] = ExuxuTs[k, l1*D:(l1+1)*D, l2*D:(l2+1)*D].T
                ExuxuTs[k, D*lags:D*lags+M, l1*D:(l1+1)*D] = ExuxuTs[k, l1*D:(l1+1)*D, D*lags:D*lags+M].T
                ExuxuTs[k, -1, l1*D:(l1+1)*D] = ExuxuTs[k, l1*D:(l1+1)*D, -1].T
            ExuxuTs[k, -1, D*lags:D*lags+M] = ExuxuTs[k, D*lags:D*lags+M, -1].T
    
        return ExuxuTs, ExuyTs, EyyTs, Ens
    



    # copied from class AutoRegressiveObservations in ssm.observations
    def m_step(self, expectations, datas, inputs, masks, tags,
               sufficient_stats=None, **kwargs):
        """Compute M-step for Gaussian Auto Regressive Observations.

        If `continuous_expectations` is not None, this function will
        compute an exact M-step using the expected sufficient statistics for the
        continuous states. In this case, we ignore the prior provided by (J0, h0),
        because the calculation is exact. `continuous_expectations` should be a tuple of
        (Ex, Ey, ExxT, ExyT, EyyT).

        If `continuous_expectations` is None, we use `datas` and `expectations,
        and (optionally) the prior given by (J0, h0). In this case, we estimate the sufficient
        statistics using `datas,` which is typically a single sample of the continuous
        states from the posterior distribution.
        """
        K, D, M, lags = self.K, self.D, self.M, 1

        # Collect sufficient statistics
        if sufficient_stats is None:
            ExuxuTs, ExuyTs, EyyTs, Ens = \
                self.expected_sufficient_stats(expectations,
                                               datas,
                                               inputs,
                                               masks,
                                               tags)
        else:
            # ExuxuTs, ExuyTs, EyyTs, Ens = \
            #     self._extend_given_sufficient_statistics(expectations, continuous_expectations, inputs)
            ExuxuTs, ExuyTs, EyyTs, Ens = sufficient_stats

        # Solve the linear regressions
        #
        # Note: this is performing the M-step in two stages, first updating
        #       weights W = (A, V, b) as if the observation variance was identity,
        #       then updating the variance given the new weights.  This is only an
        #       issue when there is a non-trivial prior on the weights and the
        #       covariance matrix, since otherwise the optimal weights are independent
        #       of the covariance of the errors.  When there is a strong prior, as in a
        #       hierarchical model, you technically have to account for the covariance
        #       of the errors when updating weights, since it affects how the prior and
        #       likelihood are combined.  Alternatively, we could use a conjugate
        #       matrix-normal-inverse-Wishart (MNIW) prior on (W, Sigma) jointly and
        #       then solve for the mode of this posterior.
        Cs = np.zeros((K, D, D * lags))
        Fs = np.zeros((K, D, M))
        ds = np.zeros((K, D))

        for k in range(K):
            Wk = np.linalg.solve(ExuxuTs[k] + self.J0[k], ExuyTs[k] + self.h0[k]).T
            Cs[k] = Wk[:, :D * lags]
            Fs[k] = Wk[:, D * lags:-1]
            ds[k] = Wk[:, -1]


        # If any states are unused, set their parameters to a perturbation of a used state
        unused = np.where(Ens < 1)[0]
        used = np.where(Ens > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = npr.choice(used)
                Cs[k] = Cs[i] + 0.01 * npr.randn(*Cs[i].shape)
                Fs[k] = Fs[i] + 0.01 * npr.randn(*Fs[i].shape)
                ds[k] = ds[i] + 0.01 * npr.randn(*ds[i].shape)

        # Update parameters via their setter
        self.Cs = Cs
        self.Fs = Fs
        self.ds = ds

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
        stats = self.expected_sufficient_stats(expectations,
                                               datas,
                                               inputs,
                                               masks,
                                               tags)

        # Scale the stats by the fraction of the sample size
        this_sample_size = self.compute_sample_size(datas, inputs, masks, tags)
        stats = tuple(map(lambda x: x * total_sample_size /  this_sample_size, stats))

        # Combine them with the running average sufficient stats
        if optimizer_state is not None:
            stats = convex_combination(optimizer_state, stats, step_size)

        # Call the regular m-step with these sufficient statistics
        self.m_step(None, None, None, None, None, sufficient_stats=stats)

        # Return the update state (i.e. the new stats)
        return stats


















# Many emissions models start with a linear layer
class _LinearEmissions(Emissions):
    """
    A simple linear mapping from continuous states x to data y.

        E[y | x] = Cx + d + Fu

    where C is an emission matrix, d is a bias, F an input matrix,
    and u is an input.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_LinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer.  Set _Cs to be private so that it can be
        # changed in subclasses.
        self._Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)     #original
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)


    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        K, N, D = self.K, self.N, self.D
        assert value.shape == (1, N, D) if self.single_subspace else (K, N, D)
        self._Cs = value

    @property
    def params(self):
        return self.Cs, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        assert self.single_subspace, "Can only invert with a single emission model"

        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, x, input, tag):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
            + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
            + self.ds

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data
        pca, xs, ll = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)

        self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
        self.ds = np.tile(pca.mean_[None, :], (Keff, 1))

        return pca



class _OrthogonalLinearEmissions(_LinearEmissions):
    """
    A linear emissions matrix constrained such that the emissions matrix
    is orthogonal. Use the rational Cayley transform to parameterize
    the set of orthogonal emission matrices. See
    https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
    for a derivation of the rational Cayley transform.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_OrthogonalLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer
        assert N > D
        self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
        self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Set the emission matrix to be a random orthogonal matrix
        C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        for k in range(C0.shape[0]):
            C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
        self.Cs = C0

    @property
    def Cs(self):
        D = self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        Bs = 0.5 * (self._Ms - T(self._Ms))    # Bs is skew symmetric
        Fs = np.matmul(T(self._As), self._As) - Bs
        trm1 = np.concatenate((np.eye(D) - Fs, 2 * self._As), axis=1)
        trm2 = np.eye(D) + Fs
        Cs = T(np.linalg.solve(T(trm2), T(trm1)))
        assert np.allclose(
            np.matmul(T(Cs), Cs),
            np.tile(np.eye(D)[None, :, :], (Cs.shape[0], 1, 1))
            )
        return Cs

    @Cs.setter
    def Cs(self, value):
        N, D = self.N, self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        # Make sure value is the right shape and orthogonal
        Keff = 1 if self.single_subspace else self.K
        assert value.shape == (Keff, N, D)
        assert np.allclose(
            np.matmul(T(value), value),
            np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
            )

        Q1s, Q2s = value[:, :D, :], value[:, D:, :]
        Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
        # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
        self._Ms = T(Fs)
        self._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
        assert np.allclose(self.Cs, value)

    @property
    def params(self):
        return self._As, self._Ms, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self._As, self._Ms, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self._As = self._As[perm]
            self._Ms = self._Ms[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]


# Sometimes we just want a bit of additive noise on the observations
class _IdentityEmissions(Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_IdentityEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        assert N == D

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass

    def forward(self, x, input, tag):
        return x[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is just the data
        """
        return np.copy(data)


# Allow general nonlinear emission models with neural networks
class _NeuralNetworkEmissions(Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_NeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        # Initialize the neural network weights
        assert N > D
        layer_sizes = (D + M,) + hidden_layer_sizes + (N,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        inputs = np.column_stack((x, input))
        for W, b in zip(self.weights, self.biases):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is... who knows!
        """
        return npr.randn(data.shape[0], self.D)


# Observation models for SLDS
class _GaussianEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_GaussianEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -1 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_GaussianEmissionsMixin, self).params + (self.inv_etas,)

    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(_GaussianEmissionsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_GaussianEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        return mus[np.arange(T), z, :] + np.sqrt(etas[z]) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:, 0, :] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class GaussianEmissions(_GaussianEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        # pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        # self.inv_etas[:,...] = np.log(pca.noise_variance_)
        pass

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
            hess = np.tile(block[None,:,:], (T, 1, 1))
        else:
            blocks = np.array([-1.0 * C.T@np.diag(1.0/np.exp(inv_eta))@C
                               for C, inv_eta in zip(self.Cs, self.inv_etas)])
            hess = np.sum(Ez[:,:,None,None] * blocks, axis=1)
        return -1 * hess

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, **kwargs):

        if self.single_subspace and np.all(masks):
            # Return exact m-step updates for C, F, d, and inv_etas
            # stack across all datas
            x = np.vstack(continuous_expectations)
            u = np.vstack(inputs)
            y = np.vstack(datas)
            T, D = np.shape(x)
            xb = np.hstack((np.ones((T,1)),x,u)) # design matrix
            params = np.linalg.lstsq(xb.T@xb, xb.T@y, rcond=None)[0].T
            self.ds = params[:,0].reshape((1,self.N))
            self.Cs = params[:,1:D+1].reshape((1,self.N,self.D))
            if self.M > 0:
                self.Fs = params[:,D+1:].reshape((1,self.N,self.M))
            mu = np.dot(xb, params.T)
            Sigma = (y-mu).T@(y-mu) / T
            self.inv_etas = np.log(np.diag(Sigma)).reshape((1,self.N))
        else:
            Emissions.m_step(self, discrete_expectations, continuous_expectations,
                             datas, inputs, masks, tags,
                             optimizer=optimizer, maxiter=maxiter, **kwargs)


class GaussianOrthogonalEmissions(_GaussianEmissionsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
            hess = np.tile(block[None,:,:], (T, 1, 1))
        else:
            blocks = np.array([-1.0 * C.T@np.diag(1.0/np.exp(inv_eta))@C
                               for C, inv_eta in zip(self.Cs, self.inv_etas)])
            hess = np.sum(Ez[:,:,None,None] * blocks, axis=1)
        return -1 * hess


class GaussianIdentityEmissions(_GaussianEmissionsMixin, _IdentityEmissions):

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * np.diag(1.0 / np.exp(self.inv_etas[0]))
            hess = np.tile(block[None, :, :], (T, 1, 1))
        else:
            raise NotImplementedError

        return -1 * hess


class GaussianNeuralNetworkEmissions(_GaussianEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _StudentsTEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_StudentsTEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_nus = np.log(4) * np.ones((1, N)) if single_subspace else np.log(4) * np.ones(K, N)

    @property
    def params(self):
        return super(_StudentsTEmissionsMixin, self).params + (self.inv_etas, self.inv_nus)

    @params.setter
    def params(self, value):
        super(_StudentsTEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_StudentsTEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]
            self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        etas, nus = np.exp(self.inv_etas), np.exp(self.inv_nus)
        mus = self.forward(x, input, tag)
        return independent_studentst_logpdf(data[:, None, :],
                                            mus, etas, nus, mask=mask[:, None, :])

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        nus = np.exp(self.inv_nus)
        etas = np.exp(self.inv_etas)
        taus = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[np.arange(T), z, :] + np.sqrt(etas[z] / taus) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class StudentsTEmissions(_StudentsTEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTOrthogonalEmissions(_StudentsTEmissionsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTIdentityEmissions(_StudentsTEmissionsMixin, _IdentityEmissions):
    pass


class StudentsTNeuralNetworkEmissions(_StudentsTEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _BernoulliEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="logit", **kwargs):
        super(_BernoulliEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        mean_functions = dict(
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == bool or (data.dtype == int and data.min() >= 0 and data.max() <= 1)
        assert self.link_name == "logit", "Log likelihood is only implemented for logit link."
        logit_ps = self.forward(x, input, tag)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return bernoulli_logpdf(data[:, None, :], logit_ps, mask=mask[:, None, :])

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, .9))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.forward(x, input, tag))
        return (npr.rand(T, self.N) < ps[np.arange(T), z,:]).astype(int)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.forward(variational_mean, input, tag))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)


class BernoulliEmissions(_BernoulliEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess




class BernoulliOrthogonalEmissions(_BernoulliEmissionsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess


class BernoulliIdentityEmissions(_BernoulliEmissionsMixin, _IdentityEmissions):
    pass


class BernoulliNeuralNetworkEmissions(_BernoulliEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _PoissonEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0, **kwargs):

        super(_PoissonEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        self.bin_size = bin_size
        mean_functions = dict(
            log=self._log_mean,
            softplus=self._softplus_mean
            )
        self.mean = mean_functions[link]
        link_functions = dict(
            log=self._log_link,
            softplus=self._softplus_link
            )
        self.link = link_functions[link]

        # Set the bias to be small if using log link
        if link == "log":
            self.ds = -3 + .5 * npr.randn(1, N) if single_subspace else npr.randn(K, N)

    def _log_mean(self, x):
        return np.exp(x) * self.bin_size

    def _softplus_mean(self, x):
        return softplus(x) * self.bin_size

    def _log_link(self, rate):
        return np.log(rate) - np.log(self.bin_size)

    def _softplus_link(self, rate):
        return inv_softplus(rate / self.bin_size)

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int
        lambdas = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, np.inf))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.forward(x, input, tag))
        y = npr.poisson(lambdas[np.arange(T), z, :])
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        return lambdas[:,0,:] if self.single_subspace else np.sum(lambdas * expected_states[:,:,None], axis=1)

class PoissonEmissions(_PoissonEmissionsMixin, _LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(PoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        # Scale down the measurement and control matrices so that
        # rate params don't explode when exponentiated.
        if self.link_name == "log":
            self.Cs /= np.exp(np.linalg.norm(self.Cs, axis=2)[:,:,None])
            self.Fs /= np.exp(np.linalg.norm(self.Fs, axis=2)[:,:,None])

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")

        if self.link_name == "log":
            lambdas = self.mean(self.forward(x, input, tag))
            hess = np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])
            return -1 * hess

        elif self.link_name == "softplus":
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            expterms = np.exp(-np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0])
            diags = (data / lambdas * (expterms - 1.0 / lambdas) - expterms * self.bin_size) / (1.0+expterms)**2
            hess = np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
            return -1 * hess

        else:
            raise Exception("No Hessian calculation for link: {}".format(self.link_name))


class PoissonOrthogonalEmissions(_PoissonEmissionsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")

        if self.link_name == "log":
            lambdas = self.mean(self.forward(x, input, tag))
            hess = np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])
            return -1 * hess

        elif self.link_name == "softplus":
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            expterms = np.exp(-np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0])
            diags = (data / lambdas * (expterms - 1.0 / lambdas) - expterms * self.bin_size) / (1.0+expterms)**2
            hess = np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
            return -1 * hess

        else:
            raise Exception("No Hessian calculation for link: {}".format(self.link_name))

class PoissonIdentityEmissions(_PoissonEmissionsMixin, _IdentityEmissions):
    pass


class PoissonNeuralNetworkEmissions(_PoissonEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _AutoRegressiveEmissionsMixin(object):
    """
    Include past observations as a covariate in the SLDS emissions.
    The AR model is restricted to be diagonal.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_AutoRegressiveEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        # Initialize AR component of the model
        self.As = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Shrink the eigenvalues of the A matrices to avoid instability.
        # Since the As are diagonal, this is just a clip.
        self.As = np.clip(self.As, -1.0 + 1e-8, 1 - 1e-8)

    @property
    def params(self):
        return super(_AutoRegressiveEmissionsMixin, self).params + (self.As, self.inv_etas)

    @params.setter
    def params(self, value):
        self.As, self.inv_etas = value[-2:]
        super(_AutoRegressiveEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_AutoRegressiveEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.As = self.inv_nus[perm]
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        mus = mus + np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :]))

        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        assert self.single_subspace, "Can only invert with a single emission model"
        pad = np.zeros((1, self.N))
        resid = data - np.concatenate((pad, self.As * data[:-1]))
        return self._invert(resid, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T, N = z.shape[0], self.N
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)

        y = np.zeros((T, N))
        y[0] = mus[0, z[0], :] + np.sqrt(etas[z[0]]) * npr.randn(N)
        for t in range(1, T):
            y[t] = mus[t, z[t], :] + self.As[z[t]] * y[t-1] + np.sqrt(etas[z[0]]) * npr.randn(N)
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        mus[1:] += self.As[None, :, :] * data[:-1, None, :]
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)


class AutoRegressiveEmissions(_AutoRegressiveEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

        # Solve a linear regression for the AR coefficients.
        from sklearn.linear_model import LinearRegression
        for n in range(self.N):
            lr = LinearRegression()
            lr.fit(np.concatenate([d[:-1, n] for d in datas])[:,None],
                   np.concatenate([d[1:, n] for d in datas]))
            self.As[:,n] = lr.coef_[0]

        # Compute the residuals of the AR model
        pad = np.zeros((1,self.N))
        mus = [np.concatenate((pad, self.As[0] * d[:-1])) for d in datas]
        residuals = [data - mu for data, mu in zip(datas, mus)]

        # Run PCA on the residuals to initialize C and d
        pca = self._initialize_with_pca(residuals, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class AutoRegressiveOrthogonalEmissions(_AutoRegressiveEmissionsMixin, _OrthogonalLinearEmissions):
    pass


class AutoRegressiveIdentityEmissions(_AutoRegressiveEmissionsMixin, _IdentityEmissions):
    pass


class AutoRegressiveNeuralNetworkEmissions(_AutoRegressiveEmissionsMixin, _NeuralNetworkEmissions):
    pass
