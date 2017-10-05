from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .param import Param, ParamList
from .model import Model
from . import transforms, conditionals, kullback_leiblers
from .mean_functions import Zero
from ._settings import settings
from .minibatch import MinibatchData
float_type = settings.dtypes.float_type


class COGP(Model):
    """
    This is the Collaborative Multi-output GP model. The key reference is

    ::

      @inproceedings{nguyen_collaborative_2014,
        address = {Quebec City, Canada},
        title = {Collaborative {Multi}-output {Gaussian} {Processes}},
        booktitle = {Proceedings  of  the  {Conference}  on
                     {Uncertainty}  in  {Artificial}  {Intelligence}},
        author = {Nguyen, Trung V. and Bonilla, Edwin V.},
        year = {2014},
        }

    """
    def __init__(self, X, Y, kerns_shared, kerns_tasks, likelihood, Z, mean_function=None,
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kerns_joint is a list of kernels for the shared processes, length >=1
        - kerns_tasks is a list of kernels for the each specific task,
          length = number of tasks
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        """
        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))

        num_shared = len(kerns_shared)
        num_tasks = len(kerns_tasks)

        assert np.unique(X[:,-1]).shape[0] == num_tasks

        # init the super class, accept args
        Model.__init__(self)
        self.X = X
        self.Y = Y
        self.kerns_shared = kerns.shared
        self.kerns_specific = kerns.specific
        self.likelihood = likelihood
        self.mean_function = mean_function or Zero()
        self.q_diag, self.whiten = q_diag, whiten
        self.num_shared = num_shared
        self.num_tasks = num_tasks
        self.Z_shared = ParamList([Param(Z.copy()) for _ in range(self.num_shared)])
        self.Z_tasks = ParamList([Param(Z.copy()) for _ in range(self.num_tasks)])
        self.num_latent = num_latent or Y.shape[1]-1
        self.num_inducing = Z.shape[0]

        # init variational parameters
        # One parameter set for each shared and task specific processes
        q_mu = np.zeros((self.num_inducing, self.num_latent))
        q_mu_shared = [Param(q_mu.copy()) for _ in range(self.num_shared)]
        q_mu_tasks = [Param(q_mu.copy()) for _ in range(self.num_tasks)]
        self.q_mu_shared = ParamList(q_mu_shared)
        self.q_mu_tasks = ParamList(q_mu_tasks)
        if self.q_diag:
            q_sqrt = np.ones((self.num_inducing, self.num_latent))
            q_sqrt_shared = [Param(q_sqrt.copy(), transforms.positive) for _ in
                             range(self.num_shared)]
            q_sqrt_tasks = [Param(q_sqrt.copy(), transforms.positive) for _ in
                             range(self.num_tasks)]
            self.q_sqrt_shared = ParamList(q_sqrt_shared)
            self.q_sqrt_tasks = ParamList(q_sqrt_tasks)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            q_sqrt_shared = [Param(q_sqrt.copy(),
                                   transforms.LowerTriangular(self.num_inducing, self.num_latent))
                             for _ in range(self.num_shared)]
            q_sqrt_tasks = [Param(q_sqrt.copy(),
                                   transforms.LowerTriangular(self.num_inducing, self.num_latent))
                             for _ in range(self.num_tasks)]
            self.q_sqrt_shared = ParamList(q_sqrt_shared)
            self.q_sqrt_tasks = ParamList(q_sqrt_tasks)

    def build_prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu, self.q_sqrt)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            K = self.kern.K(self.Z) + tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu, self.q_sqrt, K)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)
        return KL

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self.build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /\
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.Z, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=self.whiten)
        return mu + self.mean_function(Xnew), var
