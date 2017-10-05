from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from gpflow.param import Param, ParamList
from gpflow.model import Model
from gpflow import transforms, conditionals, kullback_leiblers
from gpflow.mean_functions import Zero
from gpflow._settings import settings
from gpflow.minibatch import MinibatchData
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
            gauss_kl_white = lambda x: kullback_leiblers.gauss_kl(x[0], x[1])
            KL_shared = tf.map_fn(gauss_kl_white, (self.q_mu_shared, self.q_sqrt_shared), dtype=float_type)
            KL_tasks = tf.map_fn(gauss_kl_white, (self.q_mu_tasks, self.q_sqrt_tasks), dtype=float_type)
        else:
            K = lambda x: x.K(self.Z) + tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level
            K_shared = tf.map_fn(K, self.kerns_shared, dtype=float_type)
            K_tasks = tf.map_fn(K, self.kerns_tasks, dtype=float_type)

            KL_shared = tf.map_fn(gauss_kl_white, (self.q_mu_shared, self.q_sqrt_shared, K_shared), dtype=float_type)
            KL_tasks = tf.map_fn(gauss_kl_white, (self.q_mu_tasks, self.q_sqrt_tasks, K_tasks), dtype=float_type)
        return tf.reduce_sum(KL_shared) + tf_reduce_sum(KL_tasks)

    def latent_conditionals(self, Xnew, full_cov=False):
        def f_conditional(x):
            mean, variance = conditionals.conditional(Xnew, x[0], x[1], x[2],
                                               q_sqrt=x[3], full_cov=full_con,
                                               whiten=self.whiten)
            return mean + self.mean_function(Xnew), variance

        mu_shared, var_shared = tf.map_fn(f_conditional, (self.Z_shared,
                                                          self.kern_shared,
                                                          self.q_mu_shared,
                                                          self.q_sqrt_shared),
                                          dtype=float_type)
        mu_tasks, var_tasks = tf.map_fn(f_conditional, (self.Z_tasks,
                                                          self.kern_tasks,
                                                          self.q_mu_tasks,
                                                          self.q_sqrt_tasks),
                                          dtype=float_type)
        return (mu_shared, var_shared), (mu_tasks, var_tasks)

    def build_predict(self, Xnew, full_cov=False):
        (mu_shared, var_shared), (mu_tasks, var_tasks) = self.latent_conditionals(Xnew, full_cov)
        return tf.reduce_sum(mu_shared) + tf.reduce_sum(mu_tasks), tf.reduce_sum(var_shared) + tf.reduce_sum(var_tasks)

#################################TO BE WORKED ON#####################################################
#####################################################################################################

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

#################################COPY=PASTE JOB~#####################################################
#####################################################################################################

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.build_predict(Xnew, full_cov=True)

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

