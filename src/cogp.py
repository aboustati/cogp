from __future__ import absolute_import
import warnings
import tensorflow as tf
import numpy as np
from gpflow.param import Param, ParamList, AutoFlow
from gpflow.model import Model
from gpflow.kernels import Coregion, Add
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
        - X is a data matrix augmented with task index column, size N x D
        - Y is a data matrix augmented with task index column, size N x R
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

        num_shared = len(kerns_shared)
        num_tasks = len(kerns_tasks)
        task_index_col = X.shape[1]-1

        assert Y.shape[1]>1, "Y must have a task index column"
        assert np.all(X[:,-1]==Y[:,-1]), "Task indices in inputs and outputs doesn't match"
        assert X.shape[1:]==Z.shape[1:], "X and Z must have the same number of features"
        assert np.unique(X[:,-1]).shape[0] == num_tasks, "Task indices and number of tasks do not match"

        for k in kerns_shared:
            if isinstance(k.active_dims, slice) or len(k.active_dims)>=X.shape[1]:
                warnings.warn("One of the shared kernels is acting on all dimensions including the task index!")
        for k in kerns_tasks:
            if isinstance(k.active_dims, slice) or len(k.active_dims)>=X.shape[1]:
                warnings.warn("One of the tasks kernels is acting on all dimensions including the task index!")
        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))

        # init the super class, accept args
        Model.__init__(self)
        self.X = X
        self.Y = Y
        self.kerns_shared = ParamList(kerns_shared)
        #self.kerns_tasks = ParamList(kerns_tasks)
        self.likelihood = likelihood
        self.mean_function = mean_function or Zero()
        self.q_diag, self.whiten = q_diag, whiten
        self.num_shared = num_shared
        self.num_tasks = num_tasks
        self.Z_shared = ParamList([Param(Z.copy()) for _ in range(self.num_shared)])
        #self.Z_tasks = ParamList([Param(Z.copy()) for _ in range(self.num_tasks)])
        self.num_latent = num_latent or Y.shape[1]-1
        self.num_inducing = Z.shape[0]

        task_selector = np.eye(self.num_tasks)
        kern_list = []
        Z_tasks = Z.copy()
        for i in range(self.num_tasks):
            coreg = Coregion(1, output_dim=self.num_tasks, rank=1,
                             active_dims=[task_index_col])
            coreg.kappa = task_selector[i]
            coreg.kappa.fixed = True
            k_i = kerns_tasks[i] * coreg
            kern_list.append(k_i)


        self.kerns_shared = Add(kern_list)
        self.Z_tasks = Param(Z_tasks)


        # init variational parameters
        # One parameter set for each shared and task specific processes
        q_mu = np.zeros((self.num_inducing, self.num_latent))
        q_mu_shared = [Param(q_mu.copy()) for _ in range(self.num_shared)]
        q_mu_tasks = q_mu.copy()
        self.q_mu_shared = ParamList(q_mu_shared)
        self.q_mu_tasks = Param(q_mu_tasks)
        if self.q_diag:
            q_sqrt = np.ones((self.num_inducing, self.num_latent))
            q_sqrt_shared = [Param(q_sqrt.copy(), transforms.positive) for _ in
                             range(self.num_shared)]
            q_sqrt_tasks = Param(q_sqrt.copy(), transforms.positive)
            self.q_sqrt_shared = ParamList(q_sqrt_shared)
            self.q_sqrt_tasks = q_sqrt_tasks
        else:
            q_sqrt = np.array([np.eye(self.num_inducing)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            q_sqrt_shared = [Param(q_sqrt.copy(),
                                   transforms.LowerTriangular(self.num_inducing, self.num_latent))
                             for _ in range(self.num_shared)]
            q_sqrt_tasks = Param(q_sqrt.copy(), transforms.LowerTriangular(self.num_inducing, self.num_latent))
            self.q_sqrt_shared = ParamList(q_sqrt_shared)
            self.q_sqrt_tasks = q_sqrt_tasks

    def build_prior_KL(self):
        if self.whiten:
            K = lambda x: None
        else:
            K = lambda x: x.K(self.Z) + tf.eye(self.num_inducing, dtype=float_type) * settings.numerics.jitter_level

        KL_shared = []
        for mu, sqrt, kern in zip(self.q_mu_shared, self.q_sqrt_shared, self.kerns_shared):
            kern_eval = K(kern)
            KL_shared.append(kullback_leiblers.gauss_kl(mu, sqrt, kern_eval))

        KL_tasks = []
        for mu, sqrt, kern in zip(self.q_mu_tasks, self.q_sqrt_tasks, self.kerns_tasks):
            kern_eval = K(kern)
            KL_tasks.append(kullback_leiblers.gauss_kl(mu, sqrt, kern_eval))

        return tf.add_n(KL_shared) + tf.add_n(KL_tasks)

    def latent_conditionals(self, Xnew, full_cov=False):
        def f_conditional(Z, kern, mu, sqrt):
            mean, variance = conditionals.conditional(Xnew, Z, kern, mu,
                                               q_sqrt=sqrt, full_cov=full_cov,
                                               whiten=self.whiten)
            return mean + self.mean_function(Xnew), variance

        mu_shared = []
        var_shared = []
        for Z, kern, mu, sqrt in zip(self.Z_shared,
                                     self.kerns_shared,
                                     self.q_mu_shared,
                                     self.q_sqrt_shared):
            m, v = f_conditional(Z, kern, mu, sqrt)
            mu_shared.append(m)
            var_shared.append(v)

        mu_tasks = []
        var_tasks = []

        for Z, kern, mu, sqrt in zip(self.Z_tasks,
                                     self.kerns_tasks,
                                     self.q_mu_tasks,
                                     self.q_sqrt_tasks):
            m, v = f_conditional(Z, kern, mu, sqrt)
            mu_tasks.append(m)
            var_tasks.append(v)

        return (mu_shared, var_shared), (mu_tasks, var_tasks)

    def build_predict(self, Xnew, full_cov=False):
        (mu_shared, var_shared), (mu_tasks, var_tasks) = self.latent_conditionals(Xnew, full_cov)
        sum_mu_shared = tf.add_n(mu_shared)
        sum_mu_tasks = tf.add_n(mu_tasks)
        sum_var_shared = tf.add_n(var_shared)
        sum_var_tasks = tf.add_n(var_tasks)
        return tf.add(sum_mu_shared, sum_mu_tasks), tf.add(sum_var_shared, sum_var_tasks)


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

