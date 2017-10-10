import sys

sys.path.append("../src/")

import unittest
import gpflow
import numpy as np

from cogp import COGP

class COGPTest(unittest.TestCase):
    def setUp(self):
        np.random.RandomState(0)
        self.num_samples = 100
        self.num_inducing = 10
        self.num_tasks = 2
        self.num_latent = 2
        self.dims = 1
        self.noise = 0.01
        self.X = 10*np.random.rand(self.num_samples,self.dims)
        f1 = lambda x: np.sin(x)
        f2 = lambda x: -np.sin(x)
        self.Y1 = f1(self.X) + self.noise*np.random.randn(self.num_samples,1)
        self.Y2 = f2(self.X) + self.noise*np.random.randn(self.num_samples,1)
        self.Z = 10*np.random.rand(self.num_inducing, self.dims)

        self.X_aug = np.vstack([np.hstack([self.X, np.zeros_like(self.X)]), np.hstack([self.X, np.ones_like(self.X)])])
        self.Z_aug = np.vstack([np.hstack([self.Z, np.zeros_like(self.Z)]), np.hstack([self.Z, np.ones_like(self.Z)])])
        self.Y_aug = np.vstack([np.hstack([self.Y1, np.zeros_like(self.Y1)]), np.hstack([self.Y2, np.ones_like(self.Y2)])])

        self.shared = [gpflow.kernels.RBF(1, active_dims=[0]) for _ in
                       range(self.num_latent)]
        self.tasks = [gpflow.kernels.RBF(1, active_dims=[0]) for _ in range(self.num_tasks)]

        self.lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])

        self.model = COGP(self.X_aug, self.Y_aug, self.shared, self.tasks, self.lik, self.Z_aug)

    def test_constructor(self):
        #Check number of tasks is valid
        with self.subTest():
            self.assertEqual(self.model.num_tasks, self.num_tasks)
        #Check we have the correct number of inducing points
        with self.subTest():
            self.assertEqual(self.model.Z_tasks.value.shape[0],
                         self.num_inducing*self.num_tasks)
        with self.subTest():
            self.assertEqual(self.model.Z_shared[0].value.shape[0],
                         self.num_inducing*self.num_tasks)
        #Check we have the number of latent process params
        with self.subTest():
            self.assertEqual(len(self.model.Z_shared), self.num_latent)
        with self.subTest():
            self.assertEqual(len(self.model.q_mu_shared), self.num_latent)
        with self.subTest():
            self.assertEqual(len(self.model.q_sqrt_shared), self.num_latent)

        #Check we have the dimensions for variational params
        with self.subTest():
            self.assertEqual(self.model.q_mu_tasks.shape,
                         (self.num_inducing*self.num_tasks, 1))
        with self.subTest():
            self.assertEqual(self.model.q_mu_shared[0].shape,
                         (self.num_inducing*self.num_tasks, 1))

    def test_log_likelihood(self):
        self.assertLess(self.model.compute_log_likelihood(), 0.0)

    def test_optimize(self):
        ll0 = self.model.compute_log_likelihood()
        self.model.optimize(maxiter=10)
        ll1 = self.model.compute_log_likelihood()
        self.assertGreaterEqual(ll1, ll0)


if __name__ == "__main__":
    unittest.main()
