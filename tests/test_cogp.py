import sys

sys.path.append("../src/")

import unittest
import gpflow
import numpy as np

from cogp import COGP

class COGPTest(unittest.TestCase):
    def setUp(self):
        np.random.RandomState(0)
        self.X = np.random.randn(100, 3)
        self.X_aug = np.hstack([self.X, np.zeros(self.X.shape[0])[:,None]])
        self.Y = np.sum(self.X, axis=1) + np.random.randn(100)[:, None]
        self.Z = np.random.randn(5, 3)
        self.Z_aug = np.hstack([self.Z, np.zeros(self.Z.shape[0])[:,None]])

        self.shared = [gpflow.kernels.RBF(3, active_dims=list(range(3)))]
        self.tasks = [gpflow.kernels.RBF(3, active_dims=list(range(3)))]

        self.lik = gpflow.likelihoods.Gaussian()

        self.model = COGP(self.X_aug, self.Y, self.shared, self.tasks, self.lik, self.Z_aug)

    def test_constructor(self):
        self.assertEqual(self.model.num_tasks, 1)

if __name__ == "__main__":
    unittest.main()
