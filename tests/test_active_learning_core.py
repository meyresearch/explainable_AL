
import unittest
import torch
import gpytorch
import numpy as np

from explainable_al.active_learning_core import (
    TanimotoKernel,
    GPRegressionModel,
    train_gp_model,
    ucb_selection,
    pi_selection,
    ei_selection,
)

class TestActiveLearningCore(unittest.TestCase):

    def setUp(self):
        self.fingerprints = [np.random.randint(0, 2, 128) for _ in range(20)]
        self.train_x = torch.tensor(np.array(self.fingerprints[:10])).float()
        self.train_y = torch.rand(10)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.kernel = TanimotoKernel()
        self.model = GPRegressionModel(self.train_x, self.train_y, self.likelihood, self.kernel)
        self.model, self.likelihood = train_gp_model(self.train_x, self.train_y, self.likelihood, self.model)

    def test_tanimoto_kernel(self):
        x1 = torch.tensor([[1, 1, 0, 0]], dtype=torch.float)
        x2 = torch.tensor([[1, 0, 1, 0]], dtype=torch.float)
        tanimoto_sim = self.kernel(x1, x2).evaluate().item()
        self.assertAlmostEqual(tanimoto_sim, 1/3)

    def test_gp_regression_model(self):
        self.assertIsInstance(self.model, GPRegressionModel)

    def test_train_gp_model(self):
        # The model is already trained in setUp
        self.assertTrue(self.model.training is False)

    def test_ucb_selection(self):
        indices = ucb_selection(self.fingerprints, self.model, self.likelihood, 5, 1, 1, list(range(10)))
        self.assertEqual(len(indices), 5)
        self.assertTrue(all(i >= 10 for i in indices))

    def test_pi_selection(self):
        indices = pi_selection(self.fingerprints, self.model, self.likelihood, 5, list(range(10)), 0.5)
        self.assertEqual(len(indices), 5)
        self.assertTrue(all(i >= 10 for i in indices))

    def test_ei_selection(self):
        indices = ei_selection(self.fingerprints, self.model, self.likelihood, 5, list(range(10)), 0.5)
        self.assertEqual(len(indices), 5)
        self.assertTrue(all(i >= 10 for i in indices))

if __name__ == '__main__':
    unittest.main()
