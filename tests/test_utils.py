
import unittest
import torch
import gpytorch
import numpy as np
from rdkit.DataStructs import ExplicitBitVect

from explainable_al.active_learning_core import GPRegressionModel, TanimotoKernel
from explainable_al.metrics_plots import calculate_metrics
from explainable_al import active_learning_core as alc_utils

from explainable_al import active_learning_core
from explainable_al import utils as pkg_utils

class TestUtils(unittest.TestCase):

    def test_get_ecfp_fingerprints(self):
        smiles_list = ['CCO', 'CCN']
        fingerprints = pkg_utils.get_ecfp_fingerprints(smiles_list)
        self.assertEqual(len(fingerprints), 2)
        self.assertIsInstance(fingerprints[0], ExplicitBitVect)

    def test_get_maccs_keys(self):
        smiles_list = ['CCO', 'CCN']
        fingerprints = pkg_utils.get_maccs_keys(smiles_list)
        self.assertEqual(len(fingerprints), 2)
        self.assertIsInstance(fingerprints[0], ExplicitBitVect)

    def test_calculate_metrics(self):
        # Create a dummy model and data
        train_x = torch.rand(10, 128)
        train_y = torch.rand(10)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, TanimotoKernel())
        test_x = torch.rand(5, 128)
        test_y = torch.rand(5)

        r2, spearman = calculate_metrics(model, likelihood, test_x, test_y)
        self.assertIsInstance(r2, float)
        self.assertIsInstance(spearman, float)

if __name__ == '__main__':
    unittest.main()
