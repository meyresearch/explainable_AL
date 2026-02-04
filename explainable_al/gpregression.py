import torch
import gpytorch


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto kernel compatible with the GP regression utilities in this package.

    The kernel implements a numerically-stable Tanimoto similarity for
    fingerprint-like vectors.
    """

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return self.covar_dist(x1, x2, diag=True, **params)
        else:
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            x1_dot_x2 = torch.matmul(x1 , x2.transpose(-1, -2))
            denominator = x1_norm + x2_norm.transpose(-1, -2) - x1_dot_x2
            return x1_dot_x2 / denominator.clamp(min=1e-8)


class GPRegressionModel(gpytorch.models.ExactGP):
    """GP regression model using the package's `TanimotoKernel` by default.

    This is a thin wrapper around `gpytorch.models.ExactGP`.
    """

    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel is None:
            self.covar_module = TanimotoKernel()
        else:
            self.covar_module = kernel.get_kernel()


    def forward(self, x):
        """Compute the GP prior for inputs `x` and return a MultivariateNormal."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x.add_jitter(1e-6))
