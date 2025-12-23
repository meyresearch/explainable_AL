import torch
import gpytorch


def train_gp_model(train_x, train_y, likelihood, model, epochs=50, lr=0.1, lr_decay=0.95):
    """Train a GP model with an optimizer and learning-rate scheduler.

    Parameters
    ----------
    train_x, train_y : torch.Tensor
        Training tensors.
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood associated with the GP model.
    model : gpytorch.models.ExactGP
        Model to train.
    epochs : int
        Number of optimization steps.
    lr : float
        Initial learning rate.
    lr_decay : float
        Multiplicative LR decay per epoch (scheduler gamma).

    Returns
    -------
    model, likelihood, list
        Trained model, likelihood, and list of recorded losses.
    """
    device = train_x.device
    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        losses.append(loss.item())
        if (i+1) % 10 == 0:
            print(f"Epoch {i+1}/{epochs} | Loss: {loss.item():.3f}")
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model, likelihood, losses
