import torch
from torch import nn
from typing import *
from ddks.methods import ddKS

# Author: @yiftachbeer, yiftachbeer
class RBFKernel(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None,
                 device="cuda", **kwargs):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X, **kwargs):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class PolynomialKernel(nn.Module):
    def __init__(self, degree=2, gamma=None, coef0=1, **kwargs):
        super().__init__()
        self.degree = degree
        self.degrees = torch.arange(1, self.degree+1, device="cuda")
        # self.gamma = gamma  # choose to always choose gamma to normalize of the nuber of samples
        self.coef0 = coef0

    def forward(self, X, **kwargs):

        self.gamma = 1.0 / X.size(1)

        dot_product = torch.mm(X, X.t()) * self.gamma + self.coef0

        return torch.sum(dot_product[..., None].tile((1, 1, len(self.degrees))).pow(self.degrees[None, None, ...]), dim=-1) #  sum([dot_product**d for d in range(1, self.degree+1)]) #  # sum([dot_product**d for d in range(1, self.degree+1)])

class MMDLoss(nn.Module):
    def __init__(self, kernel=RBFKernel(), **kwargs):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y, **kwargs):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        Y_size = Y.shape[0]

        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        mmd = XX - 2 * XY + YY

        # Normalize by the geometric mean of the number of samples in X and Y
        return mmd / (X_size * Y_size) ** 0.5


class VecMMD(nn.Module):
    def __init__(self, kernel: Any = RBFKernel()):
        super().__init__()
        # non batched forward call
        self._mmd_loss = MMDLoss(kernel)
        # initialize the downstream classes and functions
        self._mmd_loss.forward(torch.rand((1, 1), device="cuda"), torch.rand((1, 1), device="cuda"))
        # vectorize loss calc
        self.mmd_loss = torch.vmap(self._mmd_loss.forward, in_dims=0, out_dims=0)

    def forward(self, X, Y):
        distance = self.mmd_loss(X, Y)

        return distance


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from chainconsumer import ChainConsumer

    polykernel = PolynomialKernel(degree=5, gamma=None, coef0=1)

    distance_metric = VecMMD(polykernel)

    offset = 0.00
    broadening = 0.0  # -0.9

    _offset = torch.arange(7, device="cuda")

    ns = 256
    n_oversample = 2

    y = torch.distributions.normal.Normal(torch.ones(7, device="cuda") * _offset, torch.ones(7, device="cuda")).sample((16, ns * n_oversample, ))  # 256 sample for the true values
    # y = torch.normal(0. + offset, 1. + broadening, (16, ns, 7))  # 64 samples for the predicted values
    # y = torch.distributions.laplace.Laplace(01. + offset, 1. + broadening).rsample((16, ns, 7))  # 64 samples for the predicted values
    # y = torch.exp(torch.normal(0. + offset, 1. + broadening, (16, ns, 7)))  # 64 samples for the predicted values

    mix = torch.distributions.Categorical(torch.ones(2, device="cuda" ))

    offset = torch.zeros(2, 7, device="cuda")

    offset[0, 0] = -10

    comp = torch.distributions.Independent(torch.distributions.Normal(torch.ones(2, 7, device="cuda") * _offset + offset, torch.ones(2, 7, device="cuda")), 1)
    gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp).sample((16, ns, ))

    print(gmm.shape)

    x = gmm

    print(x.device, y.device)
    print(x.shape, y.shape)

    loss_xy = distance_metric(x, y)

    print(loss_xy)
    print(loss_xy.mean())
    print(torch.log10(loss_xy).mean())

    c = ChainConsumer()
    c.add_chain(x[0].cpu().numpy(), name="Test")
    c.add_chain(y[0].cpu().numpy(), name="Truth")
    c.configure(smooth=0)
    fig = c.plotter.plot()
    fig.suptitle(f"Loss: {loss_xy[0].cpu().numpy():.3f}, log loss: {torch.log10(loss_xy)[0].cpu().numpy():.3f}")
    plt.show()

    raise NotImplementedError

    c = ChainConsumer()
    c.add_chain(x[0].cpu().numpy())
    c.add_chain(y[0].cpu().numpy())
    c.configure(smooth=0)
    fig = c.plotter.plot()
    plt.show()

    n = 100
    x_space = np.linspace(0, 1, n)
    losses = np.zeros(n)

    for i, offset in enumerate(x_space):
        x = torch.normal(0, 1,      (16, 1, 8))  # 265 sample for the true values
        y = torch.normal(offset, 1e-4, (16, 64, 8))  # 16 samples for the predicted values


        loss_xy = loss(x, y)

        losses[i] = loss_xy.mean()

    plt.plot(x_space, losses)
    plt.show()



