import torch
from torch import nn
from ddks.methods import ddKS

# Author: @yiftachbeer, yiftachbeer
class RBF(nn.Module):
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


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF(), **kwargs):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y, **kwargs):
        K = self.kernel(torch.vstack([X.cuda(), Y.cuda()]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class VecMMDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # non batched forward call
        self._mmd_loss = MMDLoss()
        # initialize the downstream classes and functions
        self._mmd_loss.forward(torch.rand((1, 1)), torch.rand((1, 1)))
        # vectorize loss calc
        self.mmd_loss = torch.vmap(self._mmd_loss.forward, in_dims=0, out_dims=0)


    def forward(self, X, Y, **kwargs):

        neg_log_prob = torch.log10(self.mmd_loss(X, Y)).nanmean()

        return neg_log_prob


if __name__ == "__main__":
    loss = VecMMDLoss()


    x = torch.normal(0, 1, (16, 2048, 8)) # 1 sample for the true values
    y = torch.normal(1., 0.9, (16, 256, 8)) + 0.5 * torch.normal(0, 1, (16, 1, 8)) # 16 samples for the predicted values

    print(x.device, y.device)
    print(x.shape, y.shape)

    loss_xy = loss(x, y)

    print(loss_xy)

    import matplotlib.pyplot as plt
    import numpy as np
    from chainconsumer import ChainConsumer

    c = ChainConsumer()
    c.add_chain(x[0].cpu().numpy())
    c.add_chain(y[0].cpu().numpy())
    c.configure(smooth=0)
    fig = c.plotter.plot()
    plt.show()

    n = 100
    x_space = np.linspace(0, 4, n)
    losses = np.zeros(n)

    import corner as corner

    for i, offset in enumerate(x_space):
        x = torch.normal(0, 1,      (16, 265, 8))  # 265 sample for the true values
        y = torch.normal(0, 1e-4 + offset, (16, 16, 8))  # 16 samples for the predicted values


        loss_xy = loss(x, y)

        losses[i] = loss_xy

    plt.plot(x_space, losses)
    # plt.xscale("log")
    plt.show()



