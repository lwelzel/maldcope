import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Uniform, Categorical
from torchviz import make_dot
import torchaudio as ta

from itertools import islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm

from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss, AMNPE, AMNPELoss
from lampe.nn import ResMLP
from lampe.utils import GDStep

from zuko.flows import NAF, UNAF, NSF, MAF, GMM, CNF

from stat_tests import MMDLoss

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sbiear_experiment1')




class SoftClip(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()

        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))


class CustomNPELoss(NPELoss):
    def __init__(self, estimator: nn.Module):
        super().__init__(estimator)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_p = self.estimator(theta, x)

        return -log_p.mean()


class DivergenceNPELoss(NPELoss):
    def __init__(self, estimator: nn.Module, n_samples=2**3):
        super().__init__(estimator)

        self.mmd_loss = MMDLoss().cuda()
        self.n_samples = n_samples

    def test_MMD(self, theta: Tensor, x: Tensor) -> Tensor:
        z = self.estimator.flow(x).rsample((self.n_samples,))
        z = z.swapaxes(0, 1).cuda()
        _theta = torch.tile(theta.reshape((-1, 1, 7)), (1, self.n_samples, 1)).cuda()

        print(_theta.shape, z.shape)

        mmd = self.mmd_loss.forward(z, _theta)

        print(mmd.shape)
        print(mmd)

        return mmd.mean()


    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        mmd = self.test_MMD(theta, x)

        log_p = self.estimator(theta, x)

        return -log_p.mean() + mmd


class AlphaNPELoss(NPELoss):
    def __init__(self, estimator: nn.Module, alpha=0.5):
        super().__init__(estimator)

        self.alpha = alpha

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_p = self.estimator(theta, x)

        return -log_p.mean()


class MeanSubtractionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - torch.mean(x, dim=-1).reshape(-1, 1, 1)


class BaseConvBlock(nn.Module):
    def __init__(self, channels_in=32, channels_out=32, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=nn.ELU(),
                 pooling=nn.MaxPool1d, pooling_kernel_size=2,
                 normalization=nn.BatchNorm1d,
                 ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels_in,
                      channels_out,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias),
            activation,
            pooling(
                kernel_size=pooling_kernel_size,
                stride=pooling_kernel_size),
            # nn.GroupNorm(num_groups=32, num_channels=channels_out, eps=1e-8)
            normalization(channels_out),
        )

    def forward(self, x):
        return self.block(x)


class NPEWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Sequential(
            # SoftClip(1000.0),
            MeanSubtractionLayer(),
            BaseConvBlock(
                channels_in=1,
                channels_out=128,
                kernel_size=3,
                dilation=1,
            ),
            BaseConvBlock(
                channels_in=128,
                channels_out=64,
                kernel_size=3,
            ),
            BaseConvBlock(
                channels_in=64,
                channels_out=64,
                kernel_size=3,
            ),
            nn.Flatten(),
            ResMLP(
                in_features=256,
                out_features=64,
                hidden_features=[512] * 1 + [256] * 2 + [128] * 5 + [64] * 5,
                activation=nn.ELU,
                normalize=True,
            ),
            # nn.BatchNorm1d(16)

        )

        self.npe = NPE(
            7,  # theta_dim
            64,  # x_dim
            transforms=3,
            build=UNAF,
            hidden_features=[64] * 3,
            activation=nn.ELU,
        )

        self.decoder = nn.Sequential(

        )

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.npe(theta, self.embedding(x))

    def flow(self, x: Tensor):  # -> Distribution
        return self.npe.flow(self.embedding(x))


def train(i: int = 512):
    # Data

    batch_size = 2048  # 2048  # 4096
    val_batch_size = 128  # int(np.clip(batch_size / 2**3, a_min=64, a_max=512))


    trainset = H5Dataset("/home/lwelzel/Documents/git/maldcope/data/TrainingData/training_dataset.h5",
                         batch_size=batch_size,
                         shuffle=True,
                         ).to_memory()

    n_train_samples = len(trainset)
    loss_iters = int(2**np.floor(np.log2(n_train_samples / batch_size)))
    print(f"N samples: {n_train_samples}, with batches of {batch_size} for {loss_iters} iters per epoch.")

    validset = H5Dataset("/home/lwelzel/Documents/git/maldcope/data/TrainingData/validation_dataset.h5",
                         batch_size=val_batch_size,
                         shuffle=True
                         ).to_memory()

    testset = H5Dataset("/home/lwelzel/Documents/git/maldcope/data/TrainingData/testing_dataset.h5",
                         batch_size=val_batch_size,
                         shuffle=True
                         ).to_memory()

    # Training
    estimator = NPEWithEmbedding().cuda()
    loss = CustomNPELoss(estimator)  # AMNPELoss(estimator, mask_dist=Categorical(torch.tensor([0.5, 0.5]).cuda()))
    optimizer = optim.AdamW(estimator.parameters(),
                            lr=1e-2,
                            weight_decay=1e-1)
    step = GDStep(optimizer,
                  clip=1.0)
    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-6,
        patience=16,
        threshold=1e-3,
        threshold_mode='abs',
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Parameters: {count_parameters(estimator)}\n")

    def noisy(x: Tensor) -> Tensor:
        return torch.normal(mean=x[:, 0], std=x[:, 1]).reshape((-1, 1, 52))
        # return x[:, 0].reshape((-1, 1, 52))


    def noise_pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        x = noisy(x)
        return loss(theta, x)

    def clean_pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        return loss(theta, x[:, 0].reshape((-1, 1, 52)))

    for epoch in tqdm(range(i), unit='epoch'):
        estimator.train()
        start = time.time()

        losses = torch.stack([
            step(noise_pipe(theta, x))
            for theta, x in islice(trainset, loss_iters)
        ])

        end = time.time()
        estimator.eval()

        with torch.no_grad():
            losses_val = torch.stack([
                clean_pipe(theta, x)
                for theta, x in islice(validset, 4)
            ])

        if epoch % 10 == 1:

            with torch.no_grad():
                losses_test = torch.stack([
                    clean_pipe(theta, x)
                    for theta, x in islice(testset, 2)
                ])

            train_loss = torch.nanmean(losses).cpu()
            train_loss.numpy()

            val_loss = torch.nanmean(losses_val).cpu()
            val_loss.numpy()

            test_loss = torch.nanmean(losses_test).cpu()
            test_loss.numpy()

            writer.add_scalar('Loss',
                              train_loss,
                              epoch)
            writer.add_scalar('Validation Loss',
                              val_loss,
                              epoch)
            writer.add_scalar('Test Loss',
                              test_loss,
                              epoch)
            writer.add_scalar('Learning Rate',
                              optimizer.param_groups[0]['lr'],
                              epoch)
            writer.add_scalar('NANs',
                              (torch.sum(~torch.isfinite(losses)).cpu()
                              + torch.sum(~torch.isfinite(losses_val)).cpu()
                              + torch.sum(~torch.isfinite(losses_test)).cpu()).numpy(),
                              epoch)
            writer.add_scalar('speed',
                              len(losses) / (end - start),
                              epoch)



        scheduler.step(torch.nanmean(losses_val))

        if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    writer.flush()
    writer.close()

    runpath = Path("runs/sbiear_experiment1")
    runpath.mkdir(parents=True, exist_ok=True)

    torch.save(estimator.state_dict(), runpath / 'state.pth')


if __name__ == '__main__':
    train()