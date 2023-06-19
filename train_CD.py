import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import *
import gc
import sys

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor, Size
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal, Uniform, Categorical
from torchviz import make_dot
import torchaudio as ta
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import LineProfiler

from itertools import islice
from pathlib import Path
from tqdm import tqdm

from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss, AMNPE, AMNPELoss
from lampe.nn import ResMLP
from lampe.utils import GDStep

from zuko.flows import NAF, UNAF, NSF, MAF, GMM, CNF

from dataloader import PosteriorDataset
from stat_tests import VecMMD, RBFKernel, PolynomialKernel
from robust_inference import RNPE
from nn_blocks import SoftClip, MeanSubtractionLayer, BaseConvBlock, CNNEmbedding, MultiInputEmbedding

runpath = Path("runs/CD_experiment1/")
runpath.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(runpath))




class KLDNPELoss(NPELoss):
    def __init__(self, estimator: nn.Module):
        super().__init__(estimator)

    def forward(self, theta: Tensor, x: Tensor, x_prime: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_p = self.estimator(theta, x, x_prime)

        return -log_p.mean()

class MMDNPELoss(nn.Module):
    def __init__(self, estimator: nn.Module, batch_size,
                 kernel: Any = RBFKernel(), X_samples=64):
        super().__init__()

        self.estimator = estimator
        self.batch_size = batch_size

        self._distance = VecMMD(kernel=kernel)
        self.X_samples = X_samples
        # self.Y_samples = None  # not used as the dataloader defines the number of theta trace samples

    def forward(self, theta: Tensor, x: Tensor, x_prime: Tensor) -> Tensor:
        # draw reparameterized samples from the flow
        rsamples = self.estimator.mmd_rsample(x, x_prime, shape=(self.X_samples, 1))
        rsamples1 = rsamples[torch.isfinite(rsamples).all(dim=-1).all(dim=-1)]
        rsamples2 = torch.movedim(rsamples1, 0, 1)

        # compute MMD over batch (vectorized) using kernel-trick
        distance = self._distance(rsamples2, theta)

        loss = torch.log10(distance).mean()

        return loss

class KLDMMDNPELoss(nn.Module):
    def __init__(self, estimator: nn.Module, batch_size, kernel: Any = RBFKernel()):
        super().__init__()

        self.calls = 0
        # self.alpha_scheduler = None
        self.alpha = 0.2
        self.gamma = 1. - 1.e-4
        self.burn_in = int(1 * 256)

        self.estimator = estimator
        self.batch_size = batch_size

        self.X_samples = 64
        # self.Y_samples = None  # not used as the dataloader defines the number of theta trace samples

        self.mmd_loss_module = MMDNPELoss(estimator=self.estimator, batch_size=batch_size,
                                          kernel=kernel, X_samples=self.X_samples).cuda()
        self.kld_loss_module = KLDNPELoss(estimator=self.estimator).cuda()

    def forward(self, theta: Tensor, x: Tensor, x_prime: Tensor) -> Tensor:

        batch_size = x.shape[0]

        kld_loss = self.kld_loss_module(theta[-int(batch_size / 2):],
                                        x[-int(batch_size / 2):],
                                        x_prime[-int(batch_size / 2):])

        combined_loss = kld_loss

        writer.add_scalar('KLD Loss',
                          kld_loss.detach(),
                          self.calls)


        if self.calls > self.burn_in:
            mmd_loss = self.mmd_loss_module(theta[:-int(batch_size / 2)].view(int(batch_size / 2), -1, 7),
                                            x[:-int(batch_size / 2)],
                                            x_prime[:-int(batch_size / 2)])

            combined_loss = self.alpha * kld_loss + (1. - self.alpha) * mmd_loss
            # self.alpha *= self.gamma

            writer.add_scalar('MMD Loss',
                              mmd_loss,
                              self.calls)

        self.calls += 1

        return combined_loss



class NPEWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = MultiInputEmbedding()

        self.npe = NPE(
            7,  # theta_dim
            64,  # x_dim
            transforms=3,
            build=UNAF,
            hidden_features=[64] * 3,
            activation=nn.ELU,
        )

    def forward(self, theta: Tensor, x: Tensor, x_prime: Tensor) -> Tensor:
        return self.npe(theta, self.embedding(x, x_prime))

    def flow(self, x: Tensor, x_prime: Tensor):  # -> Distribution
        return self.npe.flow(self.embedding(x, x_prime))

class RNPEWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = MultiInputEmbedding()

        self.npe = RNPE(
            7,  # theta_dim
            64,  # x_dim
            transforms=5,
            build=NAF,
            hidden_features=[64] * 3,
            activation=nn.ELU,
        )

    def forward(self, theta: Tensor, x: Tensor, x_prime: Tensor) -> Tensor:
        return self.npe(theta, self.embedding(x, x_prime))

    def mmd_rsample(self, x: Tensor, x_prime: Tensor, shape: Size = ()) -> Tensor:
        return self.npe.flow(self.embedding(x, x_prime)).rsample(shape)

    def flow(self, x: Tensor, x_prime: Tensor):  # -> Distribution
        return self.npe.flow(self.embedding(x, x_prime))

def trace_handler(p):
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

def train(i: int = 512):
    # Data
    input_type = "_full"

    batch_size = 512  # 2048  # 4096
    val_batch_size = 64  # int(np.clip(batch_size / 2**3, a_min=64, a_max=512))
    theta_sample_size = 64

    train_which = "training" # "validation" # "training"
    print(f"Loading training dataset...")
    trainset = PosteriorDataset(
        file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/{train_which}_dataset{input_type}.h5"),
        batch_size=batch_size,
        sample_size=theta_sample_size,
        shuffle=True,
    )
    trainset.to_memory()

    print(f"Loading validation dataset...", "THIS IS STILL THE TEST SET DURING TESTING")
    validset = PosteriorDataset(
        file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/validation_dataset{input_type}.h5"),
        batch_size=128,
        sample_size=theta_sample_size,
        shuffle=True,
    )
    validset.to_memory()

    print(f"Loading testing dataset...")
    testset = PosteriorDataset(
        file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/testing_dataset{input_type}.h5"),
        batch_size=128,
        sample_size=theta_sample_size,
        shuffle=True,
    )
    testset.to_memory()

    n_train_samples = len(trainset)
    loss_iters = int(2**np.floor(np.log2(n_train_samples / batch_size)) / 4)
    print(f"N samples: {n_train_samples}, with batches of {batch_size} for {loss_iters} iters per epoch.")


    # Training
    kernel = PolynomialKernel(degree=2, gamma=None, coef0=1).cuda()
    # kernel = RBFKernel().cuda()
    estimator = RNPEWithEmbedding().cuda()
    loss = KLDMMDNPELoss(estimator=estimator, batch_size=batch_size, kernel=kernel)  # AMNPELoss(estimator, mask_dist=Categorical(torch.tensor([0.5, 0.5]).cuda()))
    optimizer = optim.AdamW(estimator.parameters(),
                            lr=1e-3,
                            weight_decay=1e-2)
    step = GDStep(optimizer,
                  clip=100.0)

    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-10,
        patience=16,
        threshold=1e-3,
        threshold_mode='abs',
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Parameters: {count_parameters(estimator)}\n")

    def noisy(x: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.normal(mean=x[:, 0], std=x[:, 1]).reshape((-1, 1, 52)), x[:, 2]

    def noise_pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        x, x_prime = noisy(x)
        return loss(theta, x, x_prime)

    def clean_pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        return loss(theta, x[:, 0].reshape((-1, 1, 52)), x[:, 2])

    for epoch in tqdm(range(i), unit='epoch'):
        optimizer.zero_grad()
        estimator.zero_grad()
        loss.zero_grad()
        kernel.zero_grad()

        estimator.train()
        start = time.time()

        losses = torch.stack([
            step(noise_pipe(theta, x))
            for theta, x in islice(trainset.__iter_id_theta_trace_x__(), loss_iters)
        ])


        end = time.time()
        estimator.eval()


        with torch.no_grad():
            losses_val = torch.stack([
                clean_pipe(theta, x)
                for theta, x in islice(validset.__iter_id_theta_trace_x__(), 4)
            ])

        if epoch % 3 == 1:

            with torch.no_grad():
                losses_test = torch.stack([
                    clean_pipe(theta, x)
                    for theta, x in islice(testset.__iter_id_theta_trace_x__(), 2)
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
            writer.add_scalar('TO alpha',
                              loss.alpha,
                              epoch)

        scheduler.step(torch.nanmean(losses_val))

        if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

        if epoch % 8 == 7:
            torch.save(estimator.state_dict(), runpath / 'state.pth')

    writer.flush()
    writer.close()

    runpath.mkdir(parents=True, exist_ok=True)

    torch.save(estimator.state_dict(), runpath / 'state.pth')


if __name__ == '__main__':
    train()