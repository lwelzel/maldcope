r"""Inference components such as estimators, training losses and MCMC samplers."""

# __all__ = [
#     'RNPE',
# ]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice
from torch import Tensor, BoolTensor, Size
from typing import *

from zuko.distributions import Distribution, DiagNormal, NormalizingFlow
from zuko.flows import FlowModule, MAF
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

from lampe.nn import MLP
from lampe.inference import NPE, NPELoss

class RNPE(NPE):
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        build: Callable[[int, int], FlowModule] = MAF,
        **kwargs,
    ):
        super().__init__(theta_dim, x_dim, build)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-density :math:`\log p_\phi(\theta | x)`, with shape :math:`(*,)`.
        """

        # print("RNPE", theta.shape, x.shape)

        theta, x = broadcast(theta, x, ignore=1)

        return self.flow(x).log_prob(theta)

    def rsample(self, x: Tensor, shape: Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            shape: The shape :math:`S` of the samples.

        Returns:
            The reparameterized samples :math:`\theta \sim p_\phi(\theta | x)`,
            with shape :math:`S + (*, D)`, while preserving the gradient information.
        """

        return self.flow(x).rsample(shape)

    def sample(self, x: Tensor, shape: Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            shape: The shape :math:`S` of the samples.

        Returns:
            The samples :math:`\theta \sim p_\phi(\theta | x)`,
            with shape :math:`S + (*, D)`.
        """

        with torch.no_grad():
            return self.flow(x).sample(shape)

