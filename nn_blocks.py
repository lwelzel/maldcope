import torch
import torch.nn as nn

from torch import Tensor

from lampe.inference import NPE, NPELoss, AMNPE, AMNPELoss
from lampe.nn import ResMLP


class SoftClip(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()

        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))


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


class CNNEmbedding(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.embedding = nn.Sequential(
            MeanSubtractionLayer(),
            BaseConvBlock(
                channels_in=1,
                channels_out=256,
                kernel_size=3,
                dilation=1,
            ),
            BaseConvBlock(
                channels_in=256,
                channels_out=128,
                kernel_size=5,
            ),
            BaseConvBlock(
                channels_in=128,
                channels_out=64,
                kernel_size=7,
            ),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.embedding(x)


class MultiInputEmbedding(nn.Module):
    def __init__(self, out_features=64):
        super().__init__()

        self.aux_features = 9
        self.out_features = out_features

        self.cnn_embedding = CNNEmbedding()

        self.spectrum_embedding = nn.Sequential(
            nn.Flatten(),
            ResMLP(
                in_features=52,
                out_features=128,
                hidden_features=[64] * 1 + [32] * 3 + [64] * 1,
                activation=nn.ELU,
                normalize=True,
            ),
        )


        self.aux_embedding = nn.Sequential(
            ResMLP(
                in_features=self.aux_features,
                out_features=8,
                hidden_features=[16] * 1 + [8] * 2,
                activation=nn.ELU,
                normalize=True,
            ),
        )

        self.embedding = nn.Sequential(
            ResMLP(
                in_features=128 + 8,
                out_features=self.out_features,
                hidden_features=[256] * 1 + [128] * 2 + [64] * 5, # + [64] * 5,
                activation=nn.ELU,
                normalize=True,
            ),
        )

        self.norm = nn.BatchNorm1d(num_features=out_features)

    def forward(self, x, x_prime):
        z0 = self.cnn_embedding(x)
        z1 = self.aux_embedding(x_prime[:, :self.aux_features])
        z2 = self.spectrum_embedding(x)
        z = torch.cat((z0 + z2, z1), dim=1)
        return self.embedding(z)
