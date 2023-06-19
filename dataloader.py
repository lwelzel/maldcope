from matplotlib import pyplot as plt
import numpy as np
import os
from os.path import isdir, isfile
import pandas as pd

from numpy.random import default_rng
from pathlib import Path
import scipy.ndimage as ndimage
import torch
import lampe
import zuko
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import WeightedRandomSampler, BatchSampler
from itertools import islice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import h5py
import numpy as np

from bisect import bisect
from numpy import ndarray as Array
from pathlib import Path
from torch import Tensor, Size
from torch.distributions import Distribution
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from typing import *
from corner import corner
from sklearn.preprocessing import PowerTransformer
import dill
from uncertainties import unumpy

from lampe.data import H5Dataset, JointDataset

class PosteriorDataset(Dataset):
    def __init__(
            self,
            file,
            batch_size: int = None,
            sample_size: int = 64,
            shuffle: bool = False,
    ):
        super().__init__()

        self.file = file
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.shuffle = shuffle

        self.pairs = []

        self.names = None
        self.valid_traces = None
        self.traces = None
        self.weights = None
        self.traces_samples = None
        self.fm_theta = None
        self.x = None

        self.n_trace_pairs = None

        self.batched_wg_rand_sampler = torch.vmap(self.wg_rand_samp, in_dims=0, out_dims=0, randomness="different")

    def __len__(self) -> int:
        return len(self.pairs)

    # def __getitem__(self, i: Union[int, slice]) -> Tuple[str, Tensor, Tensor, Tensor]:
    #     sample_idx = WeightedRandomSampler(self.weights[i], self.sample_size, replacement=False)
    #     return self.names[i], self.traces[i][sample_idx], self.fm_theta[i], self.x[i]
    #
    # def __iter__(self) -> Iterator[Tuple[str, Tensor, Tensor, Tensor]]:
    #     if self.shuffle:
    #         order = torch.randperm(len(self))
    #
    #         if self.batch_size is None:
    #             return (self[i] for i in order)
    #         else:
    #             return (self[i] for i in order.split(self.batch_size))
    #     else:
    #         order = torch.arange(len(self))
    #         if self.batch_size is None:
    #             return (self[i] for i in order)
    #         else:
    #             return (self[i] for i in order.split(self.batch_size))

    def __getitem__(self, i: Union[int, slice]) -> Tuple[str, bool, Tensor, Tensor, Tensor, Tensor]:
        return self.pairs[i]

    def __iter__(self) -> Iterator[Tuple[str, bool, Tensor, Tensor, Tensor, Tensor]]:
        if self.shuffle:
            order = torch.randperm(len(self))

            if self.batch_size is None:
                return (self[i] for i in order)
            else:
                return (self[i] for i in order.split(self.batch_size))
        else:
            order = torch.arange(len(self))
            if self.batch_size is None:
                return (self[i] for i in order)
            else:
                return (self[i] for i in order.split(self.batch_size))

    # TODO: all below improve using BatchSampler
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
    def __get_theta_x__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        return self.fm_theta[i], self.x[i]
    def __iter_theta_x__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        if self.shuffle:
            order = torch.randperm(len(self))

            if self.batch_size is None:
                return (self.__get_theta_x__(i) for i in order)
            else:
                return (self.__get_theta_x__(i) for i in order.split(self.batch_size)[:-1])
        else:
            order = torch.arange(len(self))
            if self.batch_size is None:
                return (self.__get_theta_x__(i) for i in order)
            else:
                return (self.__get_theta_x__(i) for i in order.split(self.batch_size)[:-1])
    def wg_rand_samp(self, weights):
        return torch.tensor(WeightedRandomSampler(weights, num_samples=self.sample_size, replacement=False))
    # def __get_trace_x__(self, i):
    #     print(i)
    #     print(self.weights[i].shape)
    #     sample_idx = self.batched_wg_rand_sampler(self.weights[i])
    #     return self.traces[i][sample_idx], self.x[i]

    # def __get_trace_x__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
    #     sample_idx = list(WeightedRandomSampler(self.weights[i], (self.sample_size), replacement=False)
    #     return self.traces[i][sample_idx], self.x[i]

    def __get_trace_x__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        sample_idx = torch.tensor([list(WeightedRandomSampler(weights, self.sample_size, replacement=False))
                      for weights in self.weights[i]]).cuda()

        traces = torch.gather(self.traces[i], 1, sample_idx[..., None].tile((1, 1, 7)))  # JFC, but otherwise I get CUDA errors

        return traces, self._x[i]

    def __iter_trace_x__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        if self.shuffle:
            order = torch.randperm(self.n_trace_pairs)

            if self.batch_size is None:
                return (self.__get_trace_x__(i) for i in order)
            else:
                return (self.__get_trace_x__(i) for i in order.split(self.batch_size)[:-1])
        else:
            order = torch.arange(len(self))
            if self.batch_size is None:
                return (self.__get_trace_x__(i) for i in order)
            else:
                return (self.__get_trace_x__(i) for i in order.split(self.batch_size)[:-1])

    def __get_ttrace_x__(self, _i: Union[int, slice]) -> Tuple[Tensor, Tensor]:

        # print(_i)

        i = _i[:, 0].flatten()
        it = _i[:, 1].flatten()

        # print(i.shape)
        # print(it.shape)

        sample_idx = torch.tensor([list(WeightedRandomSampler(weights, self.sample_size, replacement=False))
                      for weights in self.weights[i]]).cuda()

        traces = torch.gather(self.traces[i], 1, sample_idx[..., None].tile((1, 1, 7))).view(-1, 7)  # JFC, but otherwise I get CUDA errors
        # traces = torch.gather(self.traces[i], 1, sample_idx[..., None].tile((1, 1, 7)))  # JFC, but otherwise I get CUDA errors
        # traces = traces.mean(dim=1)

        theta, x = self.__get_theta_x__(it)

        # print("theta x", theta.shape, x.shape)
        # print("trace x", traces.shape, self._x[i].tile(self.sample_size, 1, 1).shape)

        ttrace = torch.cat((traces, theta))
        tx = torch.cat((self._x[i].tile(self.sample_size, 1, 1), x))  # is samples are not collapsed
        # tx = torch.cat((self._x[i], x)) # if samples are collapsed

        # print(ttrace.shape, tx.shape)

        return ttrace, tx

    def __iter_ttrace_x__(self) -> Iterator[Tuple[Tensor, Tensor]]:

        split_batch_size = int(self.batch_size / 2)

        if self.shuffle:
            order = torch.randperm(self.n_trace_pairs)
            theta_order = torch.randperm(len(self))[:self.n_trace_pairs]

            stack = torch.stack((order, theta_order), dim=1)
            # print(stack.shape)

            if self.batch_size is None:
                return (self.__get_ttrace_x__(i) for i in stack)
            else:
                return (self.__get_ttrace_x__(i) for i in stack.split(split_batch_size)[:-1])
        else:
            order = torch.arange(len(self))
            theta_order = torch.arange(len(self))[:self.n_trace_pairs]

            stack = torch.stack((order, theta_order), dim=1)

            if self.batch_size is None:
                return (self.__get_ttrace_x__(i) for i in stack)
            else:
                return (self.__get_ttrace_x__(i) for i in stack.split(split_batch_size)[:-1])

    def __get_id_theta_trace_x__(self, _i: Union[int, slice]) -> Tuple[Tensor, Tensor]:

        # print(_i)

        i = _i[:, 0].flatten()
        it = _i[:, 1].flatten()

        # print(i.shape)
        # print(it.shape)

        sample_idx = torch.tensor([list(WeightedRandomSampler(weights, self.sample_size, replacement=False))
                      for weights in self.weights[i]]).cuda()

        traces = torch.gather(self.traces[i], 1, sample_idx[..., None].tile((1, 1, 7))).view(-1, 7)  # JFC, but otherwise I get CUDA errors
        # traces = torch.gather(self.traces[i], 1, sample_idx[..., None].tile((1, 1, 7)))  # JFC, but otherwise I get CUDA errors
        # traces = traces.mean(dim=1)

        theta, x = self.__get_theta_x__(it)

        # print("theta x", theta.shape, x.shape)
        # print("trace x", traces.shape, self._x[i].shape)

        ttrace = torch.cat((traces, theta)).cuda()
        # print("ttrace", ttrace.shape)
        # tx = torch.cat((self._x[i].tile(self.sample_size, 1, 1), x))  # is samples are not collapsed
        tx = torch.cat((self._x[i], x)).cuda()  # if samples are collapsed
        # print("tx", tx.shape)

        # print(ttrace.shape, tx.shape)

        return ttrace, tx

    def __iter_id_theta_trace_x__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        # print("iter start:")
        # print(self.fm_theta.device)
        # print(self.x.device)
        # print(self.traces.device)
        # print(self.weights.device)
        # print(self._fm_theta.device)
        # print(self._x.device)
        # print("==========")

        split_batch_size = int(self.batch_size / 2)

        if self.shuffle:
            order = torch.randperm(self.n_trace_pairs)
            theta_order = torch.randperm(len(self))[:self.n_trace_pairs]

            stack = torch.stack((order, theta_order), dim=1)
            # print(stack.shape)

            if self.batch_size is None:
                return (self.__get_id_theta_trace_x__(i) for i in stack)
            else:
                return (self.__get_id_theta_trace_x__(i) for i in stack.split(split_batch_size)[:-1])
        else:
            order = torch.arange(len(self))
            theta_order = torch.arange(len(self))[:self.n_trace_pairs]

            stack = torch.stack((order, theta_order), dim=1)

            if self.batch_size is None:
                return (self.__get_id_theta_trace_x__(i) for i in stack)
            else:
                return (self.__get_id_theta_trace_x__(i) for i in stack.split(split_batch_size)[:-1])


    def __get_trace_theta_x__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor, Tensor]:
        sample_idx = torch.tensor([list(WeightedRandomSampler(weights, self.sample_size, replacement=False))
                      for weights in self.weights[i]]).cuda()

        traces = torch.gather(self.traces[i], 1, sample_idx[..., None].tile((1, 1, 7)))  # JFC, but otherwise I get CUDA errors

        return traces, self._fm_theta[i], self._x[i]


    def __iter_trace_theta_x__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        if self.shuffle:
            order = torch.randperm(self.n_trace_pairs)

            if self.batch_size is None:
                return (self.__get_trace_theta_x__(i) for i in order)
            else:
                return (self.__get_trace_theta_x__(i) for i in order.split(self.batch_size)[:-1])
        else:
            order = torch.arange(len(self))
            if self.batch_size is None:
                return (self.__get_trace_theta_x__(i) for i in order)
            else:
                return (self.__get_trace_theta_x__(i) for i in order.split(self.batch_size)[:-1])

    def _load_to_memory(self,):
        self.file = h5py.File(Path(self.file), mode='r')
        self.batch_size = self.batch_size
        self.shuffle = self.shuffle

        keys = self.file.keys()

        self.names = keys

        self.pairs = []

        for name in tqdm(keys, total=len(keys), unit=' pairs from h5 file'):

            self.pairs.append((
                name,
                bool(self.file[name]["valid_trace"][()]),
                torch.tensor(self.file[name]["trace"][()], device=device),
                torch.tensor(self.file[name]["weights"][()], device=device),
                torch.tensor(self.file[name]["fm_theta"][()], device=device),
                torch.tensor(self.file[name]["x"][()], device=device)
            ))

        self.valid_traces = torch.tensor([
            p[1] for p in self.pairs
        ], device=device)

        self.traces_samples = torch.tensor([
            len(p[3]) if p[1] else 0
            for p in self.pairs
        ], device=device)

    def load_contiguous_theta_x(self):
        if self.names is None:
            self._load_to_memory()

        n_pairs = len(self.traces_samples)
        ndim = len(self.pairs[0][4])
        x_shape = self.pairs[0][5].shape

        self.fm_theta = torch.empty(size=(n_pairs, ndim), dtype=torch.float, device=device)
        self.x =    torch.empty(size=(n_pairs, *x_shape), dtype=torch.float, device=device)

        for i, n_samples in tqdm(enumerate(self.traces_samples),
                                 total=len(self.pairs), unit=' theta-x pairs to contiguous memory'):
            self.fm_theta[i] = self.pairs[i][4]
            self.x[i] = self.pairs[i][5]

    def load_contiguous_trace_x(self):
        if self.names is None:
            self._load_to_memory()

        valid_idxs = torch.argwhere(self.valid_traces)
        max_sample_size = torch.min(self.traces_samples[valid_idxs])

        # print(torch.quantile(self.traces_samples[valid_idxs].to(torch.double),
        #                      q=torch.tensor([0.01, 0.04, 0.5, 0.96, 0.99],
        #                                     dtype=torch.double, device=device)))
        # print(f"Smallest number of available samples: {max_sample_size}")

        assert self.sample_size <= max_sample_size, f"The samples to be drawn in each trace must be less than the " \
                                                    f"smallest number of available samples. The tensors are padded with zeros.\n" \
                                                    f"Smallest number of available samples: {max_sample_size}"

        max_samples = 2048
        self.n_trace_pairs = torch.sum(self.valid_traces)

        self.names = self.file.keys()
        ndim = len(self.pairs[0][4])
        x_shape = self.pairs[0][5].shape

        self.traces =   torch.zeros(size=(self.n_trace_pairs, max_samples, ndim), dtype=torch.float, device=device)
        self.weights =  torch.zeros(size=(self.n_trace_pairs, max_samples), dtype=torch.float, device=device)
        self._fm_theta = torch.empty(size=(self.n_trace_pairs, ndim), dtype=torch.float, device=device)
        self._x =    torch.empty(size=(self.n_trace_pairs, *x_shape), dtype=torch.float, device=device)

        for i, (idx) in tqdm(enumerate(valid_idxs),
                             total=len(valid_idxs), unit=' trace-x pairs to contiguous memory'):

            pair = self.pairs[idx]
            n_samples = self.traces_samples[idx]

            trace_n_samples = torch.minimum(torch.tensor(max_samples), n_samples)

            sample_idx = list(WeightedRandomSampler(pair[3], int(trace_n_samples), replacement=False))

            self.traces[i, :trace_n_samples] =    pair[2][sample_idx]
            self.weights[i, :trace_n_samples] =   pair[3][sample_idx]
            self._fm_theta[i] = pair[4]
            self._x[i] =        pair[5]

    def to_memory(self):
        self._load_to_memory()

        self.load_contiguous_theta_x()

        self.load_contiguous_trace_x()

    @staticmethod
    def store(
            pairs: List[Tuple[str, bool, Array, Array, Array, Array]],
            file: Union[str, Path],
            overwrite: bool = False,
            dtype: np.dtype = np.float32,
            **meta,
    ) -> None:

        # Pairs
        pair_iter = iter(pairs)

        # File
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(file, 'w' if overwrite else 'w-') as f:
            ## Attributes
            f.attrs.update(meta)

            ## Store
            for i in tqdm(range(len(pairs)), total=len(pairs), unit='pair'):
                ## Datasets
                name, valid_trace, trace, weights, fm_theta, x = next(pair_iter)

                grp = f.create_group(str(name))
                grp.create_dataset(name="valid_trace", data=valid_trace, dtype=bool)
                grp.create_dataset(name="trace", data=trace, dtype=dtype)
                grp.create_dataset(name="weights", data=weights, dtype=dtype)
                grp.create_dataset(name="fm_theta", data=fm_theta, dtype=dtype)
                grp.create_dataset(name="x", data=x, dtype=dtype)

        print(f"\nFinished saving data file: {str(file.name)}\n")

def rescale_output(input, forward=False):

    input = torch.tensor(input)

    device = str(input.device)
    input = input.clone().detach().cuda()

    scale = torch.tensor([
        1.,    # planet_radius
        1.e-3, # planet_temp
        1.,    # log_H2O
        1.,    # log_CO2
        1.,    # log_CO
        1.,    # log_CH4
        1.,    # log_NH3
    ], device="cuda")

    shift = torch.tensor([
        - 0.5,   # planet_radius
        - 1000., # planet_temp
        + 6.,    # log_H2O
        + 7.,    # log_CO2
        + 5.,    # log_CO
        + 6.,    # log_CH4
        + 7.     # log_NH3
    ], device="cuda")

    out = None
    if forward:
        out = (input + shift) * scale
    elif ~forward:
        out = (input / scale) - shift
    else:
        raise NotImplementedError

    if "cuda" not in device:
        out.cpu()

    return out.cpu().numpy()

def rescale_aux_input(df):
    df['star_distance'] =          np.log10(df['star_distance'])          - 3.
    df['star_mass_kg'] =           np.log10(df['star_mass_kg'])           - 30.3
    df['star_radius_m'] =          np.log10(df['star_radius_m'])          - 8.9
    df['star_temperature'] =       np.log(df['star_temperature'])         - 8.6
    df['planet_mass_kg'] =         np.log10(df['planet_mass_kg'])         - 26.1
    df['planet_orbital_period'] =  np.log10(df['planet_orbital_period'])  - 1.
    df['planet_distance'] =        np.log10(df['planet_distance'])        + 1
    df['planet_surface_gravity'] = np.log10(df['planet_surface_gravity']) - 1.
    return df

def rescale_fm_theta_input(df):
    df['planet_radius'] =   df['planet_radius'] - 0.5
    df['planet_temp'] =     (df['planet_temp'] - 1000.) / 1000
    df['log_H2O'] =         df['log_H2O'] + 6.
    df['log_CO2'] =         df['log_CO2'] + 7.
    df['log_CO'] =          df['log_CO'] + 5.
    df['log_CH4'] =         df['log_CH4'] + 6.
    df['log_NH3'] =         df['log_NH3'] + 7.
    return df

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def zca_whiten(X):
    ZCAMatrix = zca_whitening_matrix(X)
    return np.dot(ZCAMatrix, X)

def plot_random_spectra(spectra, noise):

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 6))

    wlgrid = np.array([0.55, 0.7, 0.95, 1.156375, 1.27490344,
                       1.40558104, 1.5496531, 1.70849254, 1.88361302, 1.9695975,
                       2.00918641, 2.04957106, 2.09076743, 2.13279186, 2.17566098,
                       2.21939176, 2.26400154, 2.30950797, 2.35592908, 2.40328325,
                       2.45158925, 2.50086619, 2.5511336, 2.60241139, 2.65471985,
                       2.70807972, 2.76251213, 2.81803862, 2.8746812, 2.93246229,
                       2.99140478, 3.05153202, 3.11286781, 3.17543645, 3.23926272,
                       3.30437191, 3.37078978, 3.43854266, 3.50765736, 3.57816128,
                       3.65008232, 3.72344897, 4.03216667, 4.30545796, 4.59727234,
                       4.90886524, 5.24157722, 5.59683967, 5.97618103, 6.3812333,
                       6.81373911, 7.2755592])

    rng = np.random.default_rng()
    for i in rng.integers(0, len(spectra), 5):
        ax.errorbar(wlgrid,
                    spectra[i],  # - np.mean(mean_spectrum),
                    yerr=noise[i],
                    fmt="D", markersize=4, alpha=0.9)

    ax.set_xlabel(r"$\lambda$ [$\mu m$]")
    ax.set_ylabel(r"Transit Depth [-]")
    ax.set_title(f"Mean Spectrum")

    plt.show()

def make_full_pairs(spec_data, aux_data, trace_data, FM_parameters, weight_threshold=1e-8):
    # prepare aux
    aux_data_names = list(['star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature', 'planet_mass_kg',
                      'planet_orbital_period', 'planet_distance', 'planet_surface_gravity'])
    theta_names = list(['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3'])

    aux_arr = np.zeros((len(FM_parameters), 52))
    aux_arr[:, :len(aux_data_names)] = aux_data[aux_data_names].to_numpy()

    # prepare spectra
    spec_matrix = to_matrix(spec_data)

    ind_means = np.mean(spec_matrix[:, :, 1], axis=1).reshape((-1, 1))
    mean_stds = (ind_means - 0.00825) / 0.01165
    ind_mean_spectra = spec_matrix[:, :, 1] - ind_means

    scale_factor = 3.
    noises = spec_matrix[:, :, 2].reshape(-1, 52)
    scaled_noise = np.tanh(np.sqrt(np.square(noises - np.mean(noises, axis=1).reshape((-1, 1))) / (0.0001285 ** 2)))

    pkl_file = "/home/lwelzel/Documents/git/maldcope/data/TrainingData/training_power_transformer.pkl"

    # with open(pkl_file, "wb+") as f:
    #     pt = PowerTransformer(method="yeo-johnson")
    #     tspectra = (pt.fit_transform(ind_mean_spectra.T).T) / scale_factor
    #     dill.dump(pt, f)

    with open(pkl_file, "rb") as f:
        pt = dill.load(f)
        tspectra = (pt.transform(ind_mean_spectra.T).T) / scale_factor

    plot_random_spectra(tspectra, scaled_noise)

    planet_list = spec_data.keys()

    pairs = []
    valid_traces = []
    for i, p in tqdm(enumerate(planet_list), total=len(planet_list), unit=' pairs constructed'):
        arr_id = int(p.replace("Planet_train", "").replace("Planet_test", "")) - 1

        csv_name = p.replace("Planet_", "")

        true_spectrum = spec_data[p]['instrument_spectrum'][:]
        spectrum = tspectra[arr_id]
        noise = scaled_noise[arr_id]

        aux_arr = np.zeros(52)
        aux_arr[:len(aux_data_names)] = aux_data[aux_data_names].loc[csv_name].to_numpy()
        aux_arr[len(aux_data_names)] = (np.mean(true_spectrum) - 0.00825) / 0.01165

        trace = trace_data[p]['tracedata'][()]
        weights = trace_data[p]['weights'][()]

        valid_trace = ~np.any(np.isnan(weights))

        if valid_trace:
            thr = weights > weight_threshold
            trace = rescale_output(trace[thr], forward=True)
            weights = weights[thr]


        fm_theta = FM_parameters[theta_names].loc[csv_name].to_numpy().astype(float).flatten()

        x = np.array([
            spectrum,
            noise,
            aux_arr
        ], dtype=float)

        pair = (
            p,
            valid_trace,
            trace,
            weights,
            fm_theta,
            x
        )

        valid_traces.append(valid_trace)
        pairs.append(pair)

    return pairs, valid_traces

def make_full_test_pairs(spec_data, aux_data, trace_data, FM_parameters, weight_threshold=1e-8):
    # prepare aux
    aux_data_names = list(['star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature', 'planet_mass_kg',
                      'planet_orbital_period', 'planet_distance', 'planet_surface_gravity'])
    theta_names = list(['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3'])

    aux_arr = np.zeros((len(FM_parameters), 52))
    aux_arr[:, :len(aux_data_names)] = aux_data[aux_data_names].to_numpy()

    # prepare spectra
    spec_matrix = to_matrix(spec_data, id_name="Planet_test")

    ind_means = np.mean(spec_matrix[:, :, 1], axis=1).reshape((-1, 1))
    mean_stds = (ind_means - 0.00825) / 0.01165
    ind_mean_spectra = spec_matrix[:, :, 1] - ind_means

    scale_factor = 3.
    noises = spec_matrix[:, :, 2].reshape(-1, 52)
    scaled_noise = np.tanh(np.sqrt(np.square(noises - np.mean(noises, axis=1).reshape((-1, 1))) / (0.0001285 ** 2)))

    pkl_file = "/home/lwelzel/Documents/git/maldcope/data/TrainingData/training_power_transformer.pkl"

    # with open(pkl_file, "wb+") as f:
    #     pt = PowerTransformer(method="yeo-johnson")
    #     tspectra = (pt.fit_transform(ind_mean_spectra.T).T) / scale_factor
    #     dill.dump(pt, f)

    with open(pkl_file, "rb") as f:
        pt = dill.load(f)
        tspectra = (pt.transform(ind_mean_spectra.T).T) / scale_factor

    plot_random_spectra(tspectra, scaled_noise)

    planet_list = spec_data.keys()

    pairs = []
    valid_traces = []
    for i, p in tqdm(enumerate(planet_list), total=len(planet_list), unit=' pairs constructed'):
        arr_id = int(p.replace("Planet_train", "").replace("Planet_test", "")) - 1

        csv_name = p.replace("Planet_", "")

        true_spectrum = spec_data[p]['instrument_spectrum'][:]
        spectrum = tspectra[arr_id]
        noise = scaled_noise[arr_id]

        aux_arr = np.zeros(52)
        aux_arr[:len(aux_data_names)] = aux_data[aux_data_names].loc[csv_name].to_numpy()
        aux_arr[len(aux_data_names)] = (np.mean(true_spectrum) - 0.00825) / 0.01165

        x = np.array([
            spectrum,
            noise,
            aux_arr
        ], dtype=float)

        pair = (
            p,
            True,
            np.nan,
            np.nan,
            np.nan,
            x
        )

        valid_traces.append(True)
        pairs.append(pair)

    return pairs, valid_traces

def split_datasets(valid_traces, train_f=0.90,  val_f=0.08,
                   training_path=None):

    if training_path is None:
        training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TrainingData")

    try:
        valid_traces = np.array(valid_traces)
    except BaseException:
        valid_traces = valid_traces.cpu().numpy()

    total_samples = len(valid_traces)
    n_theta_samples = np.sum(~valid_traces)
    n_trace_samples = np.sum(valid_traces)

    train_n_theta = int(n_theta_samples * train_f)
    val_n_theta = int(n_theta_samples * val_f)
    test_n_theta = int(n_theta_samples - train_n_theta - val_n_theta)

    train_n_trace = int(n_trace_samples * train_f)
    val_n_trace = int(n_trace_samples * val_f)
    test_n_trace = int(n_trace_samples - train_n_trace - val_n_trace)

    print(f"Split:\n"
          f"\t Training:   \n"
          f"\t\t Theta:  {train_n_theta:05} ({(train_n_theta / total_samples) * 100.:03.1f} % of total,  {(train_n_theta / n_theta_samples) * 100.:03.1f} % of theta)\n"
          f"\t\t Traces: {train_n_trace:05} ({(train_n_trace / total_samples) * 100.:03.1f} % of total,  {(train_n_trace / n_trace_samples) * 100.:03.1f} % of trace)\n"
          f"\t Validation:   \n"
          f"\t\t Theta:  {val_n_theta:05} ({(val_n_theta / total_samples) * 100.:03.1f} % of total,  {(val_n_theta / n_theta_samples) * 100.:03.1f} % of theta)\n"
          f"\t\t Traces: {val_n_trace:05} ({(val_n_trace / total_samples) * 100.:03.1f} % of total,  {(val_n_trace / n_trace_samples) * 100.:03.1f} % of trace)\n"
          f"\t Testing:   \n"
          f"\t\t Theta:  {test_n_theta:05} ({(test_n_theta / total_samples) * 100.:03.1f} % of total,  {(test_n_theta / n_theta_samples) * 100.:03.1f} % of theta)\n"
          f"\t\t Traces: {test_n_trace:05} ({(test_n_trace / total_samples) * 100.:03.1f} % of total,  {(test_n_trace / n_trace_samples) * 100.:03.1f} % of trace)\n")

    idxs_thetas = np.argwhere(~valid_traces).flatten()
    idxs_traces = np.argwhere(valid_traces).flatten()

    rng = default_rng()
    rng.shuffle(idxs_thetas)
    rng.shuffle(idxs_traces)

    train_idx = list(idxs_thetas[:train_n_theta]) \
                + list(idxs_traces[:train_n_trace])
    val_idx = list(idxs_thetas[train_n_theta:train_n_theta + val_n_theta]) \
              + list(idxs_traces[train_n_trace:train_n_trace + val_n_trace])
    test_idx = list(idxs_thetas[train_n_theta + val_n_theta:]) \
               + list(idxs_traces[train_n_trace + val_n_trace:])

    np.save(str(training_path / f'train_idx.npy'), train_idx)
    np.save(str(training_path / f'val_idx.npy'), val_idx)
    np.save(str(training_path / f'test_idx.npy'), test_idx)

    return train_idx, val_idx, test_idx

def write_full_data():
    training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TrainingData/")

    # X DATA
    spec_data = h5py.File(training_path / 'SpectralData.hdf5')
    aux_data = pd.read_csv(training_path / "AuxillaryTable.csv",
                           index_col=0)

    aux_data = rescale_aux_input(aux_data)

    # Y DATA - THETA
    trace_data = h5py.File(training_path / 'Ground Truth Package/Tracedata.hdf5')
    FM_parameters = pd.read_csv(training_path / "Ground Truth Package/FM_Parameter_Table.csv",
                                index_col=1)

    trace_data = trace_data  # NOT RESCALED!
    FM_parameters = rescale_fm_theta_input(FM_parameters)

    # Pairs
    pairs, valid_traces = make_full_pairs(spec_data, aux_data, trace_data, FM_parameters)

    # TRAIN - VAL - TEST SPLIT
    train_idx, val_idx, test_idx = split_datasets(valid_traces)

    # TRAINING_DATA
    filename = training_path / f'training_dataset_full.h5'
    PosteriorDataset.store(
        pairs=[pairs[i] for i in train_idx],
        file=filename,
        overwrite=True,
    )

    # VALIDATION_DATA
    filename = training_path / f'validation_dataset_full.h5'
    PosteriorDataset.store(
        pairs=[pairs[i] for i in val_idx],
        file=filename,
        overwrite=True,
    )

    # TEST_DATA
    filename = training_path / f'testing_dataset_full.h5'
    PosteriorDataset.store(
        pairs=[pairs[i] for i in test_idx],
        file=filename,
        overwrite=True,
    )


def write_full_test_data():
    training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TrainingData/")

    # X DATA
    spec_data = h5py.File(Path("/home/lwelzel/Documents/git/maldcope/data/TestData") / 'SpectralData.hdf5')
    aux_data = pd.read_csv(Path("/home/lwelzel/Documents/git/maldcope/data/TestData") / "AuxillaryTable.csv",
                           index_col=0)

    aux_data = rescale_aux_input(aux_data)

    # Y DATA - THETA
    trace_data = h5py.File(training_path / 'Ground Truth Package/Tracedata.hdf5')
    FM_parameters = pd.read_csv(training_path / "Ground Truth Package/FM_Parameter_Table.csv",
                                index_col=1)

    trace_data = trace_data  # NOT RESCALED!
    FM_parameters = rescale_fm_theta_input(FM_parameters)

    # Pairs
    pairs, valid_traces = make_full_test_pairs(spec_data, aux_data, trace_data, FM_parameters)

    # TRAIN - VAL - TEST SPLIT
    # train_idx, val_idx, test_idx = split_datasets(valid_traces)

    # TRAINING_DATA
    filename = Path("/home/lwelzel/Documents/git/maldcope/data/TestData") / f'testing_dataset.h5'
    PosteriorDataset.store(
        pairs=pairs,
        file=filename,
        overwrite=True,
    )



def join_paras(df1, df2, l1, l2):
    c1, c2 = len(l1), len(l2)
    n1, n2 = len(df1), len(df2)

    assert n1 == n2

    x = np.zeros((n1, c1 + c2))
    x[:, :c1] = df1[l1].to_numpy()
    x[:, c1:] = df2[l2].to_numpy()

    return x, l1 + l2

def to_matrix(SpectralData, id_name="Planet_train"):
    # id is in ascending order
    num = len(SpectralData.keys())
    id_order = np.arange(num)
    # we knew the instrument resolution beforehand
    observed_spectrum = np.zeros((num, 52, 4))
    for idx, x in tqdm(enumerate(id_order), total=num):
        current_id = f'{id_name}{x + 1}'
        wlgrid = SpectralData[current_id]['instrument_wlgrid'][:]
        spectrum = SpectralData[current_id]['instrument_spectrum'][:]
        noise = SpectralData[current_id]['instrument_noise'][:]
        wlwidth = SpectralData[current_id]['instrument_width'][:]
        observed_spectrum[idx, :, :] = np.concatenate(
            [wlgrid[..., np.newaxis], spectrum[..., np.newaxis], noise[..., np.newaxis], wlwidth[..., np.newaxis]],
            axis=-1)
    return observed_spectrum


def make_pairs_base(spec_matrix, FM_parameters, idxs, theta_names, which_norm="median", **kwargs):
    ind_mean_spectra = spec_matrix[:, :, 1] - np.mean(spec_matrix[:, :, 1], axis=1).reshape((-1, 1))
    ind_mean_noise = np.abs(spec_matrix[:, :, 2] - np.mean(spec_matrix[:, :, 2], axis=1).reshape((-1, 1)))

    # mean_spectrum = np.mean(ind_mean_spectra, axis=0)
    # median_spectrum = np.median(ind_mean_spectra, axis=0)
    #
    # mean_noise = np.mean(spec_matrix[:, :, 2], axis=0)
    # median_noise = np.median(spec_matrix[:, :, 2], axis=0)

    if which_norm == "mean":
        norm_spectrum = np.mean(ind_mean_spectra, axis=0)
        norm_noise = np.mean(ind_mean_noise, axis=0)
    elif which_norm == "median":
        norm_spectrum = np.median(ind_mean_spectra, axis=0)
        norm_noise = np.median(ind_mean_noise, axis=0)
    else:
        raise NotImplementedError

    if isinstance(FM_parameters, pd.DataFrame):
        pairs = [
            (FM_parameters.iloc[i][theta_names].to_numpy().astype(float).reshape((1, -1)),
             np.array([
                 ind_mean_spectra[i] - norm_spectrum,
                 np.abs(ind_mean_noise[i] - norm_noise)
             ],
                      dtype=float).reshape((1, 2, -1)))
            for i in idxs
        ]
    elif isinstance(FM_parameters, np.ndarray):
        pairs = [
            (FM_parameters.reshape((1, -1)),
             np.array([
                 ind_mean_spectra[i] - norm_spectrum,
                 np.abs(ind_mean_noise[i] - norm_noise)
             ],
                      dtype=float).reshape((1, 2, -1)))
            for i in idxs
        ]
    else:
        raise NotImplementedError

    return pairs


def make_pairs_base_aux(spec_matrix, FM_parameters, idxs, theta_names, aux_data, which_norm="median",**kwargs):
    ind_mean_spectra = spec_matrix[:, :, 1] - np.mean(spec_matrix[:, :, 1], axis=1).reshape((-1, 1))
    ind_mean_noise = np.abs(spec_matrix[:, :, 2] - np.mean(spec_matrix[:, :, 2], axis=1).reshape((-1, 1)))

    # mean_spectrum = np.mean(ind_mean_spectra, axis=0)
    # median_spectrum = np.median(ind_mean_spectra, axis=0)
    #
    # mean_noise = np.mean(spec_matrix[:, :, 2], axis=0)
    # median_noise = np.median(spec_matrix[:, :, 2], axis=0)

    if which_norm == "mean":
        norm_spectrum = np.mean(ind_mean_spectra, axis=0)
        norm_noise = np.mean(ind_mean_noise, axis=0)
    elif which_norm == "median":
        norm_spectrum = np.median(ind_mean_spectra, axis=0)
        norm_noise = np.median(ind_mean_noise, axis=0)
    else:
        raise NotImplementedError

    aux_data_names = ['star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature', 'planet_mass_kg',
                      'planet_orbital_period', 'planet_distance', 'planet_surface_gravity']

    aux_arr = np.zeros((len(FM_parameters), 52))
    aux_arr[:, :len(aux_data_names)] = aux_data[aux_data_names].to_numpy()

    if isinstance(FM_parameters, pd.DataFrame):
        pairs = [
            (FM_parameters.iloc[i][theta_names].to_numpy().astype(float).reshape((1, -1)),
             np.array([
                 ind_mean_spectra[i] - norm_spectrum,
                 np.abs(ind_mean_noise[i] - norm_noise),
                 aux_arr[i]
             ],
                      dtype=float).reshape((1, 3, -1)))
            for i in idxs
        ]
    elif isinstance(FM_parameters, np.ndarray):
        pairs = [
            (FM_parameters.reshape((1, -1)),
             np.array([
                 ind_mean_spectra[i] - norm_spectrum,
                 np.abs(ind_mean_noise[i] - norm_noise),
                 aux_arr[i]
             ],
                      dtype=float).reshape((1, 3, -1)))
            for i in idxs
        ]
    else:
        raise NotImplementedError

    return pairs


def make_pairs_full_norm_aux(spec_matrix, FM_parameters, idxs, theta_names, aux_data, which_norm="median",**kwargs):
    ind_mean_spectra = spec_matrix[:, :, 1] - np.mean(spec_matrix[:, :, 1], axis=1).reshape((-1, 1))
    std_spectra = np.std(spec_matrix[:, :, 1], axis=0)

    ind_mean_noise = np.abs(spec_matrix[:, :, 2] - np.mean(spec_matrix[:, :, 2], axis=1).reshape((-1, 1)))
    std_noise = np.std(spec_matrix[:, :, 2], axis=0)


    # mean_spectrum = np.mean(ind_mean_spectra, axis=0)
    # median_spectrum = np.median(ind_mean_spectra, axis=0)
    #
    # mean_noise = np.mean(spec_matrix[:, :, 2], axis=0)
    # median_noise = np.median(spec_matrix[:, :, 2], axis=0)

    if which_norm == "mean":
        norm_spectrum = np.mean(ind_mean_spectra, axis=0)
        norm_noise = np.mean(ind_mean_noise, axis=0)
    elif which_norm == "median":
        norm_spectrum = np.median(ind_mean_spectra, axis=0)
        norm_noise = np.median(ind_mean_noise, axis=0)
    else:
        raise NotImplementedError

    aux_data_names = ['star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature', 'planet_mass_kg',
                      'planet_orbital_period', 'planet_distance', 'planet_surface_gravity']

    aux_arr = np.zeros((len(FM_parameters), 52))
    aux_arr[:, :len(aux_data_names)] = aux_data[aux_data_names].to_numpy()

    if isinstance(FM_parameters, pd.DataFrame):
        pairs = [
            (FM_parameters.iloc[i][theta_names].to_numpy().astype(float).reshape((1, -1)),
             np.array([
                 (ind_mean_spectra[i] - norm_spectrum) / std_spectra,
                 (np.abs(ind_mean_noise[i] - norm_noise)) / std_noise,
                 aux_arr[i]
             ],
                      dtype=float).reshape((1, 3, -1)))
            for i in idxs
        ]
    elif isinstance(FM_parameters, np.ndarray):
        pairs = [
            (FM_parameters.reshape((1, -1)),
             np.array([
                 ind_mean_spectra[i] - norm_spectrum,
                 np.abs(ind_mean_noise[i] - norm_noise),
                 aux_arr[i]
             ],
                      dtype=float).reshape((1, 3, -1)))
            for i in idxs
        ]
    else:
        raise NotImplementedError

    return pairs


def write_lampe_data(input_type="base"):
    training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TrainingData/")

    aux_data = pd.read_csv(training_path / "AuxillaryTable.csv")
    spec_data = h5py.File(training_path / 'SpectralData.hdf5')
    planet_list = [p for p in spec_data.keys()]
    FM_parameters = pd.read_csv(training_path / "Ground Truth Package/FM_Parameter_Table.csv",
                                index_col=0)

    aux_data_names = ['star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature', 'planet_mass_kg',
                      'planet_orbital_period', 'planet_distance', 'planet_surface_gravity']

    theta_names = ['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']

    aux_data = rescale_aux_input(aux_data)

    FM_parameters['planet_radius'] = FM_parameters['planet_radius'] - 0.5
    FM_parameters['planet_temp'] = (FM_parameters['planet_temp'] - 1000.) / 1000
    FM_parameters['log_H2O'] = FM_parameters['log_H2O'] + 6.
    FM_parameters['log_CO2'] = FM_parameters['log_CO2'] + 7.
    FM_parameters['log_CO'] = FM_parameters['log_CO'] + 5.
    FM_parameters['log_CH4'] = FM_parameters['log_CH4'] + 6.
    FM_parameters['log_NH3'] = FM_parameters['log_NH3'] + 7.

    # all_para, all_para_names = join_paras(FM_parameters, aux_data,
    #                                       theta_names, aux_data_names)
    #
    # fig = corner(
    #     data=all_para,
    #     labels=all_para_names
    # )
    #
    # fig.savefig("/home/lwelzel/Documents/git/maldcope/data/TrainingData/theta_aux_corner.png", dpi=400)

    spec_matrix = to_matrix(spec_data)

    total_samples = len(planet_list)
    train_f = 0.90
    val_f = 0.08

    train_n = int(total_samples * train_f)
    val_n = int(total_samples * val_f)
    test_n = int(total_samples - train_n - val_n)

    print(f"Split:\n"
          f"\t Training:   {train_n:05} ({(train_n / total_samples) * 100.:03.1f} %)\n"
          f"\t Validation: {val_n:05} ({(val_n / total_samples) * 100.:03.1f} %)\n"
          f"\t Testing:    {test_n:05} ({(test_n / total_samples) * 100.:03.1f} %)\n")

    rng = default_rng()
    idxs = np.arange(total_samples, dtype=int)
    rng.shuffle(idxs)

    train_idx = idxs[:train_n]
    val_idx = idxs[train_n:train_n + val_n]
    test_idx = idxs[train_n + val_n:]

    np.save(str(training_path / f'train_idx.npy'), train_idx)
    np.save(str(training_path / f'val_idx.npy'), val_idx)
    np.save(str(training_path / f'test_idx.npy'), test_idx)

    if input_type == "base":
        make_pairs = make_pairs_base
    elif input_type == "aux":
        make_pairs = make_pairs_base_aux
    elif input_type == "full_norm_aux":
        make_pairs = make_pairs_full_norm_aux
    else:
        raise NotImplementedError


    pairs = make_pairs(spec_matrix,
                       FM_parameters,
                       idxs,
                       theta_names,
                       which_norm="median",
                       aux_data=aux_data,
                       )

    # TRAINING_DATA
    filename = training_path / f'training_dataset_{input_type}.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs[:train_n],
        file=filename,
        size=train_n,
        overwrite=True,
    )

    # VALIDATION_DATA
    filename = training_path / f'validation_dataset_{input_type}.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs[train_n:train_n + val_n],
        file=filename,
        size=val_n,
        overwrite=True,
    )

    # TEST_DATA
    filename = training_path / f'testing_dataset_{input_type}.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs[train_n + val_n:],
        file=filename,
        size=test_n,
        overwrite=True,
    )


def write_lampe_test_data(input_type="base"):
    training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TestData/")

    aux_data = pd.read_csv(training_path / "AuxillaryTable.csv")
    spec_data = h5py.File(training_path / 'SpectralData.hdf5')
    planet_list = [p for p in spec_data.keys()]

    aux_data = rescale_aux_input(aux_data)

    theta_names = ['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']

    total_samples = len(planet_list)
    test_n = int(total_samples)

    idxs = np.arange(total_samples, dtype=int)
    empty = np.full((len(planet_list), len(theta_names)), fill_value=np.nan).reshape((-1, 7))

    spec_matrix = to_matrix(spec_data, id_name="Planet_public")

    if input_type == "base":
        make_pairs = make_pairs_base
    elif input_type == "aux":
        make_pairs = make_pairs_base_aux
    else:
        raise NotImplementedError


    pairs = make_pairs(spec_matrix,
                       empty,
                       idxs,
                       theta_names,
                       which_norm="median",
                       aux_data=aux_data,
                       )

    # TEST_DATA
    filename = training_path / f'testing_dataset.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs,
        file=filename,
        size=test_n,
        overwrite=True,
    )


def write_lampe_chain_data(input_type="base"):
    training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TrainingData/")

    aux_data = pd.read_csv(training_path / "AuxillaryTable.csv")
    spec_data = h5py.File(training_path / 'SpectralData.hdf5')
    trace_data = h5py.File(training_path / 'Ground Truth Package/Tracedata.hdf5')
    FM_parameters = pd.read_csv(training_path / "Ground Truth Package/FM_Parameter_Table.csv",
                                index_col=0)

    aux_data_names = ['star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature', 'planet_mass_kg',
                      'planet_orbital_period', 'planet_distance', 'planet_surface_gravity']

    theta_names = ['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']

    aux_data = rescale_aux_input(aux_data)

    FM_parameters['planet_radius'] = FM_parameters['planet_radius'] - 0.5
    FM_parameters['planet_temp'] = (FM_parameters['planet_temp'] - 1000.) / 1000
    FM_parameters['log_H2O'] = FM_parameters['log_H2O'] + 6.
    FM_parameters['log_CO2'] = FM_parameters['log_CO2'] + 7.
    FM_parameters['log_CO'] = FM_parameters['log_CO'] + 5.
    FM_parameters['log_CH4'] = FM_parameters['log_CH4'] + 6.
    FM_parameters['log_NH3'] = FM_parameters['log_NH3'] + 7.

    planet_list = [p for p in spec_data.keys()]
    planet_list_traces = [p for p in trace_data.keys()]

    chain_shapes = [trace_data[f"Planet_train{i+1}"]['tracedata'].shape for i in range(len(planet_list_traces))]
    valid_chains = [len(cs) != 0 for cs in chain_shapes]
    valid_idx = np.argwhere(valid_chains)
    valid_idx_keys = [f"Planet_train{idx+1}" for idx in valid_idx.flatten()]

    valid_file_name = str(training_path / "valid_chain_data.hdf5")
    valid_chain_file = h5py.File(valid_file_name, 'w')

    spec_matrix = to_matrix(spec_data)

    np.save(str(training_path / f'valid_chain_idx.npy'), valid_idx.flatten())

    if input_type == "base":
        make_pairs = make_pairs_base
    elif input_type == "aux":
        make_pairs = make_pairs_base_aux
    else:
        raise NotImplementedError

    pairs = make_pairs(spec_matrix,
                       FM_parameters,
                       valid_idx.flatten(),
                       theta_names,
                       which_norm="median",
                       aux_data=aux_data,
                       )

    spec_matrix = spec_matrix[valid_chains]

    # from train import NPEWithEmbedding
    # state = torch.load('/home/lwelzel/Documents/git/maldcope/runs/sbiear_experiment1/state.pth', map_location='cpu')
    # estimator = NPEWithEmbedding()
    # estimator.load_state_dict(state)
    # estimator.cuda()
    # estimator.eval()
    # estimator.embedding.eval()

    for i, (idx, pair) in tqdm(enumerate(zip(valid_idx.flatten(), pairs)),
                               total=len(valid_idx.flatten())):
        key = f"Planet_train{idx+1}"

        theta, x = pair

        grp = valid_chain_file.create_group(key)
        pl_id = grp.attrs['ID'] = idx
        tracedata = grp.create_dataset('tracedata', data=trace_data[key]['tracedata'])
        weights = grp.create_dataset('weights', data=trace_data[key]['weights'])
        spec_group = grp.create_dataset('spectrum', data=spec_matrix[i])

        aux = grp.create_dataset('aux_data', data=aux_data[aux_data_names].iloc[idx].to_numpy())

        theta_prime = grp.create_dataset('theta', data=FM_parameters[theta_names].iloc[idx].to_numpy())
        assert np.all(theta_prime == theta.reshape((1, 7)))
        x_data = grp.create_dataset('x', data=x.reshape((-1, 52)))

        # x_star, x_prime_star = torch.tensor(x[:, 0].reshape((1, 1, -1)), dtype=torch.float), torch.tensor(x[:, 2].reshape((1, -1)), dtype=torch.float)
        #
        # with torch.no_grad():
        #     theta_star = estimator.flow(x_star.cuda(), x_prime_star.cuda()).sample((512,)).reshape((-1, 7)).cpu().numpy()
        #
        # tracedata_model = grp.create_dataset('tracedata_model', data=theta_star)


    valid_chain_file.close()

    # TRAINING_DATA
    filename = training_path / f'valid_chain_dataset_{input_type}.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs,
        file=filename,
        size=len(valid_idx.flatten()),
        overwrite=True,
    )

if __name__ == "__main__":
    from chainconsumer import ChainConsumer
    # write_lampe_data(input_type="aux")
    # write_lampe_test_data(input_type="aux")
    # write_lampe_chain_data(input_type="aux")
    # write_lampe_chain_data(input_type="aux")
    # write_full_data()

    write_full_test_data()

    raise NotImplementedError

    input_type = "_full"

    # testset = PosteriorDataset(
    #     file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/testing_dataset{input_type}.h5"),
    #     batch_size=16,
    #     sample_size=512,
    #     shuffle=True,
    # )
    # testset.to_memory()

    validation = PosteriorDataset(
        file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/training_dataset{input_type}.h5"),
        batch_size=16,
        sample_size=512,
        shuffle=True,
    )
    validation.to_memory()

    c = ChainConsumer()
    c.add_chain(validation.traces.view(-1, 7).cpu().numpy(),
                weights=validation.weights.view(-1).cpu().numpy(),
                name="Traces")

    c.add_chain(validation._fm_theta.cpu().numpy(),
                name="FM model")


    c.configure(smooth=0)
    fig = c.plotter.plot()
    plt.show()
    raise NotImplementedError

    from chainconsumer import ChainConsumer

    for s in testset.__iter_trace_theta_x__():
        trace, theta, x = s

        c = ChainConsumer()
        c.add_chain(trace[0].cpu().numpy())
        # std = torch.std(trace[0], dim=1)
        # z = torch.zeros_like(std, device=device)
        # theta_ish = theta[0][None, ...].tile(512, 1) + torch.normal(0, 0.1, (512, 7), device=device)
        # c.add_chain(theta_ish.cpu().numpy(), shade_alpha=0.01, name="BS")

        c.configure(smooth=0)
        fig = c.plotter.plot(truth=theta[0].cpu().numpy())
        plt.show()

        break

