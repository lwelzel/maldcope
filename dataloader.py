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


def to_matrix(SpectralData, id_name="Planet_train"):
    # id is in ascending order
    num = len(SpectralData.keys())
    id_order = np.arange(num)
    # we knew the instrument resolution beforehand
    observed_spectrum = np.zeros((num, 52, 4))
    for idx, x in enumerate(id_order):
        current_id = f'{id_name}{x + 1}'
        wlgrid = SpectralData[current_id]['instrument_wlgrid'][:]
        spectrum = SpectralData[current_id]['instrument_spectrum'][:]
        noise = SpectralData[current_id]['instrument_noise'][:]
        wlwidth = SpectralData[current_id]['instrument_width'][:]
        observed_spectrum[idx, :, :] = np.concatenate(
            [wlgrid[..., np.newaxis], spectrum[..., np.newaxis], noise[..., np.newaxis], wlwidth[..., np.newaxis]],
            axis=-1)
    return observed_spectrum

def make_pairs(spec_matrix, FM_parameters, idxs, theta_names, which_norm="median"):
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
    elif which_norm =="median":
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


def write_lampe_data():
    training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TrainingData/")

    aux_data = pd.read_csv(training_path / "AuxillaryTable.csv")
    spec_data = h5py.File(training_path / 'SpectralData.hdf5')
    planet_list = [p for p in spec_data.keys()]
    FM_parameters = pd.read_csv(training_path / "Ground Truth Package/FM_Parameter_Table.csv",
                                index_col=0)


    spec_matrix = to_matrix(spec_data)

    FM_parameters['planet_radius'] = FM_parameters['planet_radius'] - 0.5
    FM_parameters['planet_temp'] = (FM_parameters['planet_temp'] - 1000.) / 1000
    FM_parameters['log_H2O'] = FM_parameters['log_H2O'] + 6.
    FM_parameters['log_CO2'] = FM_parameters['log_CO2'] + 7.
    FM_parameters['log_CO'] = FM_parameters['log_CO'] + 5.
    FM_parameters['log_CH4'] = FM_parameters['log_CH4'] + 6.
    FM_parameters['log_NH3'] = FM_parameters['log_NH3'] + 7.

    theta_names = ['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']

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

    pairs = make_pairs(spec_matrix,
                       FM_parameters,
                       idxs,
                       theta_names,
                       which_norm="median",
                       )

    # TRAINING_DATA
    filename = training_path / f'training_dataset.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs[:train_n],
        file=filename,
        size=train_n,
        overwrite=True,
    )

    # VALIDATION_DATA
    filename = training_path / f'validation_dataset.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs[train_n:train_n + val_n],
        file=filename,
        size=val_n,
        overwrite=True,
    )

    # TEST_DATA
    filename = training_path / f'testing_dataset.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs[train_n + val_n:],
        file=filename,
        size=test_n,
        overwrite=True,
    )

def write_lampe_test_data():
    training_path = Path("/home/lwelzel/Documents/git/maldcope/data/TestData/")

    aux_data = pd.read_csv(training_path / "AuxillaryTable.csv")
    spec_data = h5py.File(training_path / 'SpectralData.hdf5')
    planet_list = [p for p in spec_data.keys()]

    theta_names = ['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']

    total_samples = len(planet_list)
    test_n = int(total_samples)

    idxs = np.arange(total_samples, dtype=int)
    empty = np.full(len(theta_names), fill_value=np.nan).reshape((1, -1))

    spec_matrix = to_matrix(spec_data, id_name="Planet_public")

    pairs = make_pairs(spec_matrix,
                       empty,
                       idxs,
                       theta_names,
                       which_norm="median",
                       )

    # TEST_DATA
    filename = training_path / f'testing_dataset.h5'
    lampe.data.H5Dataset.store(
        pairs=pairs,
        file=filename,
        size=test_n,
        overwrite=True,
    )

if __name__ == "__main__":
    # write_lampe_data()
    write_lampe_test_data()
