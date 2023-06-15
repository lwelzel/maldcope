import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchviz import make_dot
from torchview import draw_graph

from pathlib import Path
from tqdm import tqdm
from graphviz import Source
from itertools import islice
import pandas as pd
import h5py

from lampe.data import H5Dataset
from lampe.diagnostics import expected_coverage_mc
from lampe.plots import nice_rc, corner, mark_point, coverage_plot

from train import NPEWithEmbedding

def rescale_output(input, forward=False):

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

    return out


def produce_AMLDC_output(model=None, dataset_path=None, verbose=False):
    if model is None:
        model = Path('/home/lwelzel/Documents/git/maldcope/runs/MALDCOPE_REV_C/state.pth')
    if dataset_path is None:
        dataset_path = Path('/home/lwelzel/Documents/git/maldcope/data/TestData/testing_dataset.h5')

    state = torch.load(str(model), map_location='cpu')

    estimator = NPEWithEmbedding()
    estimator.load_state_dict(state)
    estimator.cuda()
    estimator.eval()
    estimator.embedding.eval()

    testset = H5Dataset(dataset_path)

    n_samples = 2 ** 11
    n_select = 1250
    traces = torch.zeros((len(testset), n_select, 7))

    _n = n_select
    for i, (theta, x) in tqdm(enumerate(testset),
                              total=len(testset)):

        x, x_prime = x[0].cuda().reshape(-1, 1, 52), x[2].cuda().reshape(-1, 52)

        with torch.no_grad():
            theta = estimator.flow(x, x_prime).sample((n_samples,))

        theta = theta.reshape((-1, 7))
        good_idx = torch.isfinite(torch.sum(theta, dim=1))

        if torch.sum(torch.sum(good_idx)) < n_select:
            temp_samples = int(n_samples * (n_samples / torch.sum(torch.isfinite(torch.sum(theta, dim=1))).cpu().numpy()) * 2)
            print(f"Too few samples, sampling again for {temp_samples} draws.")
            with torch.no_grad():
                theta = estimator.flow(x, x_prime).sample((temp_samples,))

            theta = theta.reshape((-1, 7))
            good_idx = torch.isfinite(torch.sum(theta, dim=1))


            print(f"Not enough valid samples: {torch.sum(torch.sum(good_idx))} / {n_select} required")
            _n = torch.minimum(torch.tensor(_n), torch.sum(good_idx))
            _n = int(_n.cpu().numpy())

        theta = theta[good_idx]
        theta = theta[:_n]
        theta = rescale_output(theta, forward=False)

        traces[i, :_n] = theta


    traces = traces[:, :_n].cpu().numpy()
    weights = (np.ones((len(testset), _n)) / int(_n))
    to_regular_track_format(traces, weights, path=dataset_path.parent, name="RT_submission.hdf5")

    q1, q2, q3 = np.quantile(traces, [0.16, 0.5, 0.84], axis=1)

    to_light_track_format(q1, q2, q3, path=dataset_path.parent, name="LT_submission.csv")

    if verbose:
        sample_id = np.random.randint(0, 9)
        theta_star, x_star = testset[sample_id]
        theta_star, x_star = theta_star.cuda(), x_star.cuda()
        x_star, x_prime_star = x_star[0].reshape((1, 1, -1)), x_star[2].reshape((1, -1))

        with torch.no_grad():
            theta = estimator.flow(x_star.cuda(), x_prime_star.cuda()).sample((2 ** 12,))

        theta = theta.reshape((-1, 7))
        theta = rescale_output(theta, forward=False)

        LABELS, LOWER, UPPER = zip(*[
            [r'${\rm R_{P}}$', -1e2, 1e2],  # planet_radius
            [r'${\rm T_{P}}$', -1e2, 1e2],  # planet_temp
            [r'$\log X_{\rm H_{2}O}$', -1e2, 1e2],  # log_H2O
            [r'$\log X_{\rm CO_{2}}$', -1e2, 1e2],  # log_CO2
            [r'$\log X_{\rm CO}$', -1e2, 1e2],  # log_CO
            [r'$\log X_{\rm CH_{4}}$', -1e2, 1e2],  # log_CH4
            [r'$\log X_{\rm NH_{3}}$', -1e2, 1e2],  # log_NH3
        ])

        fig = corner(
            theta.cpu(),
            smooth=2,
            domain=None,  # (LOWER, UPPER),
            labels=LABELS,
            legend=r'$p_{\phi}(\theta | x^*)$',
            figsize=(12, 12),
        )
        fig.suptitle(f"ID: {sample_id}")

        fig.savefig("/home/lwelzel/Documents/git/maldcope/random_corner_AMLDC_test_data.png", dpi=300)


def to_light_track_format(q1_array, q2_array, q3_array, path, columns=None, name="LT_submission.csv"):
    """Helper function to prepare submission file for the light track,
    we assume the test data is arranged in assending order of the planet ID.

    Args:
        q1_array: N x 6 array containing the estimates for 16% percentile
        q2_array: N x 6 array containing the estimates for 50% percentile
        q3_array: N x 6 array containing the estimates for 84% percentile
        columns: columns for the df. default to none

    Returns:
        Pandas DataFrame object
    """

    submit_file = str(path / name)

    # create empty array
    LT_submission_df = pd.DataFrame(columns=columns)
    # sanity check - length should be equal
    assert len(q1_array) == len(q2_array) == len(q3_array)
    targets_label = ["planet_radius", 'T', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']
    # create columns for df
    default_quartiles = ['q1', 'q2', 'q3']
    default_columns = []
    for c in targets_label:
        for q in default_quartiles:
            default_columns.append(c + q)

    if columns is None:
        columns = default_columns
    for i in tqdm(range(len(q1_array))):
        quartiles_dict = {}
        quartiles_dict['planet_ID'] = i
        for t_idx, t in enumerate(targets_label):
            quartiles_dict[f'{t}_q1'] = q1_array[i, t_idx]
            quartiles_dict[f'{t}_q2'] = q2_array[i, t_idx]
            quartiles_dict[f'{t}_q3'] = q3_array[i, t_idx]
        LT_submission_df = pd.concat([LT_submission_df, pd.DataFrame.from_records([quartiles_dict])], axis=0,
                                     ignore_index=True)
    LT_submission_df.to_csv(submit_file, index=False)
    return LT_submission_df


def to_regular_track_format(tracedata_arr, weights_arr, path, name="RT_submission.hdf5"):
    """convert input into regular track format.
    we assume the test data is arranged in assending order of the planet ID.

    Args:
        tracedata_arr (array): Tracedata array, usually in the form of N x M x 6, where M is the number of tracedata, here we assume tracedata is of equal size. It does not have to be but you will need to craete an alternative function if the size is different.
        weights_arr (array): Weights array, usually in the form of N x M, here we assumed the number of weights is of equal size, it should have the same size as the tracedata

    Returns:
        None
    """
    submit_file = str(path / name)
    RT_submission = h5py.File(submit_file, 'w')
    for n in range(len(tracedata_arr)):
        ## sanity check - samples count should be the same for both
        assert len(tracedata_arr[n]) == len(weights_arr[n])
        ## sanity check - weights must be able to sum to one.
        assert np.isclose(np.sum(weights_arr[n]), 1)

        grp = RT_submission.create_group(f"Planet_public{n+1}")
        pl_id = grp.attrs['ID'] = n
        tracedata = grp.create_dataset('tracedata', data=tracedata_arr[n])
        weight_adjusted = weights_arr[n]

        weights = grp.create_dataset('weights', data=weight_adjusted)
    RT_submission.close()

if __name__ == '__main__':
    produce_AMLDC_output(verbose=True)