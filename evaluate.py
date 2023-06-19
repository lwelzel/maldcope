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
from chainconsumer import ChainConsumer

from lampe.data import H5Dataset
from lampe.diagnostics import expected_coverage_mc
from lampe.plots import nice_rc, corner, mark_point, coverage_plot

from dataloader import PosteriorDataset
# from train import NPEWithEmbedding
# from train_MMD import RNPEWithEmbedding
from train_KLD import NPEWithEmbedding
from train_CD import RNPEWithEmbedding
from stat_tests import VecMMD, PolynomialKernel, RBFKernel
import h5py

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

input_type = "_full"

batch_size = 512  # 2048  # 4096
val_batch_size = 64  # int(np.clip(batch_size / 2**3, a_min=64, a_max=512))
theta_sample_size = 64

para_names = ['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']

def load_dataset(name="training"):
    if name == "training":
        print(f"Loading training dataset...")
        trainset = PosteriorDataset(
            file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/training_dataset{input_type}.h5"),
            batch_size=batch_size,
            sample_size=theta_sample_size,
            shuffle=True,
        )
        trainset.to_memory()
        return trainset
    elif name == "validation":
        print(f"Loading validation dataset...")
        validset = PosteriorDataset(
            file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/validation_dataset{input_type}.h5"),
            batch_size=64,
            sample_size=theta_sample_size,
            shuffle=True,
        )
        validset.to_memory()
        return validset
    elif name == "testing":
        print(f"Loading testing dataset...")
        testset = PosteriorDataset(
            file=Path(f"/home/lwelzel/Documents/git/maldcope/data/TrainingData/testing_dataset{input_type}.h5"),
            batch_size=64,
            sample_size=theta_sample_size,
            shuffle=True,
        )
        testset.to_memory()
        return testset
    else:
        raise NotImplementedError

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_estimator():
    state = torch.load('/home/lwelzel/Documents/git/maldcope/runs/KLD_experiment2/state.pth', map_location='cuda')

    estimator = NPEWithEmbedding().cuda()
    estimator.load_state_dict(state)
    estimator.cuda()
    estimator.eval()
    estimator.embedding.eval()

    print("Model parameters:", count_parameters(estimator))

    return estimator

def plot_random_spectra(dataset_name="validation"):
    dataset = load_dataset(dataset_name)

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
    for i in rng.integers(0, len(dataset), 50):
        name, is_valid, trace, weights, fm_theta, x = dataset[i]

        x_star, x_prime_star = x[0].reshape((1, 1, -1)), x[2].reshape((1, -1))
        x_star_noise = x[1].reshape((1, 1, -1))

        ax.errorbar(wlgrid,
                    x_star.view(-1).cpu().numpy(),  # - np.mean(mean_spectrum),
                    yerr=x_star_noise.view(-1).cpu().numpy(),
                    # xerr=wlwidth,
                    fmt="D",)

    ax.set_xlabel(r"$\lambda$ [$\mu m$]")
    ax.set_ylabel(r"Transit Depth [-]")
    ax.set_title(f"Mean Spectrum")

    plt.show()


def plot_random_corner(dataset_name="validation"):
    estimator = get_estimator()

    dataset = load_dataset(dataset_name)

    is_valid = False
    sample_id = 0
    while not is_valid:
        sample_id = np.random.randint(0, len(dataset))
        is_valid = dataset[sample_id][1]

    name, is_valid, trace, weights, fm_theta, x = dataset[sample_id]

    x_star, x_prime_star = x[0].reshape((1, 1, -1)), x[2].reshape((1, -1))
    x_star_noise = x[1].reshape((1, 1, -1))

    with torch.no_grad():
        pred = estimator.flow(x_star.cuda(), x_prime_star.cuda()).sample((2 ** 12,))
        # rpred = estimator.flow(x_star.cuda(), x_prime_star.cuda()).rsample((2 ** 12,))
        # valid_samples = torch.isfinite(rpred).all(dim=-1)
        # print(sum(valid_samples))
        # rpred = rpred[valid_samples]

    fm_theta = rescale_output(fm_theta, forward=False)
    trace = rescale_output(trace, forward=False)
    pred = rescale_output(pred, forward=False)
    # rpred = rescale_output(rpred, forward=False)

    # polykernel = PolynomialKernel(degree=1, gamma=None, coef0=1)
    # rbfkernel = RBFKernel()
    # pol_mmd = VecMMD(polykernel)
    # rbf_mmd = VecMMD(rbfkernel)



    # pol_dist = pol_mmd(pred.reshape(1, *pred.shape), trace.reshape(1, *trace.shape))
    # rbf_dist = rbf_mmd(pred.reshape(1, *pred.shape), trace.reshape(1, *trace.shape))

    # rpol_dist = pol_mmd(rpred.reshape(1, *pred.shape), trace.reshape(1, *trace.shape))
    # rrbf_dist = rbf_mmd(rpred.reshape(1, *pred.shape), trace.reshape(1, *trace.shape))

    # print("Poly", pol_dist, )# rpol_dist)
    # print("RBF", rbf_dist, ) #rrbf_dist)

    c = ChainConsumer()

    c.add_chain(trace.cpu().numpy(),
                weights=weights.cpu().numpy(),
                name="MCMC")

    c.add_chain(pred[torch.argwhere(torch.all(torch.isfinite(pred), dim=1)).flatten(), :].cpu().numpy(),
                name="RNPE")

    # c.add_chain(pred[torch.argwhere(torch.all(torch.isfinite(rpred), dim=1)).flatten(), :].cpu().numpy(),
    #             name="rRNPE")

    c.configure(smooth=0)
    fig = c.plotter.plot(truth=fm_theta.cpu().numpy())
    fig.suptitle(name )#+ f" RBF: {rbf_dist.item():.2e}   |  Poly: {pol_dist.item():.2e}, ")

    plt.show()

    # c = ChainConsumer()
    #
    # c.add_chain(trace.cpu().numpy(),
    #             weights=weights.cpu().numpy(),
    #             name="MCMC")
    #
    # # c.add_chain(pred[torch.argwhere(torch.all(torch.isfinite(pred), dim=1)).flatten(), :].cpu().numpy(),
    # #             name="RNPE")
    #
    # c.add_chain(pred[torch.argwhere(torch.all(torch.isfinite(rpred), dim=1)).flatten(), :].cpu().numpy(),
    #             name="rRNPE")
    #
    # c.configure(smooth=0)
    # fig = c.plotter.plot(truth=fm_theta.cpu().numpy())
    # fig.suptitle(name + f" RBF: {rbf_dist.item():.2e}, r: {rrbf_dist.item():.2e}   |  Poly: {pol_dist.item():.2e}, r: {rpol_dist.item():.2e}")
    #
    # plt.show()

if __name__ == '__main__':
    plot_random_corner()
    # plot_random_spectra()