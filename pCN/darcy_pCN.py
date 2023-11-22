import numpy as np
import pCN_sampler as pcn
import darcySolver as ds
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm
from scipy.spatial import distance_matrix
import os
os.nice(10)
import wandb

import argparse
import json

parser = argparse.ArgumentParser(description='Read configuration file')
parser.add_argument('config_file', type=str, help='Path to the configuration file')
args = parser.parse_args()

with open(args.config_file, 'r') as f:
    configs = json.load(f)

run = wandb.init(
    # set the wandb project where this run will be logged
    project = 'darcy_MCMC_wider_bw',
    config=configs
)

def interpolate_for_fenics(xgrid,ygrid,function_values):
    interpolated_permeability_field=RectBivariateSpline(
        kernel_x_grid,
        kernel_y_grid,
        function_values,
        kx=2,
        ky=2
        )
    return lambda x:interpolated_permeability_field(x[0],x[1],grid=False)

def constant_rhs(x):
    return x[0]*0+1

def get_solver(
    observation_x_vals,#assume we observe on a regular grid
    observation_y_vals,
    num_cells = 40,
    f_rhs = constant_rhs,
    ):
    """
    Builds another wrapper for darcy solver
    """
    darcy=ds.DarcySolver(num_x_cells=num_cells,num_y_cells=num_cells)
    def solve(permeability_function,):
        sol=darcy.solve(
            permeability=permeability_function,
            f_rhs=f_rhs,
            )
        sol_observed=sol(observation_x_vals,observation_y_vals)
        return sol_observed
    return solve

def get_phi(
    observed_values,
    noise_level,
    grid_num_observed,
    kernel_x_grid,
    kernel_y_grid,
    ):
    observation_x_vals = np.linspace(0,1,grid_num_observed+2)[1:-1]
    observation_y_vals = np.linspace(0,1,grid_num_observed+2)[1:-1]
    solve_darcy = get_solver(
        observation_x_vals,
        observation_y_vals,
        )

    def phi(permeability_field_values):
        permeability_func = interpolate_for_fenics(
            kernel_x_grid,
            kernel_y_grid,
            np.exp(permeability_field_values).reshape(kernel_points,kernel_points)
            )
        solution = solve_darcy(
            permeability_func
            )
        return np.sum(((solution-observed_values)/noise_level)**2)/2
    return phi


def matern_three_half(rho):
    def k(d):
        return (1+np.sqrt(3)*d/rho)*np.exp(-np.sqrt(3)*d/rho)
    return k

def thin_and_summarize(A,thin_rate = 100):
    mean = np.mean(A,axis=0)
    cov = np.cov(A.T)
    A_thinned = A[::thin_rate]
    return A_thinned,mean,cov


kernel_points = 40
kernel_function_d = matern_three_half(configs['prior_kernel_bandwidth'])

kernel_x_grid=np.linspace(0,1,kernel_points)
kernel_y_grid=np.linspace(0,1,kernel_points)
points=np.array([[x,y] for x in kernel_x_grid for y in kernel_y_grid])
D=distance_matrix(points,points)
KMat=kernel_function_d(D)

prior_sampler = pcn.gaussian_sampler(mean = np.zeros(len(KMat)),cov = KMat)


y_observed_raw=np.load("data/X_observed.npy")
x_fields=np.log(np.load("data/true_permeability_fields.npy"))

example_index = configs["example_index"]
observed_pressure = y_observed_raw[example_index]
true_field = x_fields[example_index]

save_folder = configs["name"]

np.save(f"MCMC_results/{save_folder}/pressure_data.npy",observed_pressure)
np.save(f"MCMC_results/{save_folder}/true_permeability.npy",true_field)


grid_num_observed = configs['grid_num_observed']
obs_grid = np.linspace(0,1,grid_num_observed+2)[1:-1]

solve = get_solver(obs_grid,obs_grid,num_cells = 40)
permeability = interpolate_for_fenics(
    kernel_x_grid,
    kernel_y_grid,
    np.exp(true_field).reshape(kernel_points,kernel_points)
    )
observed_values = solve(permeability)


phi = get_phi(
    observed_values = observed_values,
    noise_level = configs['noise_level'],
    grid_num_observed = grid_num_observed,
    kernel_x_grid = kernel_x_grid,
    kernel_y_grid = kernel_y_grid
    )

mcmc_sampler = pcn.pCN_sampler(
    prior_sampler,
    phi,
    kernel_points**2
)

burnin_samples,beta_history,acceptance,weight_history,phi_vals = mcmc_sampler.burn_in_tune_beta(
        prior_sampler.next(),
        beta_initial = 0.03,
        target_acceptance = 0.3,
        num_samples = 10000,
        long_alpha = 0.99,
        short_alpha = 0.9,
        adjust_rate = 0.03,
        beta_min = 1e-3,
        beta_max = 1-1e-3,
    )


np.save(f"MCMC_results/{save_folder}/burnin_samples.npy",burnin_samples)
np.save(f"MCMC_results/{save_folder}/burnin_acceptance.npy",acceptance)
np.save(f"MCMC_results/{save_folder}/burnin_phi.npy",phi_vals)


thinning_rate = configs['thin_rate']



beta_val = np.mean(beta_history[-500:])
u_initial = burnin_samples[-1]

num_batches = configs["num_batches"]
for i in range(num_batches):
    u_samples,acceptance,phi_vals = mcmc_sampler.get_samples(
        num_samples = int(20000),
        beta = beta_val,
        u0 = u_initial
    )
    path_prefix = f"MCMC_results/{save_folder}/batch_{f'{000+i}'.zfill(3)}"
    u_thinned,mean,cov = thin_and_summarize(u_samples,thin_rate = thinning_rate)


    np.save(path_prefix + "u_samples.npy",u_thinned)
    np.save(path_prefix + "acceptance.npy",acceptance)
    np.save(path_prefix + "phi_vals.npy",phi_vals)
    np.save(path_prefix + "mean.npy",mean)
    np.save(path_prefix + "cov.npy",cov)

    u_initial = u_samples[-1]
    wandb.log({"percent_done":(i+1)/num_batches})
    wandb.log({"acceptance_rate":np.mean(acceptance)})
    for val in phi_vals[::thinning_rate]:
        wandb.log({"phi":val})
