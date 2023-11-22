import numpy as np
import darcySolver as ds
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
rng=np.random.default_rng(seed=101)

x_grid=np.linspace(0,1,100)
y_grid=np.linspace(0,1,100)
X,Y=np.meshgrid(x_grid,y_grid)

def matern_one_half(rho):
    def k(d):
        return np.exp(-d/rho)
    return k

def matern_three_half(rho):
    def k(d):
        return (1+np.sqrt(3)*d/rho)*np.exp(-np.sqrt(3)*d/rho)
    return k
    
def matern_five_half(rho):
    def k(d):
        return (1+np.sqrt(5)*d/rho+np.sqrt(5)*d**2/(rho**2))*np.exp(-np.sqrt(5)*d/rho)
    return k


def sample_permeability_fields(
        kernel_function_d,
        num_samples,
        rank=50,
        kernel_points=25,
        scaling=1
        ):
    kernel_x_grid=np.linspace(0,1,kernel_points)
    kernel_y_grid=np.linspace(0,1,kernel_points)


    points=np.array([[x,y] for x in kernel_x_grid for y in kernel_y_grid])
    D=np.array([np.linalg.norm(p1-p2) for p1 in points for p2 in points]).reshape(kernel_points**2,kernel_points**2)

    KMat=kernel_function_d(D)
    eig,P=np.linalg.eigh(KMat)
    #Set them in decreasing order for convenience
    eig=eig[::-1]
    P=P[:,::-1]
    eig_reduced=eig.copy()
    eig_reduced[rank:]=0


    gp_samples=rng.multivariate_normal(np.zeros(len(D)),cov=KMat,size=num_samples)
    fields=[np.exp(scaling*gp_sample).reshape(kernel_points,kernel_points) for gp_sample in gp_samples]

    return kernel_x_grid,kernel_y_grid,fields,KMat


def constant_rhs(x):
    return x[0]*0+1

def make_datapoint(
        permeability_xgrid,
        permeability_ygrid,
        permeability_field,
        num_eval=100,
        f_rhs=constant_rhs
        ):

    interpolated_permeability_field=RectBivariateSpline(
        permeability_xgrid,
        permeability_ygrid,
        permeability_field,
        kx=2,
        ky=2
        )

    permeability_function=lambda x:interpolated_permeability_field(x[0],x[1],grid=False)

    num_cells=100
    darcy=ds.DarcySolver(num_x_cells=num_cells,num_y_cells=num_cells)


    sol=darcy.solve(
        permeability=permeability_function,
        f_rhs=f_rhs
        )
    
    x_obs=np.linspace(0,1,num_eval)
    y_obs=np.linspace(0,1,num_eval)
    Xobs,Yobs=np.meshgrid(x_obs,y_obs)
    sol_observed=sol(Xobs,Yobs,grid=False)
    return sol_observed

params="""
n=100000
num_kernel_points=40
kernel = matern_three_half(0.5)
eval_points = 100
"""

n=100
num_kernel_points=40
print("Sampling Random Fields")
xgrid,ygrid,permeability_fields,K=sample_permeability_fields(
    matern_three_half(0.5),
    n,
    kernel_points=num_kernel_points
    )
print("Finished Sampling")

print("Solving Darcy Flow on each Field")
observations=[
    make_datapoint(
    xgrid,
    ygrid,
    field,
    num_eval=100
    ) for field in tqdm(permeability_fields)
]

X_data_observed=np.array(observations)
print("Finished Solving")
print("Saving Results")

with open('data/params.txt', 'w') as f:
    f.write(params)

np.save("data/X_observed.npy",X_data_observed)
np.save("data/kernel_matrix_prior.npy",K)
np.save("data/true_permeability_fields.npy",np.array(permeability_fields))

