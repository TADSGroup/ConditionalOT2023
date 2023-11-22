import numpy as np
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm

class PCA_transformer():
    def __init__(self,cov_mat,rank):
        self.cov_mat=cov_mat
        self.rank=rank
        eigenvalues,eigenvectors=np.linalg.eigh(cov_mat)
        eigenvectors=eigenvectors[:,::-1]
        eigenvalues=eigenvalues[::-1]
        self.sigma=eigenvalues
        self.P_full=eigenvectors
        self.P=eigenvectors[:,:rank]
        self.uncertainty_mat=self.P_full[:,rank:]@np.diag(self.sigma[rank:])@self.P_full[:,rank:].T
    
    def transform(self,x):
        variance_scaling=np.diag(1/np.sqrt(self.sigma[:self.rank]))
        if len(x.shape)==2:
            return x@self.P@variance_scaling
        else:
            return variance_scaling@self.P.T@x
    
    def inverse_transform(self,v):
        inv_variance_scaling = np.diag(np.sqrt(self.sigma[:self.rank]))
        if len(v.shape)==2:
            return v@inv_variance_scaling@self.P.T
        else:
            return self.P@inv_variance_scaling@v
        
class Normalizer():
    def __init__(self,data,eps=1e-5):
        self.eps=eps
        self.mean=np.mean(data,0,keepdims=True)
        self.std=np.std(data,0,keepdims=True)
    def transform(self,data):
        return (data-self.mean)/(self.std+self.eps)
    
    def inverse_transform(self,data):
        return data*(self.std+self.eps)+self.mean


def get_darcy_data_transformers(
        latent_dim=25,
        num_observation_grid=8,
        noise_level_y_observed=1e-3,
        num_datapoints = 10000,
        path_prefix = 'data'
        ):
    y_observed_raw=np.load(f"{path_prefix}/X_observed.npy")
    x_fields=np.log(np.load(f"{path_prefix}/true_permeability_fields.npy"))
    K_prior=np.load(f"{path_prefix}/kernel_matrix_prior.npy")
    pca=PCA_transformer(K_prior,rank=latent_dim)

    #subset data for x here
    x_data=pca.transform(x_fields[:num_datapoints].reshape(-1,40*40))

    obs_grid=np.linspace(0,1,num_observation_grid+2)[1:-1]
    original_sampling_grid=np.linspace(0,1,y_observed_raw.shape[1])
    print("Resampling observation arrays")
    y_observed=np.array(
        [
            RectBivariateSpline(
                original_sampling_grid,original_sampling_grid,y
                )(obs_grid,obs_grid).reshape(num_observation_grid**2)
            for y in tqdm(y_observed_raw[:num_datapoints])#Subset data for y here
        ]
    )
    y_Normalizer=Normalizer(y_observed)
    y_normalized = y_Normalizer.transform(y_observed)
    y_data = y_normalized + noise_level_y_observed*np.random.normal(size = y_normalized.shape)

    return y_data,x_data,pca,y_Normalizer
