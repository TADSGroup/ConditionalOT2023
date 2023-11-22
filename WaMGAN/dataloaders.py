from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset

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


def get_darcy_dataloader_transformers(
        batch_size=128,
        latent_dim=25,
        num_observation_grid=8,
        noise_level_y_observed=1e-3,
        num_datapoints = 100000,
        shuffle = True,
        path_prefix = 'data'
        ):
    y_observed_raw=np.load(path_prefix + "/X_observed.npy")
    x_fields=np.log(np.load(path_prefix + "/true_permeability_fields.npy"))
    K_prior=np.load(path_prefix + "/kernel_matrix_prior.npy")
    pca=PCA_transformer(K_prior,rank=latent_dim)

    #subset data for x here
    x_pca=pca.transform(x_fields[:num_datapoints].reshape(-1,40*40))
    x_data=torch.tensor(x_pca,dtype=torch.float)

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
    # Add a uniform noise first before centering/normalizing
    y_observed = y_observed + noise_level_y_observed*np.random.standard_normal(size = y_observed.shape)
    y_Normalizer=Normalizer(y_observed)
    y_normalized = y_Normalizer.transform(y_observed)
    y_data=torch.tensor(y_normalized,dtype=torch.float)
    print("Rough Ratio SDnoise/SDsignal:")
    print(np.mean(noise_level_y_observed/y_Normalizer.std))

    #y_data = y_data + noise_level_y_observed*torch.randn(y_data.shape)

    dataset = TensorDataset(y_data[:num_datapoints],x_data[:num_datapoints])
    xy_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return xy_loader,pca,y_Normalizer


def get_bimodal_1d_dataloader(batch_size=128,num_datapoints=128*50):
    """
    Bimodal model without conditioning: 
    y is just 0
    x is a 50/50 mixture of N(-2,1), N(2,1)
    """
    data_cond = np.zeros((num_datapoints,1))
    data_gen = np.random.choice(np.array([-1,1]),(num_datapoints,1)) * (np.random.randn(num_datapoints,1)+2)
    dataset = TensorDataset(torch.Tensor(data_cond), torch.Tensor(data_gen))
    return DataLoader(dataset,batch_size=batch_size,shuffle=True)


def get_quadratic_dataloader(batch_size=128,num_datapoints=128*50):
    """
    Quadratic model: 
    Perform inference on x after observing y
        x = y^2 + 0.5 * eps
        eps ~ N(0,1)
        y ~ N(0,1)
    """
    data_cond = np.random.randn(num_datapoints,1)
    eps=0.5 * np.random.randn(num_datapoints,1)
    data_gen = data_cond**2 + eps
    dataset = TensorDataset(torch.Tensor(data_cond), torch.Tensor(data_gen))
    return DataLoader(dataset,batch_size=batch_size,shuffle=True)


def get_simple_dataloader(batch_size=128,num_datapoints=128*50):
    """
    Linear model: 
    Perform inference on x after observing y
        y = 2*x+0.5*eps - 1
        eps ~ N(0,1)
        x ~ N(0,1)
    """
    #data_gen =np.random.uniform(-3,3,size=(num_datapoints,1))
    data_gen = np.random.randn(num_datapoints,1)
    eps=np.random.randn(num_datapoints,1)
    data_cond = 2*data_gen+0.5*eps - 1
    dataset = TensorDataset(torch.Tensor(data_cond), torch.Tensor(data_gen))
    return DataLoader(dataset,batch_size=batch_size,shuffle=True)

def bimodal_pushforward(y,v):
    return np.sign(v)*np.maximum(np.arctan(np.abs(10*y*v)),np.abs(v)) + 5*y

def get_bimodal_samples(num_target,reference_multiplier = 1, seed = 10):
    """
    We match the marginal on the part that we're conditioning on
    """
    rand=np.random.default_rng(seed=10)
    cond_part = rand.uniform(0,1,(num_target,1))

    target=bimodal_pushforward(cond_part[:,0],rand.standard_normal(num_target)).reshape(num_target,1)
    return cond_part,target

def get_bimodal_dataloader(batch_size = 128,num_datapoints = 128*50,seed = 10):
    cond_part,target = get_bimodal_samples(num_datapoints,seed = seed)

    dataset = TensorDataset(torch.Tensor(cond_part), torch.Tensor(target))
    return DataLoader(dataset,batch_size=batch_size,shuffle=True)


def get_mnist_dataloaders(batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST('../data', train=False,
                               transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST('../fashion_data', train=False,
                                      transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_lsun_dataloader(path_to_data='../lsun', dataset='bedroom_train',
                        batch_size=64):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Get dataset
    lsun_dset = datasets.LSUN(db_path=path_to_data, classes=[dataset],
                              transform=transform)

    # Create dataloader
    return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)
