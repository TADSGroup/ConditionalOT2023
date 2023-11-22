import numpy as np
from typing import Callable
from tqdm.auto import tqdm
from scipy.special import logit

class gaussian_sampler():
    def __init__(
        self,
        mean,
        cov,
        batch_size = 5000,
        seed = None
    ) -> None:
        self.mean = mean
        self.cov = cov
        self.batch_size = 5000
        self.rng = np.random.default_rng(seed = seed)
        self.d = mean.shape[0]
        
        print("Computing eigendecomposition")
        eigs,P = np.linalg.eigh(cov)
        if np.min(eigs)<-1e-4:
            print("Significant negative component in eigenvalues, very degenerate")
        eigs = np.maximum(0,eigs)
        self.Tmat = np.diag(np.sqrt(eigs))@P.T
        self.current_batch = self.generate_batch(batch_size)
        self.current_position = 0
        
    def generate_batch(self,num_samples):
        white_samples = self.rng.standard_normal(size = (num_samples,self.d))
        return white_samples@self.Tmat+self.mean
    
    def next(self):
        if self.current_position>=self.batch_size:
            self.current_batch = self.generate_batch(self.batch_size)
            self.current_position = 0
        self.current_position += 1
        return self.current_batch[self.current_position - 1]

class pCN_sampler():
    def __init__(
        self,
        prior_sampler: gaussian_sampler,
        phi: Callable,
        dim:int,
        seed = None
    ) -> None:
        """
        prior_sampler: gaussian_sampler, we call next from it
        phi: potential function of likelihood
        dim: dimension of random variable
        """
        self.prior_sampler = prior_sampler
        self.phi = phi
        self.dim = dim
        self.rng = np.random.default_rng(seed = seed)
    
    def get_samples(
        self,
        num_samples,
        beta,
        u0
    ):
        """
        Gets num_samples with step size beta
        """
        u = u0.copy()
        u_samples = np.zeros((num_samples,self.dim))
        acceptance = np.zeros(num_samples)
        phi_vals = np.zeros(num_samples)
        phi_u = None

        for i in tqdm(range(num_samples)):
            u,accepted,phi_u = self.pCN_step(u,beta,phi_u)
            u_samples[i]=u
            acceptance[i]=accepted
            phi_vals[i]=phi_u
        return u_samples,acceptance,phi_vals
    
    def pCN_step(
        self,
        u,
        beta,
        phi_u = None
    ):
        """
        Takes one step of pCN with step size beta
        """
        v = np.sqrt(1-beta**2) * u + beta * self.prior_sampler.next()
        if phi_u == None:
            phi_u = self.phi(u)

        phi_v = self.phi(v)
        accept_prob = np.minimum(1,np.exp(phi_u-phi_v))
        is_accepted = self.rng.binomial(1,accept_prob)
        u_next = is_accepted*v + (1-is_accepted)*u
        phi_next = is_accepted*phi_v + (1-is_accepted)*phi_u
        return u_next , is_accepted, phi_next
    
    def burn_in_tune_beta(
        self,
        u0,
        beta_initial,
        target_acceptance,
        num_samples,
        long_alpha = 0.99,
        short_alpha = 0.9,
        adjust_rate = 0.01,
        beta_min = 1e-3,
        beta_max = 1-1e-3,
    ):
        u = u0.copy()
        u_samples = np.zeros((num_samples,self.dim))
        acceptance = np.zeros(num_samples)

        beta = beta_initial
        long_average = target_acceptance
        short_average = target_acceptance
        beta_history = []
        weight_history = []
        phi_vals = []
        phi_u = None

        for i in tqdm(range(num_samples)):
            u,accepted,phi_u = self.pCN_step(u,beta,phi_u)
            u_samples[i]=u
            acceptance[i]=accepted
            long_average = long_alpha * long_average + accepted*(1-long_alpha)
            short_average = short_alpha * long_average + accepted*(1-short_alpha)

            adjusted_beta = (
                beta + 
                adjust_rate * (long_average - target_acceptance) + 
                (0.5)*adjust_rate * (short_average - target_acceptance)
            )
            beta = np.clip(adjusted_beta,beta_min,beta_max)
            beta_history+=[beta]
            weight_history+=[long_average]
            phi_vals+=[phi_u]
        return u_samples,beta_history,acceptance,weight_history,phi_vals