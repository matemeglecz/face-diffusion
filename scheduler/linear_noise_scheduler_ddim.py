import torch
import numpy as np


class LinearNoiseSchedulerDDIM:
    r"""
    Class for the linear noise scheduler that is used in DDPM and can also support DDIM.
    """
    
    def __init__(self, num_timesteps, beta_start, beta_end, ddim_eta=0.0):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.ddim_eta = ddim_eta  # eta controls stochasticity; set to 0 for deterministic DDIM
        
        # Mimicking how compvis repo creates schedule
        self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        )
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    def sample_prev_timestep(self, xt, noise_pred, t, t_prev):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted with DDIM sampling
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :param t_prev: previous timestep in the sampling process
        :return:
        """

        
        # Get the values for the current and previous timestep alphas
        alpha_t = self.alpha_cum_prod.to(xt.device)[t]
        alpha_t_prev = self.alpha_cum_prod.to(xt.device)[t_prev]

        # Compute the predicted x_0 using the model's noise prediction
        x0 = (xt - self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred) / self.sqrt_alpha_cum_prod.to(xt.device)[t]
        #x0 = torch.clamp(x0, -1., 1.)

        # DDIM equation to sample xt_prev (deterministic path)
        xt_prev = (
            torch.sqrt(alpha_t_prev) * x0 +  # First part, weighted by previous timestep's alpha
            torch.sqrt(1 - alpha_t_prev) * noise_pred  # Second part
        )


        '''
        # Get the values for the current and previous timestep alphas
        alpha_t = self.alpha_cum_prod.to(xt.device)[t]
        alpha_t_prev = self.alpha_cum_prod.to(xt.device)[t_prev]

        # Compute the predicted x_0 using the model's noise prediction
        x0 = (xt - self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred) / self.sqrt_alpha_cum_prod.to(xt.device)[t]
        x0 = torch.clamp(x0, -1., 1.)

        # DDIM equation to sample xt_prev (deterministic path)
        xt_prev = (
            torch.sqrt(alpha_t_prev) * x0 +  # First part, weighted by previous timestep's alpha
            torch.sqrt(1 - alpha_t_prev) * (xt - torch.sqrt(alpha_t) * x0) / torch.sqrt(1 - alpha_t)  # Second part
        )
        '''
        
        
        # If DDIM is not fully deterministic, add noise according to eta
        if self.ddim_eta > 0:
            variance = (1 - alpha_t_prev) / (1 - alpha_t) * self.betas.to(xt.device)[t]
            sigma = self.ddim_eta * torch.sqrt(variance)
            noise = torch.randn_like(xt)
            xt_prev = xt_prev + sigma * noise

        return xt_prev, x0
    
