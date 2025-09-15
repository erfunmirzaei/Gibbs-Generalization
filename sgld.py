"""
SGLD (Stochastic Gradient Langevin Dynamics) optimizer implementation.

This module contains the SGLD optimizer used for Bayesian learning in the
Gibbs generalization bound experiments.
"""

import numpy as np
import torch
from torch.optim import optimizer
from typing import Generator, Iterable


class SGLD(optimizer.Optimizer):
    """Stochastic Gradient Langevin Dynamics (SGLD).
    An algorithm for Bayesian learning from large scale datasets. 

    Weight decay is specified in terms of the Gaussian prior's sigma.
    Inverse temperature (beta) controls the noise level in the SGLD updates.

    Welling and Teh, 2011. Bayesian Learning via Stochastic Gradient Langevin 
    Dynamics. Paper link: https://bit.ly/3ngnyRA

    Args:
        params (Iterable): an iterable of `torch.Tensor`s or
            `dict`s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
        lr (float): learning rate. 
        sigma_gauss_prior (float, optional): Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter. Defaults to 1.0.
        add_noise (bool, optional): Defaults to True. 
    
    Attributes:
        param_group (OptimizerParamGroup): Stores parameters in the param_group
            and stores a pointer to the OptimizerOptions. 
            docs: https://preview.tinyurl.com/2y272xmv

    Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of 
        dictionaries.
    """ 

    def __init__(self, params: Iterable, lr: float, 
                 sigma_gauss_prior: float = 0.1, beta: float = 1.0, add_noise: bool = True):
        if isinstance(sigma_gauss_prior, (complex)):
            if sigma_gauss_prior.imag != 0:
                raise ValueError(f"sigma_gauss_prior must be a real number.")

        weight_decay = 1 / (sigma_gauss_prior * sigma_gauss_prior * beta) if beta > 0 else 1/(sigma_gauss_prior * sigma_gauss_prior)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta, add_noise=add_noise)
        super(SGLD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Updates neural network parameters. Called once the gradients are 
        computed using loss.backward(). Performs a single parameter update.

        Args:
            closure (callable, optional): A closure that reevaluates the 
                model and returns the loss. 
        This function should not modify the gradient field of the parameters, 
            i.e. `parameter.grad`. 
        """
        loss = None
        def params() -> Generator:
            for param_group in self.param_groups:
                weight_decay = param_group['weight_decay']
                beta = param_group['beta']
                for param in param_group['params']:
                    yield param, weight_decay, beta, param_group
        
        # 'loss' gets updated from the following loop (under the hood)
        for param, weight_decay, beta, param_group in params():
            if param.grad is None:
                continue
            gradient = param.grad.data
            if weight_decay != 0:
                gradient.add_(param.data, alpha=weight_decay)
            if param_group['add_noise']:
                # Langevin noise scaled by inverse temperature: N(0, 1) / sqrt(lr * beta)
                # Handle beta = 0 case to avoid division by zero
                    # Use device-appropriate noise generation for better GPU performance
                    if param.data.is_cuda:
                        langevin_noise = torch.cuda.FloatTensor(param.data.size()).normal_(
                            mean=0, std=1) / np.sqrt(param_group['lr'] * beta)
                    else:
                        langevin_noise = param.data.new(param.data.size()).normal_(
                            mean=0, std=1) / np.sqrt(param_group['lr'] * beta)
                    param.data.add_(0.5*gradient + langevin_noise, alpha=-param_group['lr'])
                
            else: # don't add noise
                param.data.add_(0.5*gradient, alpha=-param_group['lr'])
        return loss
