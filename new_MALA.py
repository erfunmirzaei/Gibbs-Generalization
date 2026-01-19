import torch
from torch.optim import Optimizer
import numpy as np

class MALA(Optimizer):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA).
    
    A Markov Chain Monte Carlo (MCMC) sampler that uses Langevin dynamics to propose 
    new states and a Metropolis-Hastings step to ensure exact convergence to the posterior.
    
    Args:
        params (Iterable): Iterable of parameters to optimize.
        lr (float): Step size (epsilon). Controls the proposal variance.
            - If too large: Acceptance rate drops to 0.
            - If too small: Acceptance rate goes to 1, but exploration is slow.
        sigma_gauss_prior (float, optional): Standard deviation of Gaussian prior. 
            Equivalent to L2 regularization. Defaults to 0.1.
    
    Attributes:
        acceptance_rate (float): Exponential moving average of acceptance probability.
            Target range: 0.4 - 0.7.
    """

    def __init__(self, params, lr: float = 1e-2, sigma_gauss_prior: float = 0.1, beta: float = 1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if sigma_gauss_prior <= 0.0:
            raise ValueError(f"Invalid prior sigma: {sigma_gauss_prior}")

        defaults = dict(lr=lr, sigma_gauss_prior=sigma_gauss_prior, beta = beta)
        super(MALA, self).__init__(params, defaults)
        
        # Track acceptance rate (Exponential Moving Average)
        self.acceptance_rate = 0.5 
        self.ema_decay = 0.99 # Smoothing factor for the rate

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single MCMC step.
        
        Args:
            closure (callable): A closure that re-evaluates the model 
                and returns the loss. 
        
        Returns:
            loss (float): The loss value of the current state (after accept/reject).
        """
        assert closure is not None, "MALA requires a closure to calculate the Metropolis ratio."

        # 1. Capture Current State (t)
        # We need the gradient of the potential U(theta_t)
        # U(theta) = Loss(Data) + Loss(Prior)
        # Enable gradients for the closure (since we're inside @torch.no_grad())
        with torch.enable_grad():
            loss_t = float(closure())
        
        # Store State t (Params and Grads) to allow rollback if rejected
        params_t = []
        grads_t = []
        
        # Log-Transition Probability inputs
        log_q_forward = 0.0  # q(theta* | theta_t)
        log_q_backward = 0.0 # q(theta_t | theta*)
        
        # --- PROPOSAL GENERATION ---
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']    
            sigma = group['sigma_gauss_prior']
            # Prior Weight Decay Term: lambda = 1/sigma^2
            weight_decay = 1 / (sigma * sigma * beta) if beta > 0 else 1/(sigma * sigma)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Snapshot current state
                params_t.append(p.data.clone())
                
                # Calculate full Gradient of Potential U:
                # Grad_U = Grad_Likelihood (p.grad) + Grad_Prior (lambda * p)
                grad_u_t = p.grad.data + weight_decay * p.data
                grads_t.append(grad_u_t.clone()) # Save for backward calculation
                
                # Langevin Dynamics Proposal:
                # theta* = theta_t - (eps/2)*Grad_U + sqrt(eps)*noise
                noise = torch.randn_like(p.data)
                mean_forward = p.data - 0.5 * lr * grad_u_t
                
                # Update p.data to proposed theta*
                p.data.copy_(mean_forward + np.sqrt(lr/beta) * noise)
                
                # Accumulate Forward Transition Probability (q)
                # q(theta* | theta_t) propto exp( - ||theta* - mean_forward||^2 / 2eps )
                # Since theta* - mean_forward = sqrt(eps)*noise, 
                # The exponent is: - (eps * ||noise||^2) / (2eps) = -0.5 * ||noise||^2
                log_q_forward += -0.5 * torch.sum(noise ** 2)

        # --- EVALUATE PROPOSAL (theta*) ---
        # Compute loss and gradients at theta*
        # Enable gradients for the closure (since we're inside @torch.no_grad())
        with torch.enable_grad():
            loss_star = float(closure())
        
        # --- METROPOLIS-HASTINGS RATIO ---
        idx = 0
        prior_nll_t = 0.0
        prior_nll_star = 0.0
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']    
            sigma = group['sigma_gauss_prior']
            # Prior Weight Decay Term: lambda = 1/sigma^2
            weight_decay = 1 / (sigma * sigma * beta) if beta > 0 else 1/(sigma * sigma)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Retrieve saved state
                p_old = params_t[idx]
                
                # Calculate Prior NLL (Negative Log Likelihood) for Energy U
                # Prior NLL = p^2 / (2 * sigma^2)
                prior_nll_t += 0.5 * weight_decay * torch.sum(p_old ** 2)
                prior_nll_star += 0.5 * weight_decay * torch.sum(p.data ** 2)
                
                # Calculate Backward Transition Kernel q(theta_t | theta*)
                # We need Grad_U at theta* (which is currently in p.grad)
                grad_u_star = p.grad.data + weight_decay * p.data
                
                # Mean of the reverse proposal
                mean_backward = p.data - 0.5 * lr * grad_u_star
                
                # Distance from "Mean Backward" to "Old State"
                diff = p_old - mean_backward
                log_q_backward += -0.5 * beta * torch.sum(diff ** 2) / lr
                
                idx += 1
                
        # Total Potential Energy U = Data Loss + Prior Loss
        U_t = beta * (loss_t + prior_nll_t)
        U_star = beta * (loss_star + prior_nll_star)
        
        # Metropolis Acceptance Probability
        # log(alpha) = -U(theta*) - log_q(theta*|theta_t) + U(theta_t) + log_q(theta_t|theta*)
        # (Note: signs are tricky. Standard ratio is p(x*)/p(x) * q(x|x*)/q(x*|x))
        # Log version: log p(x*) - log p(x) + log q(x|x*) - log q(x*|x)
        # Since p(x) = exp(-U(x)), log p(x) = -U(x)
        log_alpha = (-U_star) - (-U_t) + log_q_backward - log_q_forward
        self.alpha = torch.exp(log_alpha)
        # --- ACCEPT / REJECT ---
        accepted = False
        if torch.log(torch.rand(1)) < log_alpha:
            accepted = True
            final_loss = loss_star
        else:
            # Reject: Rollback parameters to theta_t
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    p.data.copy_(params_t[idx])
                    # We usually don't need to restore p.grad, but for safety in some loops:
                    if p.grad is not None:
                        p.grad.data.copy_(grads_t[idx]) 
                    idx += 1
            final_loss = loss_t

        # Update Acceptance Rate (EMA)
        self.acceptance_rate = (self.ema_decay * self.acceptance_rate) + \
                               ((1 - self.ema_decay) * float(accepted))
                               
        return final_loss
    
    def set_step_size(self, new_lr: float):
        """Update the step size (learning rate) for all parameter groups."""
        for group in self.param_groups:
            group['lr'] = new_lr
        

class StepSizeTuner:
    def __init__(self, initial_step_size, target_accept=0.57, decay_rate = 0.75):
        self.step_size = initial_step_size
        self.target = target_accept
        self.k = decay_rate      # Decay rate
        
        # Hyperparameters for the adaptation (Robbins-Monro)
        self.log_step = np.log(initial_step_size)
        self.t = 0         # Iteration counter
        
    def update(self, current_acceptance_prob):
        self.t += 1
        
        # Calculate the error (difference from target)
        # If acceptance is HIGH, we want to INCREASE step size (error is positive)
        # If acceptance is LOW, we want to DECREASE step size (error is negative)
        diff = current_acceptance_prob - self.target
        
        # Compute learning rate for the hyperparameter
        learning_rate = self.t ** (-self.k)
        
        # Update the log step size
        self.log_step = self.log_step + learning_rate * diff
        
        # Recover real step size
        self.step_size = np.exp(self.log_step)
        
        return self.step_size