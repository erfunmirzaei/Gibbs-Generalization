import torch
from torch.optim import Optimizer
import numpy as np

class MALA(Optimizer):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA).
    
    MALA is a Markov Chain Monte Carlo (MCMC) method that uses Langevin dynamics 
    to propose new states and a Metropolis-Hastings step to accept or reject them. 
    Unlike SGLD, MALA theoretically converges to the exact posterior distribution 
    (assuming full-batch gradients).

    Args:
        params (Iterable): Iterable of parameters to optimize.
        lr (float): Step size (epsilon) for the Langevin dynamics.
        sigma_gauss_prior (float, optional): Standard deviation of the Gaussian prior. 
            Equivalent to weight decay. Defaults to 0.1.
        beta (float, optional): Inverse temperature. 1.0 corresponds to the standard 
            posterior. < 1.0 flattens the distribution (tempering). Defaults to 1.0.

    Reference:
        Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of 
        Langevin distributions and their discrete approximations.
    """

    def __init__(self, params, lr: float = 1e-2, sigma_gauss_prior: float = 0.1, beta: float = 1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if sigma_gauss_prior <= 0.0:
            raise ValueError(f"Invalid prior sigma: {sigma_gauss_prior}")

        # Weight decay corresponds to the gradient of the log-prior: -log p(theta) = theta^2 / (2*sigma^2)
        # Gradient is theta / sigma^2.
        weight_decay = 1.0 / (sigma_gauss_prior ** 2)
        
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta)
        super(MALA, self).__init__(params, defaults)
        # Track acceptance rate (Exponential Moving Average)
        self.acceptance_rate = 0.5 
        self.ema_decay = 0.95 # Smoothing factor for the rate

        
    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step (MCMC transition).
        
        Args:
            closure (callable): A closure that re-evaluates the model, 
                computes gradients, and returns the loss. 
                REQUIRED for MALA to calculate the acceptance ratio.
        
        Returns:
            loss (float): The loss value of the accepted state.
        """
        assert closure is not None, "MALA requires a closure to calculate the acceptance probability."

        # 1. Calculate current state (theta_t) properties
        # We need the gradient at the current state. 
        # Note: We assume the user has NOT called backward() yet, or we re-call it 
        # to ensure consistency. To be safe, we call the closure to get current Loss and Grads.
        loss_t = closure() 
        loss_t = float(loss_t)

        # Store current parameters and gradients for potential rollback
        params_t = []
        grads_t = []
        
        # We need to compute the 'log q(theta* | theta_t)' and 'log q(theta_t | theta*)'
        # To do this, we need to collect the full proposal vectors.
        log_q_forward = 0.0
        log_q_backward = 0.0
        
        # --- PROPOSAL STEP ---
        # Generate proposal theta* using Langevin dynamics
        # theta* = theta_t - lr * (0.5 * grad_U(theta_t)) + sqrt(lr) * noise
        
        # We store the noise and gradients to compute transition probs later
        sq_lr = None # Will be set in loop
        
        for group in self.param_groups:
            eps = group['lr']
            beta = group['beta']
            decay = group['weight_decay']
            sq_lr = np.sqrt(eps)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Save state t
                params_t.append(p.clone())
                
                # Full gradient U = Gradient_Likelihood + Gradient_Prior
                # closure() gives Gradient_Likelihood. We add Gradient_Prior manually.
                d_p = p.grad.data + decay * p.data
                grads_t.append(d_p.clone()) # Save grad t

                # Generate Noise
                noise = torch.randn_like(p.data)
                
                # Proposal Update: theta*
                # Mean_forward = theta_t - (eps/2) * d_p
                # theta* = Mean_forward + sqrt(eps) * noise / sqrt(beta) 
                # (Standard MALA assumes beta=1 for exact sampling, but we scale noise for tempering)
                
                # Note: SGLD uses 0.5 * gradient. MALA standard discretization is:
                # x_new = x - (eps/2)*grad + sqrt(eps)*z.
                mean_forward = p.data - 0.5 * eps * d_p
                
                # Apply update to p.data in-place to form theta*
                p.data.copy_(mean_forward + sq_lr * noise)

                # Contribution to forward transition log probability
                # q(theta* | theta_t) \propto exp( - || theta* - mean_forward ||^2 / (2*eps) )
                # || theta* - mean_forward ||^2 = || sqrt(eps) * noise ||^2 = eps * ||noise||^2
                # term = - (eps * ||noise||^2) / (2*eps) = -0.5 * ||noise||^2
                # This is equivalent to log prob of the gaussian noise.
                # However, for the ratio, we compute || theta_new - (theta_old - eps/2 grad_old) ||^2
                
                # Let's do it explicitly to avoid confusion:
                # diff = theta* - mean_forward
                # log_q_forward += -0.5 * torch.sum(diff**2) / eps 
                # Since diff = sqrt(eps)*noise, diff^2/eps = noise^2
                log_q_forward += -0.5 * torch.sum(noise ** 2)

        # --- EVALUATE PROPOSAL ---
        # Now parameters are at theta*. We need gradients at theta* for the backward transition.
        loss_star = closure()
        loss_star = float(loss_star)
        
        # --- ACCEPTANCE STEP ---
        for i, group in enumerate(self.param_groups):
            eps = group['lr']
            decay = group['weight_decay']
            
            # Re-iterate over params to calculate backward transition probability
            # q(theta_t | theta*)
            # Mean_backward = theta* - (eps/2) * grad_U(theta*)
            
            # Because we iterated groups/params in order, we can pop from our saved lists 
            # (using an index tracker would be cleaner, but flat iteration works if order is deterministic)
            # A simple flat iterator is safer here.
            pass 

        # Let's redo the loop structure to be safer with the lists
        flat_params_t = params_t
        flat_grads_t = grads_t
        idx = 0
        
        for group in self.param_groups:
            eps = group['lr']
            decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                p_star = p.data # This is currently theta*
                p_old = flat_params_t[idx]
                grad_old = flat_grads_t[idx]
                
                # Current grad at theta* (from loss_star closure call) + Prior
                grad_star = p.grad.data + decay * p_star
                
                # Backward Mean: theta* - (eps/2) * grad_star
                mean_backward = p_star - 0.5 * eps * grad_star
                
                # Calculate || theta_t - mean_backward ||^2
                diff = p_old - mean_backward
                log_q_backward += -0.5 * torch.sum(diff**2) / eps
                
                idx += 1

        # Metropolis-Hastings Ratio
        # log_alpha = log p(theta*) + log q(theta_t | theta*) - log p(theta_t) - log q(theta* | theta_t)
        # Note: Loss = - log p(theta). So log p(theta) = -Loss
        
        # We must also include the Prior in the Loss terms if the closure didn't included it.
        # Usually closures calculate Data Loss. We calculated gradients of prior manually.
        # We should add Prior NLL to loss_t and loss_star to get full potential U.
        # U(theta) = Loss_Data + Weight_Decay_Term
        # decay_term = 0.5 * weight_decay * ||theta||^2
        
        prior_nll_t = 0.0
        prior_nll_star = 0.0
        idx = 0
        for group in self.param_groups:
            decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                prior_nll_t += 0.5 * decay * torch.sum(flat_params_t[idx]**2)
                prior_nll_star += 0.5 * decay * torch.sum(p.data**2)
                idx += 1
                
        U_t = loss_t + prior_nll_t
        U_star = loss_star + prior_nll_star
        
        log_alpha = (-U_star + log_q_backward) - (-U_t + log_q_forward)
        
        # Accept or Reject
        # We accept if log(uniform(0,1)) < log_alpha
        if torch.log(torch.rand(1)) < log_alpha:
            # Accept: The parameters are already set to theta*. 
            # Return new loss.
            return loss_star
        else:
            # Reject: Revert parameters to theta_t
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    p.data.copy_(flat_params_t[idx])
                    # Optionally revert gradients too, though standard optimizers usually 
                    # don't promise valid grads after step()
                    p.grad.data.copy_(flat_grads_t[idx]) 
                    idx += 1
            return loss_t