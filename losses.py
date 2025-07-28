"""
Loss functions for the Gibbs generalization bound experiments.

This module contains loss function implementations used in the SGLD experiments,
including bounded cross-entropy loss and zero-one loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TangentLoss(nn.Module):
    """
    Tangent loss: phi(v) = (2 * arctan(v) - 1)^2
    where v = y * logits, with y in {-1, +1}
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: Tensor of shape (N,) or (N, 1)
        targets: Tensor of shape (N,) with values {+1, -1}
        """

        # Ensure targets are {-1, +1}
        y = targets.float()
        if y.min() == 0:
            y = y * 2 - 1

        v = logits.flatten() * y
        loss = (2 * torch.atan(v) - 1).pow(2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
class BoundedCrossEntropyLoss(nn.Module):
    def __init__(self, ell_max=4.0):
        super(BoundedCrossEntropyLoss, self).__init__()
        self.ell_max = ell_max
        self.e_neg_ell_max = torch.exp(torch.tensor(-ell_max))

    def forward(self, logits, target):
        """
        logits: Tensor of shape (batch_size, num_classes) or (batch_size, 1) for binary
        target: Tensor of shape (batch_size,) with integer class labels
        """
        if logits.shape[-1] == 1:
            # Binary classification with single output
            # Convert single logit to two-class probabilities
            p_pos = torch.sigmoid(logits.squeeze(-1))
            p_neg = 1 - p_pos
            probs = torch.stack([p_neg, p_pos], dim=1)
        else:
            # Multi-class case
            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=1)

        # Apply psi(p) transformation
        psi_probs = self.e_neg_ell_max + (1 - 2 * self.e_neg_ell_max) * probs

        # Gather psi(p_y) for the correct classes
        psi_p_y = psi_probs.gather(dim=1, index=target.long().unsqueeze(1)).squeeze(1)

        # Compute bounded cross entropy: -log(psi(p_y))
        loss = -torch.log(psi_p_y)

        # Ensure numerical stability
        loss = torch.clamp(loss, min=0.0, max=self.ell_max)

        return loss.mean()


class ZeroOneLoss(nn.Module):
    """
    Zero-one loss function for classification.
    
    The zero-one loss counts the fraction of misclassified examples:
    L_{0-1}(h) = (1/n) * sum_i I[h(x_i) != y_i]
    
    This is useful for evaluating actual classification performance
    but is not differentiable, so it's used only for evaluation.
    """
    
    def __init__(self):
        super(ZeroOneLoss, self).__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the zero-one loss (classification error rate).
        
        Args:
            logits (torch.Tensor): Model outputs (raw logits)
            targets (torch.Tensor): Target labels
            
        Returns:
            torch.Tensor: Classification error rate (scalar)
        """
        # Get predictions
        if logits.shape[-1] == 1:
            # Binary classification with single output
            predictions = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
        else:
            # Multi-class or binary with 2 outputs
            predictions = torch.argmax(logits, dim=-1).float()
        
        # Compute error rate
        errors = (predictions != targets).float()
        return errors.mean()
