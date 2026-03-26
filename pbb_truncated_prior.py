"""
Utilities for PBB-style truncated Gaussian priors with layer-wise scales.

This module is additive and optional. It provides:
1) Truncated Gaussian initialization with per-layer sigma based on fan-in.
2) Layer-wise sigma map for SGLD prior regularization.
3) Per-parameter optimizer groups with custom weight decay.

The default sigma rule follows the PBB initialization style:
    sigma_l = sigma_scale / sqrt(fan_in_l)
with truncation bounds +/- truncation * sigma_l.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn


def _fan_in_from_weight(weight: torch.Tensor) -> int:
    if weight.dim() < 2:
        return max(weight.numel(), 1)
    fan_in = weight.shape[1]
    if weight.dim() > 2:
        for dim in weight.shape[2:]:
            fan_in *= dim
    return max(int(fan_in), 1)


def _trunc_normal_(tensor: torch.Tensor, mean: float, std: float, a: float, b: float) -> torch.Tensor:
    # PyTorch supports trunc_normal_ in init for modern versions.
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def initialize_prior_truncated_gaussian(
    model: nn.Module,
    sigma_scale: float = 1.0,
    truncation: float = 2.0,
    seed: int = None,
    zero_bias: bool = True,
) -> nn.Module:
    """
    Initialize model using layer-wise truncated Gaussian prior.

    For each weight tensor (dim >= 2):
        sigma_l = sigma_scale / sqrt(fan_in_l)
        w ~ TruncNormal(0, sigma_l^2, [-truncation*sigma_l, +truncation*sigma_l])

    Bias is set to zero by default (PBB-style for deterministic initial layer means).
    """
    if sigma_scale <= 0:
        raise ValueError(f"sigma_scale must be positive, got {sigma_scale}")
    if truncation <= 0:
        raise ValueError(f"truncation must be positive, got {truncation}")

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                w = module.weight
                if w.dim() >= 2:
                    fan_in = _fan_in_from_weight(w)
                    sigma_l = sigma_scale / math.sqrt(float(fan_in))
                    lower = -truncation * sigma_l
                    upper = truncation * sigma_l
                    _trunc_normal_(w, mean=0.0, std=sigma_l, a=lower, b=upper)
                else:
                    w.zero_()

            if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor) and module.bias is not None:
                if zero_bias:
                    module.bias.zero_()

    return model


def build_layerwise_sigma_map(
    model: nn.Module,
    sigma_scale: float = 1.0,
    bias_strategy: str = "same_as_weight",
    fallback_sigma: float = 1.0,
) -> Dict[str, float]:
    """
    Build per-parameter sigma values for SGLD prior regularization.

    Returns a dict mapping model.named_parameters() names to sigma.
    """
    if sigma_scale <= 0:
        raise ValueError(f"sigma_scale must be positive, got {sigma_scale}")
    if fallback_sigma <= 0:
        raise ValueError(f"fallback_sigma must be positive, got {fallback_sigma}")

    sigma_map: Dict[str, float] = {}

    for module_name, module in model.named_modules():
        local_sigmas: Dict[str, float] = {}

        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight
            if w.dim() >= 2:
                fan_in = _fan_in_from_weight(w)
                local_sigmas["weight"] = sigma_scale / math.sqrt(float(fan_in))

        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor) and module.bias is not None:
            if bias_strategy == "same_as_weight" and "weight" in local_sigmas:
                local_sigmas["bias"] = local_sigmas["weight"]
            elif bias_strategy == "fallback":
                local_sigmas["bias"] = fallback_sigma
            else:
                local_sigmas["bias"] = local_sigmas.get("weight", fallback_sigma)

        for param_name, _ in module.named_parameters(recurse=False):
            if param_name in local_sigmas:
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                sigma_map[full_name] = float(local_sigmas[param_name])

    # Ensure all parameters have a sigma (fallback for unusual/custom params)
    for full_name, _ in model.named_parameters():
        sigma_map.setdefault(full_name, float(fallback_sigma))

    return sigma_map


def build_sgld_param_groups_from_sigma_map(
    model: nn.Module,
    sigma_map: Dict[str, float],
    beta: float,
) -> List[Dict[str, object]]:
    """
    Create per-parameter optimizer groups with layer-wise weight decay.

    weight_decay_i = 1 / (sigma_i^2 * beta) if beta>0 else 1 / sigma_i^2
    """
    param_groups: List[Dict[str, object]] = []

    for name, param in model.named_parameters():
        sigma = float(sigma_map.get(name, 1.0))
        if sigma <= 0:
            raise ValueError(f"Invalid sigma for parameter '{name}': {sigma}")

        weight_decay = 1.0 / (sigma * sigma * beta) if beta > 0 else 1.0 / (sigma * sigma)
        param_groups.append({
            "params": [param],
            "weight_decay": weight_decay,
        })

    return param_groups
