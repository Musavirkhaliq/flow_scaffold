"""
Flow matching implementation for protein generation.

Flow matching provides a deterministic alternative to diffusion models,
enabling faster sampling through ODE integration instead of iterative denoising.

Key differences from diffusion:
- Predicts velocity field v(x,t) instead of noise ε
- Uses continuous time t ∈ [0,1] instead of discrete steps
- Deterministic ODE flow instead of stochastic process
- Faster sampling: 50 steps vs 1000 steps

References:
- Lipman et al. (2023) "Flow Matching for Generative Modeling"
- Liu et al. (2023) "Flow Straight and Fast"
"""
import logging
from typing import *

import torch
import torch.nn as nn
import numpy as np


class FlowMatchingSchedule:
    """
    Handles time scheduling and interpolation for flow matching.
    
    Flow matching learns to transform noise (t=1) to data (t=0) via
    a continuous normalizing flow defined by an ODE:
        dx/dt = v_θ(x_t, t)
    
    Args:
        sigma_min: Minimum noise level (for numerical stability)
    """
    
    def __init__(self, sigma_min: float = 0.001):
        self.sigma_min = sigma_min
        logging.info(f"FlowMatchingSchedule with sigma_min={sigma_min}")
    
    def get_interpolant(
        self, 
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute linear interpolation between data and noise.
        
        x_t = (1-t) * x_0 + t * x_1
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            t: Time [batch] or [batch, 1, 1]
        
        Returns:
            x_t: Interpolated state [batch, seq_len, features]
        """
        # Ensure t has correct shape for broadcasting
        if t.ndim == 1:
            t = t.view(-1, 1, 1)
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t.view(-1, 1, 1)
        
        return (1 - t) * x_0 + t * x_1
    
    def get_target_velocity(
        self, 
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute target velocity field.
        
        For linear interpolation, the velocity is constant:
        v_t = dx/dt = x_1 - x_0
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            t: Time (unused for linear flow, kept for API compatibility)
        
        Returns:
            v_t: Target velocity [batch, seq_len, features]
        """
        return x_1 - x_0
    
    def sample_time(
        self, 
        batch_size: int, 
        device: torch.device,
        importance_weighting: bool = True,
        alpha: float = 2.0
    ) -> torch.Tensor:
        """
        Sample time from [0, 1] with optional importance weighting.
        
        Importance weighting samples more frequently near t=0 and t=1,
        which are more critical for flow matching performance.
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            importance_weighting: If True, use beta distribution for sampling
            alpha: Shape parameter for beta distribution (higher = more emphasis on boundaries)
        
        Returns:
            t: Time samples [batch_size]
        """
        if importance_weighting:
            # Beta distribution concentrates samples near 0 and 1
            # Beta(alpha, alpha) gives symmetric distribution
            t = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
            return t
        else:
            return torch.rand(batch_size, device=device)


class ConditionalFlowMatching:
    """
    Conditional flow matching for motif scaffolding.
    
    Extends flow matching to handle conditional generation where
    motif regions are fixed and only scaffold regions flow.
    
    Args:
        sigma_min: Minimum noise level
    """
    
    def __init__(self, sigma_min: float = 0.001):
        self.schedule = FlowMatchingSchedule(sigma_min)
        logging.info("ConditionalFlowMatching initialized")
    
    def get_conditional_interpolant(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        motif_mask: torch.Tensor,
        smooth_transition: bool = True,
        transition_width: float = 0.1
    ) -> torch.Tensor:
        """
        Interpolate only scaffold regions, keep motif fixed.
        
        With smooth_transition=True, adds a smooth boundary between motif and scaffold
        to prevent discontinuities that can hurt training.
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            t: Time [batch]
            motif_mask: [batch, seq_len, 1] where 1=motif, 0=scaffold
            smooth_transition: If True, use smooth boundary instead of hard mask
            transition_width: Width of transition region (in residues)
        
        Returns:
            x_t: Conditional interpolant [batch, seq_len, features]
        """
        # Get standard interpolant
        x_t = self.schedule.get_interpolant(x_0, x_1, t)
        
        # Expand mask to match features
        mask_expanded = motif_mask.expand_as(x_t)
        
        if smooth_transition:
            # Apply Gaussian smoothing to mask boundaries for smoother transitions
            # This helps the model learn better boundary conditions
            from torch.nn.functional import conv1d, pad
            # Convert mask to float and add channel dimension for conv1d
            mask_1d = motif_mask.squeeze(-1)  # [batch, seq_len]
            # Create 1D Gaussian kernel for smoothing
            kernel_size = int(transition_width * mask_1d.shape[1])
            if kernel_size > 1 and kernel_size % 2 == 1:
                sigma = kernel_size / 6.0
                kernel_1d = torch.exp(-0.5 * ((torch.arange(kernel_size, device=mask_1d.device) - kernel_size // 2) / sigma) ** 2)
                kernel_1d = kernel_1d / kernel_1d.sum()
                kernel_1d = kernel_1d.view(1, 1, -1)  # [1, 1, kernel_size]
                
                # Apply convolution with padding
                mask_padded = pad(mask_1d.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode='replicate')
                mask_smooth = conv1d(mask_padded, kernel_1d, padding=0).squeeze(1)
                mask_expanded = mask_smooth.unsqueeze(-1).expand_as(x_t)
        
        # Keep motif regions fixed at x_0, scaffold regions interpolate
        x_t = (1 - mask_expanded) * x_t + mask_expanded * x_0
        
        return x_t
    
    def get_conditional_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        motif_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity field for conditional flow.
        
        Velocity is zero in motif regions, x_1 - x_0 in scaffold regions.
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            motif_mask: [batch, seq_len, 1] where 1=motif, 0=scaffold
        
        Returns:
            v_t: Conditional velocity [batch, seq_len, features]
        """
        # Get standard velocity
        v_t = self.schedule.get_target_velocity(x_0, x_1)
        
        # Expand mask to match features
        mask_expanded = motif_mask.expand_as(v_t)
        
        # Zero out velocity in motif regions
        v_t = (1 - mask_expanded) * v_t
        
        return v_t


class OptimalTransportFlowMatching:
    """
    Optimal transport flow matching for improved sample quality.
    
    Uses optimal transport to find better paths between noise and data,
    potentially improving generation quality compared to linear interpolation.
    
    Note: This is a placeholder for future implementation.
    For now, uses linear interpolation like standard flow matching.
    
    Args:
        sigma_min: Minimum noise level
    """
    
    def __init__(self, sigma_min: float = 0.001):
        self.schedule = FlowMatchingSchedule(sigma_min)
        logging.warning(
            "OptimalTransportFlowMatching not fully implemented, "
            "using linear interpolation"
        )
    
    def get_interpolant(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OT-based interpolation.
        
        TODO: Implement optimal transport coupling.
        Currently uses linear interpolation.
        """
        return self.schedule.get_interpolant(x_0, x_1, t)
    
    def get_target_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OT-based velocity field.
        
        TODO: Implement optimal transport velocity.
        Currently uses constant velocity.
        """
        return self.schedule.get_target_velocity(x_0, x_1, t)


def compute_flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute flow matching loss.
    
    L = E_t,x_0,x_1 [||v_θ(x_t, t) - v_target||²]
    
    Args:
        predicted_velocity: Model prediction [batch, seq_len, features]
        target_velocity: Target velocity [batch, seq_len, features]
        mask: Optional mask [batch, seq_len] for valid positions
    
    Returns:
        loss: Scalar loss value
    """
    if mask is not None:
        # Compute loss only on valid positions
        mask_expanded = mask.unsqueeze(-1).expand_as(predicted_velocity)
        diff = (predicted_velocity - target_velocity) * mask_expanded
        loss = (diff ** 2).sum() / mask_expanded.sum()
    else:
        # Compute loss on all positions
        loss = torch.mean((predicted_velocity - target_velocity) ** 2)
    
    return loss


def compute_diversity_penalty(
    samples: torch.Tensor,
    mask: torch.Tensor,
    diversity_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute diversity penalty to prevent mode collapse.
    
    Penalizes samples that are too similar within a batch.
    Encourages diverse backbone generation.
    
    Args:
        samples: [batch, seq_len, features] samples
        mask: [batch, seq_len] attention mask
        diversity_weight: Weight for diversity penalty
    
    Returns:
        penalty: Diversity penalty (to be subtracted from loss)
    """
    import torch.nn.functional as F
    
    batch_size = samples.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=samples.device)
    
    # Only consider valid positions
    mask_expanded = mask.unsqueeze(-1).expand_as(samples)
    valid_samples = samples * mask_expanded
    
    # Flatten samples
    samples_flat = valid_samples.reshape(batch_size, -1)
    
    # Compute pairwise cosine similarity
    samples_norm = F.normalize(samples_flat, p=2, dim=1)
    similarity = torch.mm(samples_norm, samples_norm.t())
    
    # Remove diagonal (self-similarity) and lower triangle
    mask_triu = torch.triu(torch.ones_like(similarity), diagonal=1)
    similarities = similarity * mask_triu
    
    # Penalize high similarity (low diversity)
    # We want low similarity = high diversity
    # Return negative penalty (to be subtracted from loss)
    diversity_penalty = similarities.abs().mean()
    
    return diversity_weight * diversity_penalty


def compute_angular_flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    is_angular: List[bool],
    mask: Optional[torch.Tensor] = None,
    motif_mask: Optional[torch.Tensor] = None,
    scaffold_weight: float = 2.0,
    diversity_samples: Optional[torch.Tensor] = None,
    diversity_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute flow matching loss with special handling for angular features.
    
    For angular features, we need to handle the circular nature properly.
    Optionally weights scaffold regions more heavily than motif regions.
    
    Args:
        predicted_velocity: Model prediction [batch, seq_len, features]
        target_velocity: Target velocity [batch, seq_len, features]
        is_angular: List indicating which features are angular
        mask: Optional mask [batch, seq_len] for valid positions
        motif_mask: Optional [batch, seq_len, 1] motif mask for weighting
        scaffold_weight: Weight for scaffold regions (motif regions have weight 1.0)
    
    Returns:
        loss: Scalar loss value
    """
    losses = []
    
    # Create weighting mask if motif_mask is provided
    weight_mask = None
    if motif_mask is not None:
        # Scaffold regions get scaffold_weight, motif regions get 1.0
        weight_mask = (1 - motif_mask.squeeze(-1)) * scaffold_weight + motif_mask.squeeze(-1)
    
    for i, angular in enumerate(is_angular):
        pred_v = predicted_velocity[:, :, i]
        target_v = target_velocity[:, :, i]
        
        if angular:
            # For angular velocities, the difference should be computed carefully
            # Since velocities are derivatives, they don't wrap like angles
            # But we still want to handle large differences properly
            diff = pred_v - target_v
            # Use smooth L1 loss for better gradient behavior
            abs_diff = torch.abs(diff)
            # Huber loss: quadratic for small errors, linear for large
            huber_delta = 1.0
            loss_per_element = torch.where(
                abs_diff < huber_delta,
                0.5 * diff ** 2,
                huber_delta * (abs_diff - 0.5 * huber_delta)
            )
        else:
            # For non-angular features, use standard MSE
            diff = pred_v - target_v
            loss_per_element = diff ** 2
        
        # Apply position mask (valid residues)
        if mask is not None:
            loss_per_element = loss_per_element * mask
        
        # Apply scaffold/motif weighting
        if weight_mask is not None:
            loss_per_element = loss_per_element * weight_mask
        
        # Compute mean loss
        if mask is not None:
            # Normalize by weighted mask sum
            if weight_mask is not None:
                norm_mask = mask * weight_mask
            else:
                norm_mask = mask
            mask_sum = norm_mask.sum()
            if mask_sum > 0:
                loss = loss_per_element.sum() / mask_sum
            else:
                loss = torch.tensor(0.0, device=predicted_velocity.device)
        else:
            if weight_mask is not None:
                # Weighted mean
                weighted_sum = (loss_per_element * weight_mask).sum()
                weight_sum = weight_mask.sum()
                loss = weighted_sum / weight_sum if weight_sum > 0 else torch.tensor(0.0, device=predicted_velocity.device)
            else:
                loss = torch.mean(loss_per_element)
        
        losses.append(loss)
    
    main_loss = torch.mean(torch.stack(losses))
    
    # Add diversity penalty if samples provided
    if diversity_samples is not None and diversity_weight > 0:
        diversity_penalty = compute_diversity_penalty(
            diversity_samples, mask, diversity_weight
        )
        # Subtract penalty (we want to maximize diversity)
        main_loss = main_loss - diversity_penalty
    
    return main_loss


# Utility functions for time encoding

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10000.0
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: [batch_size] timestep values in [0, 1]
        embedding_dim: Dimension of embedding
        max_period: Maximum period for sinusoids
    
    Returns:
        embeddings: [batch_size, embedding_dim]
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    ).to(timesteps.device)
    
    args = timesteps[:, None] * freqs[None, :]
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if embedding_dim % 2 == 1:
        embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
    return embeddings
