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
        alpha: float = 2.0,
        use_u_shaped: bool = True  # NEW: U-shaped distribution for rectified flows (2025 improvement)
    ) -> torch.Tensor:
        """
        Sample time from [0, 1] with optional importance weighting.
        
        Importance weighting samples more frequently near t=0 and t=1,
        which are more critical for flow matching performance.
        
        NEW (2025): U-shaped distribution option for rectified flows, which has been
        shown to improve performance in low NFE settings.
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            importance_weighting: If True, use beta distribution for sampling
            alpha: Shape parameter for beta distribution (higher = more emphasis on boundaries)
            use_u_shaped: If True, use U-shaped distribution (better for rectified flows)
        
        Returns:
            t: Time samples [batch_size]
        """
        if importance_weighting:
            if use_u_shaped:
                # U-shaped distribution: samples more at boundaries (t=0 and t=1)
                # This is better for rectified flows (web research: arxiv.org/abs/2405.20320)
                # Use a mixture: 50% from Beta(0.5, 0.5) (U-shaped) and 50% from Beta(alpha, alpha)
                u_samples = torch.distributions.Beta(0.5, 0.5).sample((batch_size // 2,)).to(device)
                beta_samples = torch.distributions.Beta(alpha, alpha).sample((batch_size - batch_size // 2,)).to(device)
                t = torch.cat([u_samples, beta_samples])
                # Shuffle to mix the distributions
                t = t[torch.randperm(batch_size, device=device)]
            else:
                # Beta distribution concentrates samples near 0 and 1
                # Beta(alpha, alpha) gives symmetric distribution
                t = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
            return t
        else:
            return torch.rand(batch_size, device=device)


class RiemannianFlowMatchingSchedule:
    """
    Riemannian Flow Matching for torus geometry (dihedral angles).
    
    CRITICAL FIX: Protein dihedrals live on a Torus (T^n), not Euclidean space.
    Standard Flow Matching takes the "straight line" through the middle of the circle,
    which is wrong. For example, if target is 170° and noise is -170°, the Euclidean
    average is 0°, but the shortest path on a circle is through 180°.
    
    This class implements:
    - Geodesic interpolation (shortest path on circle)
    - Wrapped Gaussian noise sampling
    - Periodic velocity computation
    
    References:
    - Chen & Lipman (2024) "Riemannian Flow Matching"
    - Yim et al. (2023) "FoldFlow: Straightening the Protein Folding Path"
    
    Args:
        sigma_min: Minimum noise level (for numerical stability)
        is_angular: List indicating which features are angular (circular)
    """
    
    def __init__(self, sigma_min: float = 0.001, is_angular: Optional[List[bool]] = None):
        self.sigma_min = sigma_min
        self.is_angular = is_angular if is_angular is not None else []
        logging.info(f"RiemannianFlowMatchingSchedule with sigma_min={sigma_min}, is_angular={is_angular}")
    
    def sample_wrapped_gaussian(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        mean: float = 0.0,
        std: float = 1.0
    ) -> torch.Tensor:
        """
        Sample from wrapped Gaussian distribution on the circle.
        
        Wrapped Gaussian: Sample from N(mean, std^2) then wrap to [-π, π].
        This is the correct noise distribution for circular variables.
        
        Args:
            shape: Shape of tensor to sample
            device: Device to create tensor on
            mean: Mean of Gaussian (will be wrapped)
            std: Standard deviation of Gaussian
            device: Device to create tensor on
        
        Returns:
            Samples from wrapped Gaussian in range [-π, π]
        """
        # Sample from standard Gaussian
        samples = torch.randn(shape, device=device) * std + mean
        # Wrap to [-π, π]
        samples = torch.atan2(torch.sin(samples), torch.cos(samples))
        return samples
    
    def geodesic_interpolant(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        is_angular: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        Compute geodesic interpolation on torus for angular features.
        
        For angular features: uses shortest path on circle (geodesic).
        For non-angular features: uses linear interpolation (Euclidean).
        
        Geodesic interpolation on circle:
        - Compute periodic difference: diff = atan2(sin(x_1 - x_0), cos(x_1 - x_0))
        - Interpolate along geodesic: x_t = x_0 + t * diff (wrapped)
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            t: Time [batch] or [batch, 1, 1]
            is_angular: List indicating which features are angular
        
        Returns:
            x_t: Geodesic interpolant [batch, seq_len, features]
        """
        if is_angular is None:
            is_angular = self.is_angular
        
        # Ensure t has correct shape for broadcasting
        if t.ndim == 1:
            t = t.view(-1, 1, 1)  # [batch, 1, 1]
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t.view(-1, 1, 1)
        
        # Start with Euclidean interpolation (for non-angular features)
        x_t = (1 - t) * x_0 + t * x_1
        
        # Apply geodesic interpolation for angular features
        if is_angular and len(is_angular) > 0:
            # Convert to tensor if needed
            if not isinstance(is_angular, torch.Tensor):
                is_angular_tensor = torch.tensor(is_angular, device=x_0.device, dtype=torch.bool)
            else:
                is_angular_tensor = is_angular
            
            # Expand to match feature dimensions
            if is_angular_tensor.ndim == 1:
                is_angular_tensor = is_angular_tensor.view(1, 1, -1)  # [1, 1, features]
            
            # Compute periodic difference for angular features
            # diff = atan2(sin(x_1 - x_0), cos(x_1 - x_0)) gives shortest path
            diff = x_1 - x_0
            diff_angular = torch.atan2(torch.sin(diff), torch.cos(diff))
            
            # Interpolate along geodesic: x_t = x_0 + t * diff (wrapped)
            x_t_angular = x_0 + t * diff_angular
            # Wrap to [-π, π]
            x_t_angular = torch.atan2(torch.sin(x_t_angular), torch.cos(x_t_angular))
            
            # Combine: angular features use geodesic, non-angular use Euclidean
            angular_mask = is_angular_tensor.expand_as(x_t)
            x_t = torch.where(angular_mask, x_t_angular, x_t)
        
        return x_t
    
    def get_interpolant(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        is_angular: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        Compute interpolant using geodesic for angular features.
        
        This is the main entry point that delegates to geodesic_interpolant.
        """
        return self.geodesic_interpolant(x_0, x_1, t, is_angular)
    
    def get_target_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        is_angular: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        Compute target velocity field for geodesic paths on torus.
        
        CRITICAL FIX: For geodesic paths on a circle, the velocity is the derivative
        of the geodesic path. Since the geodesic is x_t = x_0 + t * diff_angular,
        the velocity is constant: v_t = diff_angular = atan2(sin(x_1 - x_0), cos(x_1 - x_0)).
        
        This is NOT the same as wrapping the Euclidean difference! The velocity must
        be the modular (periodic) difference that defines the geodesic path.
        
        For angular features: velocity = periodic difference (constant along geodesic).
        For non-angular features: velocity = Euclidean difference (constant).
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            t: Time (unused, velocity is constant for geodesic paths)
            is_angular: List indicating which features are angular
        
        Returns:
            v_t: Target velocity [batch, seq_len, features] (constant for geodesic paths)
        """
        if is_angular is None:
            is_angular = self.is_angular
        
        # Start with Euclidean difference (for non-angular features)
        v_t = x_1 - x_0
        
        # Apply periodic difference for angular features
        if is_angular and len(is_angular) > 0:
            # Convert to tensor if needed
            if not isinstance(is_angular, torch.Tensor):
                is_angular_tensor = torch.tensor(is_angular, device=x_0.device, dtype=torch.bool)
            else:
                is_angular_tensor = is_angular
            
            # Expand to match feature dimensions
            if is_angular_tensor.ndim == 1:
                is_angular_tensor = is_angular_tensor.view(1, 1, -1)  # [1, 1, features]
            
            # CRITICAL FIX: Velocity for geodesic path on circle
            # The geodesic path is: x_t = x_0 + t * diff_angular
            # The velocity is the derivative: v_t = diff_angular (constant!)
            # where diff_angular = atan2(sin(x_1 - x_0), cos(x_1 - x_0))
            # This is the modular difference that defines the shortest path on the circle.
            diff_linear = x_1 - x_0
            diff_angular = torch.atan2(torch.sin(diff_linear), torch.cos(diff_linear))
            
            # Combine: angular features use periodic diff (geodesic velocity), non-angular use Euclidean
            angular_mask = is_angular_tensor.expand_as(v_t)
            v_t = torch.where(angular_mask, diff_angular, v_t)
        
        return v_t
    
    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        importance_weighting: bool = True,
        alpha: float = 2.0,
        use_u_shaped: bool = True
    ) -> torch.Tensor:
        """
        Sample time from [0, 1] (same as standard flow matching).
        
        This method is identical to FlowMatchingSchedule.sample_time()
        since time sampling doesn't depend on geometry.
        """
        if importance_weighting:
            if use_u_shaped:
                # U-shaped distribution: Beta(alpha, alpha)
                # Higher probability near t=0 and t=1
                from torch.distributions import Beta
                dist = Beta(alpha, alpha)
                t = dist.sample((batch_size,)).to(device)
            else:
                # Standard importance weighting: Beta(alpha, 1)
                from torch.distributions import Beta
                dist = Beta(alpha, 1.0)
                t = dist.sample((batch_size,)).to(device)
            
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
    
    def __init__(self, sigma_min: float = 0.001, schedule: Optional[Any] = None):
        if schedule is not None:
            self.schedule = schedule
        else:
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
    
    def get_harmonized_interpolant(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        motif_mask: torch.Tensor,
        motif_noise_scale: float = 0.1,
        is_angular: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        Harmonized interpolant that slightly perturbs motif regions.
        
        CRITICAL FIX: Instead of hard-pasting the motif (which creates discontinuities),
        we slightly perturb the motif with reduced noise. This creates a smooth boundary
        that the model can learn more easily.
        
        CRITICAL FIX 2: Time-Dependent Motif Anchoring
        As t→0, the motif_noise_scale should approach 0 faster than the scaffold noise.
        This ensures the motif becomes perfectly clean as we approach the data distribution,
        preventing "frayed ends" where the scaffold doesn't dock into the motif.
        
        Formula: motif_noise_scale(t) = motif_noise_scale * (1 - t)^2
        This makes the motif anchor faster than linear decay, ensuring clean docking.
        
        Uses geodesic interpolation for angular features (torus geometry).
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            t: Time [batch] or [batch, 1]
            motif_mask: [batch, seq_len, 1] where 1=motif, 0=scaffold
            motif_noise_scale: Base noise scale for motif (0.0 = no noise, 1.0 = full noise)
            is_angular: List indicating which features are angular (for geodesic interpolation)
        
        Returns:
            x_t: Harmonized interpolant [batch, seq_len, features]
        """
        # Ensure t has correct shape for broadcasting
        if t.ndim == 1:
            t = t.view(-1, 1, 1)  # [batch, 1, 1]
        elif t.ndim == 2:
            t = t.view(-1, 1, 1)
        
        # CRITICAL FIX: Time-dependent motif anchoring
        # As t→0, motif_noise_scale approaches 0 faster than scaffold noise
        # This ensures the motif becomes perfectly clean as we approach the data distribution
        # Formula: motif_noise_scale(t) = motif_noise_scale * (1 - t)^2
        # The squared term makes it decay faster than linear, ensuring clean docking
        motif_noise_scale_time = motif_noise_scale * (1 - t) ** 2
        
        # Expand mask to match features
        mask_expanded = motif_mask.expand_as(x_0)
        
        # Check if schedule supports geodesic interpolation
        if hasattr(self.schedule, 'geodesic_interpolant'):
            # Use geodesic interpolation (Riemannian flow matching)
            x_t_scaffold = self.schedule.geodesic_interpolant(x_0, x_1, t, is_angular)
            t_motif = t * motif_noise_scale_time
            x_t_motif = self.schedule.geodesic_interpolant(x_0, x_1, t_motif, is_angular)
        else:
            # Fall back to standard interpolation
            x_t_scaffold = self.schedule.get_interpolant(x_0, x_1, t)
            t_motif = t * motif_noise_scale_time
            x_t_motif = self.schedule.get_interpolant(x_0, x_1, t_motif)
        
        # Combine: scaffold uses full noise, motif uses reduced noise
        # For angular features, we need to handle wrapping properly
        if is_angular and any(is_angular):
            # For angular features, we can't simply mix - need to compute geodesic between the two
            # For now, use mask-based combination (works because both are on same torus)
            x_t = (1 - mask_expanded) * x_t_scaffold + mask_expanded * x_t_motif
            # Wrap angular features
            is_angular_tensor = torch.tensor(is_angular, device=x_0.device, dtype=torch.bool).view(1, 1, -1)
            angular_mask = is_angular_tensor.expand_as(x_t)
            x_t_angular = torch.atan2(torch.sin(x_t), torch.cos(x_t))
            x_t = torch.where(angular_mask, x_t_angular, x_t)
        else:
            x_t = (1 - mask_expanded) * x_t_scaffold + mask_expanded * x_t_motif
        
        return x_t
    
    def get_harmonized_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        motif_mask: torch.Tensor,
        motif_noise_scale: float = 0.1,
        is_angular: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        Compute harmonized velocity field.
        
        CRITICAL FIX: Time-Dependent Motif Anchoring
        The velocity in motif regions is scaled down by time-dependent motif_noise_scale,
        matching the interpolant computation. As t→0, the motif velocity approaches 0,
        ensuring the motif becomes perfectly anchored.
        
        Uses periodic difference for angular features (torus geometry).
        
        Args:
            x_0: Clean data [batch, seq_len, features]
            x_1: Noise [batch, seq_len, features]
            t: Time [batch] or [batch, 1]
            motif_mask: [batch, seq_len, 1] where 1=motif, 0=scaffold
            motif_noise_scale: Base velocity scale for motif regions
            is_angular: List indicating which features are angular (for periodic difference)
        
        Returns:
            v_t: Harmonized velocity [batch, seq_len, features]
        """
        # Ensure t has correct shape for broadcasting
        if t.ndim == 1:
            t = t.view(-1, 1, 1)  # [batch, 1, 1]
        elif t.ndim == 2:
            t = t.view(-1, 1, 1)
        
        # CRITICAL FIX: Time-dependent motif anchoring (matches interpolant)
        # As t→0, motif_noise_scale approaches 0 faster than scaffold noise
        motif_noise_scale_time = motif_noise_scale * (1 - t) ** 2
        
        # Get velocity (uses periodic difference for angular features if schedule supports it)
        if hasattr(self.schedule, 'get_target_velocity'):
            v_t = self.schedule.get_target_velocity(x_0, x_1, t, is_angular)
        else:
            v_t = self.schedule.get_target_velocity(x_0, x_1, t)
        
        # Expand mask to match features
        mask_expanded = motif_mask.expand_as(v_t)
        
        # Scale down velocity in motif regions using time-dependent scale (smooth transition, not hard zero)
        v_t = (1 - mask_expanded) * v_t + mask_expanded * v_t * motif_noise_scale_time
        
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
            # CRITICAL FIX: Improved angular loss computation
            # For angular velocities, we need to handle the circular nature properly
            # Even though velocities are derivatives, large differences should be penalized
            diff = pred_v - target_v
            
            # CRITICAL: For omega (index 2), velocities should be near 0 (trans preference)
            # The model learns mean-centered angles, so omega velocity should be small
            if i == 2:  # Omega
                # Omega should stay near 0 (mean-centered), so penalize large velocities more
                # Use L2 loss with higher weight for omega stability
                loss_per_element = diff ** 2
                # Additional penalty for large omega velocities (encourage trans bonds)
                omega_penalty = torch.clamp(torch.abs(pred_v), min=0.0) ** 2 * 0.5
                loss_per_element = loss_per_element + omega_penalty
            else:
                # For phi and psi, use smooth L1 (Huber) loss for better gradient behavior
                abs_diff = torch.abs(diff)
                huber_delta = 1.0
                loss_per_element = torch.where(
                    abs_diff < huber_delta,
                    0.5 * diff ** 2,
                    huber_delta * (abs_diff - 0.5 * huber_delta)
                )
            
            # BEST PRACTICE: Feature-specific loss weighting
            # Omega (index 2) needs higher weight for trans preference
            # Feature weights: phi=1.0, psi=1.0, omega=2.0, tau=1.0, CA:C:1N=1.0, C:1N:1CA=1.0
            feature_weights = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]  # omega gets 2.0x weight (increased from 1.5)
            if i < len(feature_weights):
                loss_per_element = loss_per_element * feature_weights[i]
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
    
    CRITICAL FIX: Scale timesteps by 1000 before embedding.
    Timesteps in [0, 1] change too slowly for high-frequency sine/cosine components,
    leading to "Time Collapse" where the model cannot distinguish between t=0.1 and t=0.2.
    Scaling by 1000 maps [0, 1] to [0, 1000], which is appropriate for sinusoidal embeddings.
    
    Args:
        timesteps: [batch_size] timestep values in [0, 1]
        embedding_dim: Dimension of embedding
        max_period: Maximum period for sinusoids
    
    Returns:
        embeddings: [batch_size, embedding_dim]
    """
    # CRITICAL FIX: Scale timesteps by 1000 before embedding
    # This prevents "Time Collapse" where model cannot distinguish between nearby timesteps
    timesteps_scaled = timesteps * 1000.0
    
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    ).to(timesteps.device)
    
    args = timesteps_scaled[:, None] * freqs[None, :]
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if embedding_dim % 2 == 1:
        embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
    return embeddings
