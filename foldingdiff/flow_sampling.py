"""
Sampling using flow matching with ODE integration.

This module provides fast sampling through ODE integration,
achieving 10-20x speedup compared to diffusion models.
"""
import logging
from typing import *

import torch
from torch import nn
from tqdm.auto import tqdm

from foldingdiff import utils


@torch.no_grad()
def euler_sample_step(
    model: nn.Module,
    x_t: torch.Tensor,
    t: float,
    dt: float,
    attention_mask: torch.Tensor,
    motif_mask: Optional[torch.Tensor] = None,
    motif_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Single Euler integration step for flow matching.
    
    x_{t-dt} = x_t - dt * v_θ(x_t, t)
    
    Args:
        model: Flow matching model
        x_t: Current state [batch, seq_len, features]
        t: Current time (scalar)
        dt: Time step size
        attention_mask: [batch, seq_len] attention mask
        motif_mask: Optional [batch, seq_len, 1] motif mask
        motif_features: Optional [batch, seq_len, features] motif features
    
    Returns:
        x_next: Next state [batch, seq_len, features]
    """
    batch_size = x_t.shape[0]
    device = x_t.device
    
    # Create time tensor
    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
    
    # Predict velocity
    v_t = model(
        x_t, t_tensor,
        attention_mask=attention_mask,
        motif_mask=motif_mask,
        motif_features=motif_features,
        coords=None,  # Will be auto-computed
        aa_types=None,  # Will default to unknown
        secondary_structure=None,  # Will be auto-computed
    )
    
    # Euler step: x_{t-dt} = x_t - dt * v_t
    # (negative because we're going backwards from t=1 to t=0)
    x_next = x_t - dt * v_t
    
    # Keep motif regions fixed if provided
    if motif_mask is not None and motif_features is not None:
        mask_expanded = motif_mask.expand_as(x_next)
        x_next = (1 - mask_expanded) * x_next + mask_expanded * motif_features
    
    return x_next


@torch.no_grad()
def rk4_sample_step(
    model: nn.Module,
    x_t: torch.Tensor,
    t: float,
    dt: float,
    attention_mask: torch.Tensor,
    motif_mask: Optional[torch.Tensor] = None,
    motif_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Single Runge-Kutta 4th order integration step.
    
    More accurate than Euler but requires 4 model evaluations per step.
    
    Args:
        model: Flow matching model
        x_t: Current state [batch, seq_len, features]
        t: Current time (scalar)
        dt: Time step size
        attention_mask: [batch, seq_len] attention mask
        motif_mask: Optional [batch, seq_len, 1] motif mask
        motif_features: Optional [batch, seq_len, features] motif features
    
    Returns:
        x_next: Next state [batch, seq_len, features]
    """
    batch_size = x_t.shape[0]
    device = x_t.device
    
    def get_velocity(x, time):
        t_tensor = torch.full((batch_size,), time, device=device, dtype=torch.float32)
        return model(
            x, t_tensor,
            attention_mask=attention_mask,
            motif_mask=motif_mask,
            motif_features=motif_features,
            coords=None,
            aa_types=None,
            secondary_structure=None,
        )
    
    # RK4 coefficients
    k1 = get_velocity(x_t, t)
    k2 = get_velocity(x_t - 0.5 * dt * k1, t - 0.5 * dt)
    k3 = get_velocity(x_t - 0.5 * dt * k2, t - 0.5 * dt)
    k4 = get_velocity(x_t - dt * k3, t - dt)
    
    # Weighted average
    x_next = x_t - (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Keep motif regions fixed
    if motif_mask is not None and motif_features is not None:
        mask_expanded = motif_mask.expand_as(x_next)
        x_next = (1 - mask_expanded) * x_next + mask_expanded * motif_features
    
    return x_next


@torch.no_grad()
def sample_flow_matching(
    model: nn.Module,
    shape: Tuple[int, int, int],
    num_steps: int = 50,
    method: Literal['euler', 'rk4'] = 'euler',
    attention_mask: Optional[torch.Tensor] = None,
    motif_mask: Optional[torch.Tensor] = None,
    motif_features: Optional[torch.Tensor] = None,
    is_angular: Optional[List[bool]] = None,
    disable_pbar: bool = False,
    return_trajectory: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Sample from flow matching model using ODE integration.
    
    Args:
        model: Flow matching model
        shape: (batch_size, seq_len, n_features)
        num_steps: Number of integration steps (default: 50)
        method: Integration method ('euler' or 'rk4')
        attention_mask: [batch, seq_len] attention mask
        motif_mask: [batch, seq_len, 1] motif mask
        motif_features: [batch, seq_len, features] motif features
        is_angular: List indicating which features are angular
        disable_pbar: Disable progress bar
        return_trajectory: Return full sampling trajectory
    
    Returns:
        x_0: Final sample [batch, seq_len, features]
        trajectory: (optional) List of intermediate states
    """
    device = next(model.parameters()).device
    batch_size, seq_len, n_features = shape
    
    # Start from noise (t=1)
    x_t = torch.randn(shape, device=device)
    
    # Set motif regions if provided
    if motif_mask is not None and motif_features is not None:
        motif_mask = motif_mask.to(device)
        motif_features = motif_features.to(device)
        mask_expanded = motif_mask.expand_as(x_t)
        x_t = (1 - mask_expanded) * x_t + mask_expanded * motif_features
    
    # Create default attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_len, device=device)
    else:
        attention_mask = attention_mask.to(device)
    
    # Time step size
    dt = 1.0 / num_steps
    
    # Integration method
    if method == 'euler':
        step_fn = euler_sample_step
    elif method == 'rk4':
        step_fn = rk4_sample_step
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logging.info(
        f"Flow matching sampling: {num_steps} steps, method={method}, "
        f"shape={shape}, motif={motif_mask is not None}"
    )
    
    trajectory = [] if return_trajectory else None
    
    # Integrate from t=1 to t=0
    for i in tqdm(
        range(num_steps),
        desc=f"Flow sampling ({method})",
        disable=disable_pbar
    ):
        t = 1.0 - i * dt
        
        x_t = step_fn(
            model=model,
            x_t=x_t,
            t=t,
            dt=dt,
            attention_mask=attention_mask,
            motif_mask=motif_mask,
            motif_features=motif_features,
        )
        
        # Wrap angular features
        if is_angular is not None:
            for j, angular in enumerate(is_angular):
                if angular:
                    x_t[:, :, j] = utils.modulo_with_wrapped_range(
                        x_t[:, :, j],
                        range_min=-torch.pi,
                        range_max=torch.pi
                    )
        
        if return_trajectory:
            trajectory.append(x_t.cpu())
    
    if return_trajectory:
        return x_t, trajectory
    return x_t


@torch.no_grad()
def sample_flow_matching_with_guidance(
    model: nn.Module,
    shape: Tuple[int, int, int],
    motif_mask: torch.Tensor,
    motif_features: torch.Tensor,
    guidance_scale: float = 2.0,
    num_steps: int = 50,
    method: Literal['euler', 'rk4'] = 'euler',
    attention_mask: Optional[torch.Tensor] = None,
    is_angular: Optional[List[bool]] = None,
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Sample with classifier-free guidance for flow matching.
    
    v_guided = v_uncond + scale * (v_cond - v_uncond)
    
    Args:
        model: Flow matching model
        shape: (batch_size, seq_len, n_features)
        motif_mask: [batch, seq_len, 1] motif mask
        motif_features: [batch, seq_len, features] motif features
        guidance_scale: Guidance strength (1.0 = no guidance)
        num_steps: Number of integration steps
        method: Integration method
        attention_mask: [batch, seq_len] attention mask
        is_angular: List indicating which features are angular
        disable_pbar: Disable progress bar
    
    Returns:
        x_0: Final sample [batch, seq_len, features]
    """
    device = next(model.parameters()).device
    batch_size, seq_len, n_features = shape
    
    # Start from noise
    x_t = torch.randn(shape, device=device)
    
    # Move to device
    motif_mask = motif_mask.to(device)
    motif_features = motif_features.to(device)
    
    # Set motif regions
    mask_expanded = motif_mask.expand_as(x_t)
    x_t = (1 - mask_expanded) * x_t + mask_expanded * motif_features
    
    # Create attention mask
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_len, device=device)
    else:
        attention_mask = attention_mask.to(device)
    
    dt = 1.0 / num_steps
    
    logging.info(
        f"Flow matching with guidance: scale={guidance_scale}, "
        f"steps={num_steps}, method={method}"
    )
    
    for i in tqdm(
        range(num_steps),
        desc=f"Guided flow sampling",
        disable=disable_pbar
    ):
        t = 1.0 - i * dt
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        
        # Prepare inputs for enhanced model
        # Coords will be auto-computed from angles if needed
        # AA types will default to unknown (20) if not provided
        # SS will be auto-computed if needed
        
        # Conditional velocity
        v_cond = model(
            x_t, t_tensor,
            attention_mask=attention_mask,
            motif_mask=motif_mask,
            motif_features=motif_features,
            coords=None,  # Will be auto-computed
            aa_types=None,  # Will default to unknown
            secondary_structure=None,  # Will be auto-computed
        )
        
        # Unconditional velocity
        v_uncond = model(
            x_t, t_tensor,
            attention_mask=attention_mask,
            motif_mask=torch.zeros_like(motif_mask),
            motif_features=None,
            coords=None,
            aa_types=None,
            secondary_structure=None,
        )
        
        # Guided velocity
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        # Integration step
        x_t = x_t - dt * v_guided
        
        # Keep motif regions fixed
        x_t = (1 - mask_expanded) * x_t + mask_expanded * motif_features
        
        # Wrap angular features
        if is_angular is not None:
            for j, angular in enumerate(is_angular):
                if angular:
                    x_t[:, :, j] = utils.modulo_with_wrapped_range(
                        x_t[:, :, j],
                        range_min=-torch.pi,
                        range_max=torch.pi
                    )
    
    return x_t


def estimate_sampling_speedup(
    diffusion_steps: int = 1000,
    flow_steps: int = 50
) -> float:
    """
    Estimate speedup from using flow matching vs diffusion.
    
    Args:
        diffusion_steps: Number of diffusion steps
        flow_steps: Number of flow matching steps
    
    Returns:
        speedup: Expected speedup factor
    """
    return diffusion_steps / flow_steps
