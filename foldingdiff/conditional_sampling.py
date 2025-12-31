"""
Sampling utilities for conditional motif scaffolding.

This module provides sampling functions that preserve motif regions
while generating scaffold regions using diffusion models.
"""
import logging
from typing import *

import torch
from torch import nn
from tqdm.auto import tqdm

from foldingdiff import utils


@torch.no_grad()
def p_sample_conditional(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    t_index: int,
    betas: torch.Tensor,
    motif_mask: Optional[torch.Tensor] = None,
    motif_angles: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Sample one timestep with motif conditioning.
    
    Args:
        model: Diffusion model
        x: Current noisy state [batch, seq_len, features]
        t: Timestep tensor [batch]
        seq_lens: Sequence lengths
        t_index: Current timestep index
        betas: Beta schedule
        motif_mask: [batch, seq_len, 1] binary mask (1=motif, 0=scaffold)
        motif_angles: [batch, seq_len, features] clean motif angles
        guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
        
    Returns:
        x_prev: Denoised state at t-1
    """
    from foldingdiff import beta_schedules
    
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])
    
    # Select based on time
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]
    
    # Create attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, length in enumerate(seq_lens):
        attn_mask[i, :length] = 1.0
    
    # Predict noise with conditioning
    if guidance_scale != 1.0 and motif_mask is not None:
        # Classifier-free guidance
        # Conditional prediction
        noise_cond = model(
            x, t, 
            attention_mask=attn_mask,
            motif_mask=motif_mask,
            motif_features=motif_angles
        )
        
        # Unconditional prediction
        noise_uncond = model(
            x, t,
            attention_mask=attn_mask,
            motif_mask=torch.zeros_like(motif_mask) if motif_mask is not None else None,
            motif_features=None
        )
        
        # Guided prediction
        model_output = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    else:
        # Standard conditional prediction
        model_output = model(
            x, t,
            attention_mask=attn_mask,
            motif_mask=motif_mask,
            motif_features=motif_angles
        )
    
    # Compute mean using predicted noise
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        x_prev = model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
    
    # Replace motif regions with clean angles (inpainting)
    if motif_mask is not None and motif_angles is not None:
        # Expand mask to match feature dimension
        mask_expanded = motif_mask.expand_as(x_prev)
        x_prev = (1 - mask_expanded) * x_prev + mask_expanded * motif_angles
    
    return x_prev


@torch.no_grad()
def p_sample_loop_conditional(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
    motif_mask: Optional[torch.Tensor] = None,
    motif_angles: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    disable_pbar: bool = False,
    return_history: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Sample from diffusion model with motif conditioning.
    
    Args:
        model: Diffusion model
        lengths: Sequence lengths for each sample
        noise: Initial noise [batch, seq_len, features]
        timesteps: Number of diffusion timesteps
        betas: Beta schedule
        is_angle: Which features are angular
        motif_mask: [batch, seq_len, 1] binary mask
        motif_angles: [batch, seq_len, features] clean motif angles
        guidance_scale: Classifier-free guidance scale
        disable_pbar: Disable progress bar
        return_history: Return full sampling trajectory
        
    Returns:
        final_sample: [batch, seq_len, features] final denoised sample
        history: (optional) List of samples at each timestep
    """
    device = next(model.parameters()).device
    b = noise.shape[0]
    img = noise.to(device)
    
    # Move motif data to device
    if motif_mask is not None:
        motif_mask = motif_mask.to(device)
    if motif_angles is not None:
        motif_angles = motif_angles.to(device)
    
    logging.info(
        f"Starting conditional sampling with noise {noise.shape}, "
        f"guidance_scale={guidance_scale}, "
        f"motif_regions={motif_mask.sum().item() if motif_mask is not None else 0}"
    )
    
    imgs = [] if return_history else None
    
    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="Conditional sampling",
        total=timesteps,
        disable=disable_pbar,
    ):
        # Sample one step
        img = p_sample_conditional(
            model=model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),
            seq_lens=lengths,
            t_index=i,
            betas=betas,
            motif_mask=motif_mask,
            motif_angles=motif_angles,
            guidance_scale=guidance_scale,
        )
        
        # Wrap angular features
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        
        if return_history:
            imgs.append(img.cpu())
    
    if return_history:
        return img, imgs
    return img


def sample_conditional(
    model: nn.Module,
    train_dset,
    motif_regions: List[List[Tuple[int, int]]],
    motif_angles_list: List[torch.Tensor],
    n: int = 10,
    sweep_lengths: Optional[Tuple[int, int]] = (50, 128),
    batch_size: int = 512,
    guidance_scale: float = 2.0,
    feature_key: str = "angles",
    disable_pbar: bool = False,
) -> List[torch.Tensor]:
    """
    Sample protein scaffolds with specified motif regions.
    
    Args:
        model: Trained conditional diffusion model
        train_dset: Training dataset (for noise schedule)
        motif_regions: List of motif regions for each sample
                      Each element is a list of (start, end) tuples
        motif_angles_list: List of motif angles for each sample
        n: Number of samples per length
        sweep_lengths: (min_len, max_len) range of lengths to sample
        batch_size: Batch size for sampling
        guidance_scale: Classifier-free guidance scale
        feature_key: Feature key in dataset
        disable_pbar: Disable progress bar
        
    Returns:
        samples: List of sampled angle tensors
    """
    from foldingdiff.motif_scaffolding import create_motif_mask_from_regions
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get noise schedule from dataset
    betas = train_dset.alpha_beta_terms["betas"]
    is_angular = train_dset.feature_is_angular[feature_key]
    n_features = len(is_angular)
    
    # Prepare sampling
    if sweep_lengths is not None:
        min_len, max_len = sweep_lengths
        lengths_to_sample = list(range(min_len, max_len))
    else:
        lengths_to_sample = [len(angles) for angles in motif_angles_list]
    
    all_samples = []
    
    for length in tqdm(lengths_to_sample, desc="Sampling lengths", disable=disable_pbar):
        for sample_idx in range(n):
            # Get motif information for this sample
            if sample_idx < len(motif_regions):
                regions = motif_regions[sample_idx]
                motif_angs = motif_angles_list[sample_idx]
            else:
                # No motif for this sample
                regions = []
                motif_angs = torch.zeros(length, n_features)
            
            # Create motif mask
            motif_mask = create_motif_mask_from_regions(
                regions, length, train_dset.pad
            ).unsqueeze(0)  # Add batch dimension
            
            # Pad motif angles
            if motif_angs.shape[0] < train_dset.pad:
                motif_angs = torch.nn.functional.pad(
                    motif_angs,
                    (0, 0, 0, train_dset.pad - motif_angs.shape[0]),
                    value=0
                )
            motif_angs = motif_angs.unsqueeze(0)  # Add batch dimension
            
            # Sample noise
            noise = torch.randn(1, train_dset.pad, n_features)
            
            # Sample
            sample = p_sample_loop_conditional(
                model=model,
                lengths=[length],
                noise=noise,
                timesteps=train_dset.timesteps,
                betas=betas,
                is_angle=is_angular,
                motif_mask=motif_mask,
                motif_angles=motif_angs,
                guidance_scale=guidance_scale,
                disable_pbar=True,
            )
            
            # Trim to actual length
            sample = sample[0, :length, :].cpu()
            all_samples.append(sample)
    
    return all_samples


def compute_motif_preservation_metrics(
    generated_samples: List[torch.Tensor],
    target_motif_angles: List[torch.Tensor],
    motif_regions: List[List[Tuple[int, int]]],
) -> Dict[str, float]:
    """
    Compute metrics for how well motifs are preserved.
    
    Args:
        generated_samples: List of generated angle tensors
        target_motif_angles: List of target motif angle tensors
        motif_regions: List of motif regions for each sample
        
    Returns:
        metrics: Dictionary of preservation metrics
    """
    from foldingdiff import nerf
    from foldingdiff.motif_scaffolding import compute_motif_rmsd
    
    rmsds = []
    angle_errors = []
    
    for gen, target, regions in zip(
        generated_samples, target_motif_angles, motif_regions
    ):
        if len(regions) == 0:
            continue
        
        # Extract motif regions
        for start, end in regions:
            gen_motif = gen[start:end]
            tgt_motif = target[start:end]
            
            # Angle error
            angle_diff = torch.abs(gen_motif - tgt_motif)
            # Handle circular angles
            angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)
            angle_errors.append(angle_diff.mean().item())
            
            # RMSD (convert to coordinates first)
            # This is a simplified version - full implementation would use NERF
            # For now, just use angle error as proxy
    
    metrics = {
        "mean_angle_error": np.mean(angle_errors) if angle_errors else 0.0,
        "std_angle_error": np.std(angle_errors) if angle_errors else 0.0,
        "num_motifs": len(angle_errors),
    }
    
    return metrics
