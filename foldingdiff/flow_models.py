"""
Flow matching models for protein generation.

These models predict velocity fields instead of noise,
enabling faster sampling through ODE integration.
"""
import logging
import time
from typing import *

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from foldingdiff.conditional_diffusion import BertForConditionalDiffusion
from foldingdiff.flow_matching import (
    FlowMatchingSchedule,
    ConditionalFlowMatching,
    compute_angular_flow_matching_loss,
)


class BertForFlowMatching(BertForConditionalDiffusion):
    """
    BERT model for flow matching.
    
    Extends the conditional diffusion model to predict velocity fields
    instead of noise. The architecture remains the same, only the
    training objective and sampling procedure change.
    
    Args:
        config: BERT configuration
        **kwargs: Additional arguments passed to BertForConditionalDiffusion
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        logging.info("BertForFlowMatching initialized (predicts velocity)")
    
    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: Optional[torch.Tensor] = None,
        motif_features: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass predicting velocity field.
        
        Args:
            inputs: Current state x_t [batch, seq_len, features]
            timestep: Time t ∈ [0,1] [batch] or [batch, 1]
            attention_mask: [batch, seq_len]
            motif_mask: [batch, seq_len, 1]
            motif_features: [batch, seq_len, features]
        
        Returns:
            velocity: Predicted velocity v_θ(x_t, t) [batch, seq_len, features]
        """
        # The base model forward pass is the same
        # Only interpretation changes: output is velocity not noise
        return super().forward(
            inputs=inputs,
            timestep=timestep,
            attention_mask=attention_mask,
            motif_mask=motif_mask,
            motif_features=motif_features,
            **kwargs
        )


class BertForFlowMatchingTraining(BertForFlowMatching, pl.LightningModule):
    """
    Training wrapper for flow matching model.
    
    Implements flow matching training objective:
    L = E_t,x_0,x_1 [||v_θ(x_t, t) - (x_1 - x_0)||²]
    
    Args:
        lr: Learning rate
        guidance_dropout: Probability of dropping conditioning
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        lr: float = 5e-5,
        l2: float = 0.0,
        l1: float = 0.0,
        epochs: int = 1,
        steps_per_epoch: int = 250,
        lr_scheduler: Optional[str] = None,
        guidance_dropout: float = 0.1,
        **kwargs
    ):
        BertForFlowMatching.__init__(self, **kwargs)
        
        self.learning_rate = lr
        self.l2_lambda = l2
        self.l1_lambda = l1
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler = lr_scheduler
        self.guidance_dropout = guidance_dropout
        
        # Flow matching schedule
        self.flow_schedule = FlowMatchingSchedule()
        self.conditional_flow = ConditionalFlowMatching()
        
        # Epoch counters
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()
        
        # Loss tracking for debugging
        self.train_losses = []
        self.val_losses = []
        
        logging.info(
            f"Flow matching training: lr={lr}, guidance_dropout={guidance_dropout}"
        )
    
    def _apply_guidance_dropout(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Randomly drop conditioning for classifier-free guidance.
        
        Improved strategy:
        - Drop entire motif mask (unconditional generation)
        - This enables classifier-free guidance during sampling
        """
        if 'motif_mask' in batch and torch.rand(1).item() < self.guidance_dropout:
            batch = batch.copy()
            # Zero out motif mask (unconditional)
            batch['motif_mask'] = torch.zeros_like(batch['motif_mask'])
            # Also zero out motif features to ensure unconditional generation
            if 'motif_angles' in batch:
                batch['motif_angles'] = torch.zeros_like(batch['motif_angles'])
        return batch
    
    def training_step(self, batch, batch_idx):
        """
        Flow matching training step.
        
        1. Sample time t ~ Uniform(0, 1)
        2. Sample noise x_1 ~ N(0, I)
        3. Compute interpolant x_t = (1-t)*x_0 + t*x_1
        4. Predict velocity v_θ(x_t, t)
        5. Compute loss ||v_θ - (x_1 - x_0)||²
        """
        # Apply guidance dropout
        batch = self._apply_guidance_dropout(batch)
        
        x_0 = batch['angles']  # Clean data
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample time with importance weighting and curriculum learning
        # Early training: focus on easier timesteps (t near 0)
        # Later training: use importance-weighted sampling (more samples near t=0 and t=1)
        if self.train_epoch_counter < self.epochs * 0.2:
            # Early training: focus on t near 0 (easier)
            t = torch.rand(batch_size, device=device) * 0.5
        elif self.train_epoch_counter < self.epochs * 0.5:
            # Mid training: gradually introduce full range
            t = torch.rand(batch_size, device=device) * 0.8 + 0.1
        else:
            # Later training: importance-weighted sampling
            t = self.flow_schedule.sample_time(
                batch_size, device, 
                importance_weighting=True, 
                alpha=2.0
            )
        
        # Sample noise
        x_1 = torch.randn_like(x_0)
        
        # Get interpolant
        if 'motif_mask' in batch and batch['motif_mask'].sum() > 0:
            # Conditional flow matching with smooth transitions
            x_t = self.conditional_flow.get_conditional_interpolant(
                x_0, x_1, t, batch['motif_mask'],
                smooth_transition=True,  # Enable smooth boundaries
                transition_width=0.1
            )
            v_target = self.conditional_flow.get_conditional_velocity(
                x_0, x_1, batch['motif_mask']
            )
        else:
            # Unconditional flow matching
            x_t = self.flow_schedule.get_interpolant(x_0, x_1, t)
            v_target = self.flow_schedule.get_target_velocity(x_0, x_1)
        
        # Predict velocity
        v_pred = self.forward(
            x_t, t,
            attention_mask=batch['attn_mask'],
            motif_mask=batch.get('motif_mask', None),
            motif_features=batch.get('motif_angles', None),
        )
        
        # Compute loss with angular handling and scaffold weighting
        loss = compute_angular_flow_matching_loss(
            v_pred, v_target,
            is_angular=self.ft_is_angular,
            mask=batch['attn_mask'],
            motif_mask=batch.get('motif_mask', None),
            scaffold_weight=2.0,  # Weight scaffold regions 2x more than motifs
            diversity_samples=x_0,  # Add diversity regularization
            diversity_weight=0.01  # Small weight to encourage diversity
        )
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss detected at batch {batch_idx}")
            # Skip this batch
            return None
        
        # L1 regularization
        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            loss += self.l1_lambda * l1_penalty
        
        # Track loss for debugging
        self.train_losses.append(loss.item())
        
        # Compute batch diversity for monitoring (invert similarity to get diversity)
        batch_size = x_0.shape[0]
        if batch_size > 1:
            from foldingdiff.flow_matching import compute_diversity_penalty
            # Get similarity penalty (higher = less diverse)
            similarity_penalty = compute_diversity_penalty(
                x_0, batch['attn_mask'], diversity_weight=1.0
            )
            # Convert to diversity metric (higher = more diverse)
            batch_diversity = 1.0 - similarity_penalty.item()
        else:
            batch_diversity = 0.0
        
        # Log with more details
        log_dict = {
            'train_loss': loss,
            'train_loss_raw': loss.item(),
            'mean_v_pred': v_pred.abs().mean(),
            'mean_v_target': v_target.abs().mean(),
            'train_batch_diversity': batch_diversity,  # Monitor diversity
        }
        if 'motif_mask' in batch:
            motif_ratio = batch['motif_mask'].sum() / batch['attn_mask'].sum()
            log_dict['train_motif_ratio'] = motif_ratio
        
        self.log_dict(log_dict)
        
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        """Log average training loss over epoch"""
        # Filter out None values (skipped batches)
        valid_outputs = [o for o in outputs if o is not None and "loss" in o]
        if not valid_outputs:
            logging.warning("No valid outputs in epoch!")
            return
            
        losses = torch.stack([o["loss"] for o in valid_outputs])
        mean_loss = torch.mean(losses)
        std_loss = torch.std(losses)
        t_delta = time.time() - self.train_epoch_last_time
        
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: "
            f"{mean_loss:.4f} ± {std_loss:.4f} ({t_delta:.2f} seconds)"
        )
        
        # Check if loss is decreasing
        if len(self.train_losses) > 100:
            recent_avg = np.mean(self.train_losses[-100:])
            older_avg = np.mean(self.train_losses[-200:-100]) if len(self.train_losses) > 200 else recent_avg
            if recent_avg >= older_avg * 0.99:  # Not decreasing by at least 1%
                pl.utilities.rank_zero_info(
                    f"⚠️  Loss not decreasing! Recent: {recent_avg:.4f}, Older: {older_avg:.4f}"
                )
        
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()
    
    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Validation step"""
        with torch.no_grad():
            x_0 = batch['angles']
            batch_size = x_0.shape[0]
            device = x_0.device
            
            # Sample time
            t = self.flow_schedule.sample_time(batch_size, device)
            
            # Sample noise
            x_1 = torch.randn_like(x_0)
            
            # Get interpolant and target
            if 'motif_mask' in batch and batch['motif_mask'].sum() > 0:
                x_t = self.conditional_flow.get_conditional_interpolant(
                    x_0, x_1, t, batch['motif_mask'],
                    smooth_transition=True,
                    transition_width=0.1
                )
                v_target = self.conditional_flow.get_conditional_velocity(
                    x_0, x_1, batch['motif_mask']
                )
            else:
                x_t = self.flow_schedule.get_interpolant(x_0, x_1, t)
                v_target = self.flow_schedule.get_target_velocity(x_0, x_1)
            
            # Predict velocity
            v_pred = self.forward(
                x_t, t,
                attention_mask=batch['attn_mask'],
                motif_mask=batch.get('motif_mask', None),
                motif_features=batch.get('motif_angles', None),
            )
            
            # Compute loss with scaffold weighting
            loss = compute_angular_flow_matching_loss(
                v_pred, v_target,
                is_angular=self.ft_is_angular,
                mask=batch['attn_mask'],
                motif_mask=batch.get('motif_mask', None),
                scaffold_weight=2.0
            )
        
        # Log
        log_dict = {'val_loss': loss}
        if 'motif_mask' in batch:
            motif_ratio = batch['motif_mask'].sum() / batch['attn_mask'].sum()
            log_dict['val_motif_ratio'] = motif_ratio
        
        self.log_dict(log_dict, rank_zero_only=True)
        
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs) -> None:
        """Log average validation loss over epoch"""
        losses = torch.stack([o["val_loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        pl.utilities.rank_zero_info(
            f"Valid loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f}"
        )
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                from transformers.optimization import get_linear_schedule_with_warmup
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval
