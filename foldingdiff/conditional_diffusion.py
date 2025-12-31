"""
Conditional diffusion models for motif scaffolding.

This module extends the base diffusion model to support conditional generation
where specific motif regions are preserved while scaffold regions are generated.
"""
import logging
import time
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from foldingdiff.modelling import (
    BertForDiffusionBase,
    BertForDiffusion,
    LOSS_KEYS,
    LR_SCHEDULE
)


class BertForConditionalDiffusion(BertForDiffusionBase):
    """
    BERT-based diffusion model with motif conditioning support.
    
    Extends the base diffusion model to accept motif masks and features,
    enabling conditional generation for motif scaffolding.
    
    Args:
        config: BERT configuration
        use_motif_conditioning: Whether to use motif conditioning
        motif_embed_mode: How to embed motif features ('add', 'concat', 'replace')
        **kwargs: Additional arguments passed to BertForDiffusionBase
    """
    
    def __init__(
        self,
        config,
        use_motif_conditioning: bool = True,
        motif_embed_mode: Literal['add', 'concat', 'replace'] = 'replace',
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.use_motif_conditioning = use_motif_conditioning
        self.motif_embed_mode = motif_embed_mode
        
        if use_motif_conditioning:
            # Motif embedding layer
            self.motif_embed = nn.Linear(self.n_inputs, config.hidden_size)
            
            if motif_embed_mode == 'concat':
                # If concatenating, need to adjust the input dimension
                self.inputs_to_hidden_dim = nn.Linear(
                    self.n_inputs, config.hidden_size // 2
                )
                self.combine_motif = nn.Linear(config.hidden_size, config.hidden_size)
            
            logging.info(
                f"Conditional diffusion with motif_embed_mode={motif_embed_mode}"
            )
    
    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: Optional[torch.Tensor] = None,
        motif_features: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass with optional motif conditioning.
        
        Args:
            inputs: [batch, seq_len, n_features] noisy angles
            timestep: [batch, 1] timestep values
            attention_mask: [batch, seq_len] attention mask
            motif_mask: [batch, seq_len, 1] binary mask (1=motif, 0=scaffold)
            motif_features: [batch, seq_len, n_features] clean motif angles
            
        Returns:
            per_token_decoded: [batch, seq_len, n_features] predicted noise
        """
        # Embed inputs
        inputs_upscaled = self.inputs_to_hidden_dim(inputs)
        
        # Add motif conditioning if provided
        if self.use_motif_conditioning and motif_mask is not None:
            if motif_features is None:
                # If no motif features provided, use zeros
                motif_features = torch.zeros_like(inputs)
            
            # Embed motif features
            motif_embed = self.motif_embed(motif_features)
            
            # Expand motif_mask to match hidden dimension
            # motif_mask: [batch, seq_len, 1] -> [batch, seq_len, hidden_size]
            motif_mask_expanded = motif_mask.expand(-1, -1, inputs_upscaled.shape[-1])
            
            if self.motif_embed_mode == 'replace':
                # Replace scaffold embeddings with motif embeddings where mask=1
                inputs_upscaled = (
                    (1 - motif_mask_expanded) * inputs_upscaled + 
                    motif_mask_expanded * motif_embed
                )
            elif self.motif_embed_mode == 'add':
                # Add motif embeddings (weighted by mask)
                inputs_upscaled = inputs_upscaled + motif_mask_expanded * motif_embed
            elif self.motif_embed_mode == 'concat':
                # Concatenate and combine
                combined = torch.cat([inputs_upscaled, motif_embed], dim=-1)
                inputs_upscaled = self.combine_motif(combined)
        
        # Continue with standard forward pass
        input_shape = inputs.size()
        batch_size, seq_length, *_ = input_shape
        
        # Position IDs
        if position_ids is None:
            position_ids = (
                torch.arange(seq_length)
                .expand(batch_size, -1)
                .type_as(timestep)
            )
        
        # Pass through embeddings
        inputs_upscaled = self.embeddings(inputs_upscaled, position_ids=position_ids)
        
        # Add time encoding
        time_encoded = self.time_embed(timestep.squeeze(dim=-1)).unsqueeze(1)
        inputs_with_time = inputs_upscaled + time_encoded
        
        # Prepare attention mask
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(attention_mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Encoder
        encoder_outputs = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Decode
        sequence_output = encoder_outputs[0]
        per_token_decoded = self.token_decoder(sequence_output)
        
        return per_token_decoded


class BertForConditionalDiffusionTraining(BertForConditionalDiffusion, pl.LightningModule):
    """
    Training wrapper for conditional diffusion with classifier-free guidance.
    
    Implements classifier-free guidance by randomly dropping conditioning
    during training, allowing the model to learn both conditional and
    unconditional generation.
    
    Args:
        guidance_dropout: Probability of dropping conditioning during training
        **kwargs: Arguments passed to BertForConditionalDiffusion and BertForDiffusion
    """
    
    def __init__(
        self,
        lr: float = 5e-5,
        loss: Union[Callable, LOSS_KEYS] = "smooth_l1",
        use_pairwise_dist_loss: Union[float, Tuple[float, float, int]] = 0.0,
        l2: float = 0.0,
        l1: float = 0.0,
        circle_reg: float = 0.0,
        epochs: int = 1,
        steps_per_epoch: int = 250,
        lr_scheduler: LR_SCHEDULE = None,
        write_preds_to_dir: Optional[str] = None,
        guidance_dropout: float = 0.1,
        **kwargs
    ):
        """Initialize conditional diffusion training model"""
        BertForConditionalDiffusion.__init__(self, **kwargs)
        
        # Store training parameters (from BertForDiffusion)
        self.learning_rate = lr
        self.guidance_dropout = guidance_dropout
        
        # Loss function setup (copied from BertForDiffusion)
        if isinstance(loss, str):
            logging.info(
                f"Mapping loss {loss} to list of losses corresponding to angular {self.ft_is_angular}"
            )
            if loss in self.loss_autocorrect_dict:
                logging.info(
                    "Autocorrecting {} to {}".format(
                        loss, self.loss_autocorrect_dict[loss]
                    )
                )
                loss = self.loss_autocorrect_dict[loss]
            self.loss_func = [
                self.angular_loss_fn_dict[loss]
                if is_angular
                else self.nonangular_loss_fn_dict[loss]
                for is_angular in self.ft_is_angular
            ]
        else:
            logging.warning(
                f"Using pre-given callable loss: {loss}. This may not handle angles correctly!"
            )
            self.loss_func = loss
        
        pl.utilities.rank_zero_info(f"Using loss: {self.loss_func}")
        if isinstance(self.loss_func, (tuple, list)):
            assert (
                len(self.loss_func) == self.n_inputs
            ), f"Got {len(self.loss_func)} loss functions, expected {self.n_inputs}"
        
        self.use_pairwise_dist_loss = use_pairwise_dist_loss
        self.l1_lambda = l1
        self.l2_lambda = l2
        self.circle_lambda = circle_reg
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler = lr_scheduler
        
        # Output directory
        self.write_preds_to_dir = write_preds_to_dir
        self.write_preds_counter = 0
        
        # Epoch counters
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()
        
        logging.info(
            f"Conditional diffusion training with guidance_dropout={guidance_dropout}"
        )
    
    def _apply_guidance_dropout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly drop conditioning for classifier-free guidance.
        
        Args:
            batch: Batch dictionary with motif_mask and motif_angles
            
        Returns:
            Modified batch with conditioning potentially dropped
        """
        if 'motif_mask' in batch and torch.rand(1).item() < self.guidance_dropout:
            # Drop conditioning by zeroing out motif mask
            batch = batch.copy()
            batch['motif_mask'] = torch.zeros_like(batch['motif_mask'])
        return batch
    
    def _get_loss_terms(
        self, batch, write_preds: Optional[str] = None
    ) -> List[torch.Tensor]:
        """
        Compute loss terms for conditional diffusion.
        
        This is adapted from BertForDiffusion._get_loss_terms to handle
        motif conditioning.
        """
        # Apply guidance dropout
        batch = self._apply_guidance_dropout(batch)
        
        known_noise = batch["known_noise"]
        
        # Forward pass with motif conditioning
        predicted_noise = self.forward(
            batch["corrupted"],
            batch["t"],
            attention_mask=batch["attn_mask"],
            position_ids=batch["position_ids"],
            motif_mask=batch.get("motif_mask", None),
            motif_features=batch.get("motif_angles", None),
        )
        
        assert (
            known_noise.shape == predicted_noise.shape
        ), f"{known_noise.shape} != {predicted_noise.shape}"
        
        # Compute loss only on unmasked regions
        unmask_idx = torch.where(batch["attn_mask"])
        assert len(unmask_idx) == 2
        
        loss_terms = []
        for i in range(known_noise.shape[-1]):
            loss_fn = (
                self.loss_func[i]
                if isinstance(self.loss_func, list)
                else self.loss_func
            )
            
            # Check if loss accepts circle_penalty
            import inspect
            loss_args = inspect.getfullargspec(loss_fn)
            if (
                "circle_penalty" in loss_args.args
                or "circle_penalty" in loss_args.kwonlyargs
            ):
                l = loss_fn(
                    predicted_noise[unmask_idx[0], unmask_idx[1], i],
                    known_noise[unmask_idx[0], unmask_idx[1], i],
                    circle_penalty=self.circle_lambda,
                )
            else:
                l = loss_fn(
                    predicted_noise[unmask_idx[0], unmask_idx[1], i],
                    known_noise[unmask_idx[0], unmask_idx[1], i],
                )
            loss_terms.append(l)
        
        # Write predictions if requested
        if write_preds is not None:
            import json
            with open(write_preds, "w") as f:
                d_to_write = {
                    "known_noise": known_noise.cpu().numpy().tolist(),
                    "predicted_noise": predicted_noise.cpu().numpy().tolist(),
                    "attn_mask": batch["attn_mask"].cpu().numpy().tolist(),
                    "losses": [l.item() for l in loss_terms],
                }
                if 'motif_mask' in batch:
                    d_to_write['motif_mask'] = batch['motif_mask'].cpu().numpy().tolist()
                json.dump(d_to_write, f)
        
        return torch.stack(loss_terms)
    
    def training_step(self, batch, batch_idx):
        """Training step with conditional diffusion"""
        loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss_terms)
        
        # L1 regularization
        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            avg_loss += self.l1_lambda * l1_penalty
        
        # Log losses
        pseudo_ft_names = self.ft_names
        loss_dict = {
            f"train_loss_{val_name}": val
            for val_name, val in zip(pseudo_ft_names, loss_terms)
        }
        loss_dict["train_loss"] = avg_loss
        
        # Log motif statistics if available
        if 'motif_mask' in batch:
            motif_ratio = batch['motif_mask'].sum() / batch['attn_mask'].sum()
            loss_dict['train_motif_ratio'] = motif_ratio
        
        self.log_dict(loss_dict)
        
        return avg_loss
    
    def training_epoch_end(self, outputs) -> None:
        """Log average training loss over epoch"""
        losses = torch.stack([o["loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)"
        )
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()
    
    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Validation step"""
        with torch.no_grad():
            loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss_terms)
        
        # Log losses
        pseudo_ft_names = self.ft_names
        loss_dict = {
            f"val_loss_{val_name}": self.all_gather(val)
            for val_name, val in zip(pseudo_ft_names, loss_terms)
        }
        loss_dict["val_loss"] = avg_loss
        
        # Log motif statistics
        if 'motif_mask' in batch:
            motif_ratio = batch['motif_mask'].sum() / batch['attn_mask'].sum()
            loss_dict['val_motif_ratio'] = motif_ratio
        
        self.log_dict(loss_dict, rank_zero_only=True)
        
        return {"val_loss": avg_loss}
    
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
