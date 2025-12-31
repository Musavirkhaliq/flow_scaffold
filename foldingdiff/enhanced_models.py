"""
Enhanced models using rich structural embeddings.

These models use ProteinStructureEmbedding to incorporate
multiple modalities for better generation quality.
"""
import logging
from typing import *

import torch
from torch import nn

from foldingdiff.flow_models import (
    BertForFlowMatching,
    BertForFlowMatchingTraining,
)
from foldingdiff.embeddings import ProteinStructureEmbedding


class BertForFlowMatchingEnhanced(BertForFlowMatching):
    """
    Flow matching model with enhanced embeddings.
    
    Uses ProteinStructureEmbedding to combine:
    - Backbone angles
    - Backbone coordinates
    - Local frames
    - Pairwise distances
    - Amino acid types (optional)
    - Secondary structure (optional)
    
    Args:
        config: BERT configuration
        use_enhanced_embedding: Whether to use enhanced embeddings
        embedding_config: Configuration for ProteinStructureEmbedding
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        config,
        use_enhanced_embedding: bool = True,
        embedding_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Don't call super().__init__ yet - we need to modify the input layer
        from transformers.models.bert.modeling_bert import BertPreTrainedModel
        BertPreTrainedModel.__init__(self, config)
        
        self.config = config
        self.use_enhanced_embedding = use_enhanced_embedding
        
        # Store feature information
        self.ft_is_angular = kwargs.get('ft_is_angular', [True] * 6)
        self.n_inputs = len(self.ft_is_angular)
        self.ft_names = kwargs.get('ft_names', [f"ft{i}" for i in range(self.n_inputs)])
        
        # Motif conditioning
        self.use_motif_conditioning = kwargs.get('use_motif_conditioning', True)
        self.motif_embed_mode = kwargs.get('motif_embed_mode', 'replace')
        
        if use_enhanced_embedding:
            # Use rich embedding
            if embedding_config is None:
                embedding_config = {
                    'use_sequence': True,
                    'use_coords': True,
                    'use_local_frames': True,
                    'use_pairwise': True,
                    'use_secondary_structure': False,
                }
            
            self.input_embedder = ProteinStructureEmbedding(
                hidden_size=config.hidden_size,
                **embedding_config
            )
        else:
            # Use simple embedding (backward compatible)
            self.inputs_to_hidden_dim = nn.Linear(self.n_inputs, config.hidden_size)
        
        # Motif embedding (if using conditioning)
        if self.use_motif_conditioning:
            self.motif_embed = nn.Linear(self.n_inputs, config.hidden_size)
            if self.motif_embed_mode == 'concat':
                self.combine_motif = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # Rest of model architecture
        from foldingdiff.modelling import (
            BertEmbeddings,
            BertEncoder,
            AnglesPredictor,
            GaussianFourierProjection,
        )
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.token_decoder = AnglesPredictor(config.hidden_size, self.n_inputs)
        self.time_embed = GaussianFourierProjection(config.hidden_size)
        
        self.init_weights()
        
        logging.info(
            f"BertForFlowMatchingEnhanced: enhanced_embedding={use_enhanced_embedding}, "
            f"motif_conditioning={self.use_motif_conditioning}"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        aa_types: Optional[torch.Tensor] = None,
        secondary_structure: Optional[torch.Tensor] = None,
        motif_mask: Optional[torch.Tensor] = None,
        motif_features: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with enhanced features.
        
        Args:
            inputs: [batch, seq_len, n_features] angles
            timestep: [batch] or [batch, 1] time values
            attention_mask: [batch, seq_len] attention mask
            coords: [batch, seq_len, 4, 3] backbone coordinates (optional)
            aa_types: [batch, seq_len] amino acid types (optional)
            secondary_structure: [batch, seq_len, 3] SS predictions (optional)
            motif_mask: [batch, seq_len, 1] motif mask (optional)
            motif_features: [batch, seq_len, n_features] motif angles (optional)
            position_ids: [batch, seq_len] position IDs (optional)
        
        Returns:
            velocity: [batch, seq_len, n_features] predicted velocity field
        """
        # Embed inputs
        if self.use_enhanced_embedding:
            inputs_upscaled = self.input_embedder(
                angles=inputs,
                coords=coords,
                aa_types=aa_types,
                secondary_structure=secondary_structure,
                attn_mask=attention_mask
            )
        else:
            inputs_upscaled = self.inputs_to_hidden_dim(inputs)
        
        # Add motif conditioning if provided
        if self.use_motif_conditioning and motif_mask is not None:
            if motif_features is None:
                motif_features = torch.zeros_like(inputs)
            
            # Embed motif features (using simple embedding, not enhanced)
            motif_embed = self.motif_embed(motif_features)
            
            # Expand motif_mask
            motif_mask_expanded = motif_mask.expand(-1, -1, inputs_upscaled.shape[-1])
            
            if self.motif_embed_mode == 'replace':
                inputs_upscaled = (
                    (1 - motif_mask_expanded) * inputs_upscaled + 
                    motif_mask_expanded * motif_embed
                )
            elif self.motif_embed_mode == 'add':
                inputs_upscaled = inputs_upscaled + motif_mask_expanded * motif_embed
            elif self.motif_embed_mode == 'concat':
                combined = torch.cat([inputs_upscaled, motif_embed], dim=-1)
                inputs_upscaled = self.combine_motif(combined)
        
        # Continue with standard forward pass
        batch_size, seq_length = inputs.shape[:2]
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_length).expand(batch_size, -1).to(inputs.device)
        
        # Pass through embeddings
        inputs_upscaled = self.embeddings(inputs_upscaled, position_ids=position_ids)
        
        # Add time encoding
        if timestep.ndim > 1:
            timestep = timestep.squeeze(-1)
        time_encoded = self.time_embed(timestep).unsqueeze(1)
        inputs_with_time = inputs_upscaled + time_encoded
        
        # Prepare attention mask
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(attention_mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Encoder
        encoder_outputs = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
        )
        
        # Decode
        sequence_output = encoder_outputs[0]
        per_token_decoded = self.token_decoder(sequence_output)
        
        return per_token_decoded


class BertForFlowMatchingEnhancedTraining(BertForFlowMatchingEnhanced, BertForFlowMatchingTraining):
    """
    Training wrapper for enhanced flow matching model.
    
    Combines enhanced embeddings with flow matching training.
    
    Args:
        lr: Learning rate
        embedding_config: Configuration for ProteinStructureEmbedding
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        lr: float = 5e-5,
        embedding_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize enhanced model
        BertForFlowMatchingEnhanced.__init__(
            self,
            embedding_config=embedding_config,
            **kwargs
        )
        
        # Initialize training components (from BertForFlowMatchingTraining)
        self.learning_rate = lr
        self.l2_lambda = kwargs.get('l2', 0.0)
        self.l1_lambda = kwargs.get('l1', 0.0)
        self.epochs = kwargs.get('epochs', 1)
        self.steps_per_epoch = kwargs.get('steps_per_epoch', 250)
        self.lr_scheduler = kwargs.get('lr_scheduler', None)
        self.guidance_dropout = kwargs.get('guidance_dropout', 0.1)
        
        # Flow matching schedule
        from foldingdiff.flow_matching import FlowMatchingSchedule, ConditionalFlowMatching
        self.flow_schedule = FlowMatchingSchedule()
        self.conditional_flow = ConditionalFlowMatching()
        
        # Epoch counters
        import time
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()
        
        # Loss tracking for debugging
        self.train_losses = []
        self.val_losses = []
        
        logging.info(
            f"Enhanced flow matching training: lr={lr}, "
            f"guidance_dropout={self.guidance_dropout}"
        )
    
    def training_step(self, batch, batch_idx):
        """
        Training step with enhanced features.
        
        Extracts enhanced features from batch and passes to model.
        """
        # Apply guidance dropout
        batch = self._apply_guidance_dropout(batch)
        
        x_0 = batch['angles']
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample time with importance weighting
        t = self.flow_schedule.sample_time(
            batch_size, device,
            importance_weighting=True,
            alpha=2.0
        )
        
        # Sample noise
        x_1 = torch.randn_like(x_0)
        
        # Get interpolant with smooth transitions
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
        
        # Predict velocity with enhanced features
        v_pred = self.forward(
            x_t, t,
            attention_mask=batch['attn_mask'],
            coords=batch.get('coords_computed', None),
            aa_types=batch.get('aa_types', None),
            secondary_structure=batch.get('secondary_structure', None),
            motif_mask=batch.get('motif_mask', None),
            motif_features=batch.get('motif_angles', None),
        )
        
        # Compute loss with scaffold weighting and diversity regularization
        from foldingdiff.flow_matching import compute_angular_flow_matching_loss
        loss = compute_angular_flow_matching_loss(
            v_pred, v_target,
            is_angular=self.ft_is_angular,
            mask=batch['attn_mask'],
            motif_mask=batch.get('motif_mask', None),
            scaffold_weight=2.0,
            diversity_samples=x_0,  # Add diversity regularization
            diversity_weight=0.01  # Small weight to encourage diversity
        )
        
        # L1 regularization
        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            loss += self.l1_lambda * l1_penalty
        
        # Compute batch diversity for monitoring
        batch_size = x_0.shape[0]
        if batch_size > 1:
            from foldingdiff.flow_matching import compute_diversity_penalty
            similarity_penalty = compute_diversity_penalty(
                x_0, batch['attn_mask'], diversity_weight=1.0
            )
            batch_diversity = 1.0 - similarity_penalty.item()
        else:
            batch_diversity = 0.0
        
        # Log
        log_dict = {
            'train_loss': loss,
            'train_batch_diversity': batch_diversity,  # Monitor diversity
        }
        if 'motif_mask' in batch:
            motif_ratio = batch['motif_mask'].sum() / batch['attn_mask'].sum()
            log_dict['train_motif_ratio'] = motif_ratio
        
        self.log_dict(log_dict)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with enhanced features"""
        with torch.no_grad():
            x_0 = batch['angles']
            batch_size = x_0.shape[0]
            device = x_0.device
            
            # Sample time with importance weighting
            t = self.flow_schedule.sample_time(
                batch_size, device,
                importance_weighting=True,
                alpha=2.0
            )
            
            # Sample noise
            x_1 = torch.randn_like(x_0)
            
            # Get interpolant with smooth transitions
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
                coords=batch.get('coords_computed', None),
                aa_types=batch.get('aa_types', None),
                secondary_structure=batch.get('secondary_structure', None),
                motif_mask=batch.get('motif_mask', None),
                motif_features=batch.get('motif_angles', None),
            )
            
            # Compute loss with scaffold weighting
            from foldingdiff.flow_matching import compute_angular_flow_matching_loss
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
