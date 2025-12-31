"""
Enhanced embedding strategies for protein structures.

This module provides rich embeddings that combine multiple modalities:
- Backbone angles (existing)
- Backbone coordinates (N, CA, C, O positions)
- Local coordinate frames (orientation)
- Pairwise distances (geometric context)
- Amino acid types (sequence information)
- Secondary structure (structural context)

These richer representations enable better generation quality.
"""
import logging
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProteinStructureEmbedding(nn.Module):
    """
    Rich embedding for protein structures combining multiple modalities.
    
    Combines:
    - Backbone angles (6D): phi, psi, omega, tau, CA:C:1N, C:1N:1CA
    - Backbone coordinates (12D): N, CA, C, O positions
    - Local frames (9D): 3 orthonormal orientation vectors
    - Pairwise features: Distance-based aggregation
    - Amino acid types (20D): One-hot encoding
    - Secondary structure (3D): Helix, sheet, coil
    
    Args:
        hidden_size: Output embedding dimension
        use_sequence: Include amino acid type embeddings
        use_coords: Include coordinate embeddings
        use_local_frames: Include local frame embeddings
        use_pairwise: Include pairwise distance features
        use_secondary_structure: Include secondary structure
    """
    
    def __init__(
        self,
        hidden_size: int = 384,
        use_sequence: bool = True,
        use_coords: bool = True,
        use_local_frames: bool = True,
        use_pairwise: bool = True,
        use_secondary_structure: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_sequence = use_sequence
        self.use_coords = use_coords
        self.use_local_frames = use_local_frames
        self.use_pairwise = use_pairwise
        self.use_secondary_structure = use_secondary_structure
        
        # Angle embedding (existing - always used)
        self.angle_embed = nn.Linear(6, hidden_size // 4)
        
        # Sequence embedding
        if use_sequence:
            self.aa_embed = nn.Embedding(21, hidden_size // 4)  # 20 AA + unknown
        
        # Coordinate embedding
        if use_coords:
            self.coord_embed = nn.Linear(12, hidden_size // 4)  # 4 atoms × 3D
        
        # Local frame embedding
        if use_local_frames:
            self.frame_embed = nn.Linear(9, hidden_size // 4)  # 3 vectors × 3D
        
        # Secondary structure embedding
        if use_secondary_structure:
            self.ss_embed = nn.Linear(3, hidden_size // 8)  # helix, sheet, coil
        
        # Calculate total dimension
        total_dim = hidden_size // 4  # angles (always included)
        if use_sequence:
            total_dim += hidden_size // 4
        if use_coords:
            total_dim += hidden_size // 4
        if use_local_frames:
            total_dim += hidden_size // 4
        if use_secondary_structure:
            total_dim += hidden_size // 8
        
        # Combine all embeddings
        self.combine = nn.Linear(total_dim, hidden_size)
        
        # Pairwise features (computed in forward)
        if use_pairwise:
            self.pairwise_embed = PairwiseFeatureEmbedding(hidden_size)
        
        # Initialize weights properly
        self._init_weights()
        
        logging.info(
            f"ProteinStructureEmbedding: hidden_size={hidden_size}, "
            f"sequence={use_sequence}, coords={use_coords}, "
            f"frames={use_local_frames}, pairwise={use_pairwise}"
        )
    
    def _init_weights(self):
        """Initialize weights with proper scaling for deep networks."""
        # Xavier/Glorot initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        angles: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        aa_types: Optional[torch.Tensor] = None,
        secondary_structure: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass combining all modalities.
        
        Args:
            angles: [batch, seq_len, 6] backbone angles
            coords: [batch, seq_len, 4, 3] backbone atom coordinates (N, CA, C, O)
            aa_types: [batch, seq_len] amino acid type indices (0-19, 20=unknown)
            secondary_structure: [batch, seq_len, 3] SS probabilities (H, E, C)
            attn_mask: [batch, seq_len] attention mask
        
        Returns:
            embeddings: [batch, seq_len, hidden_size]
        """
        embeddings = []
        
        # Angle embedding (always included)
        angle_emb = self.angle_embed(angles)
        embeddings.append(angle_emb)
        
        # Auto-compute coords if needed but not provided
        if (self.use_coords or self.use_local_frames or self.use_pairwise) and coords is None:
            coords = angles_to_coords_simple(angles)
        
        # Sequence embedding
        if self.use_sequence:
            if aa_types is not None:
                aa_emb = self.aa_embed(aa_types)
            else:
                # Use unknown token (20) if not provided
                batch_size, seq_len = angles.shape[:2]
                aa_types = torch.full((batch_size, seq_len), 20, 
                                     dtype=torch.long, device=angles.device)
                aa_emb = self.aa_embed(aa_types)
            embeddings.append(aa_emb)
        
        # Coordinate embedding
        if self.use_coords:
            if coords is not None:
                # Flatten coordinates: [batch, seq_len, 4, 3] -> [batch, seq_len, 12]
                coords_flat = coords.reshape(coords.shape[0], coords.shape[1], -1)
                coord_emb = self.coord_embed(coords_flat)
                embeddings.append(coord_emb)
        
        # Local frame embedding
        if self.use_local_frames:
            if coords is not None:
                frames = self.compute_local_frames(coords)
                frame_emb = self.frame_embed(frames)
                embeddings.append(frame_emb)
        
        # Secondary structure embedding
        if self.use_secondary_structure:
            if secondary_structure is None:
                # Auto-compute from angles if not provided
                secondary_structure = compute_secondary_structure_simple(angles)
            ss_emb = self.ss_embed(secondary_structure)
            embeddings.append(ss_emb)
        
        # Combine all embeddings
        combined = torch.cat(embeddings, dim=-1)
        output = self.combine(combined)
        
        # Add pairwise features
        if self.use_pairwise and coords is not None:
            output = self.pairwise_embed(output, coords, attn_mask)
        
        return output
    
    def compute_local_frames(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute local coordinate frames from backbone atoms.
        
        Creates orthonormal frame at each residue using:
        - e1: CA -> C direction
        - e2: perpendicular to CA-C and CA-N plane
        - e3: perpendicular to e1 and e2
        
        Args:
            coords: [batch, seq_len, 4, 3] (N, CA, C, O)
        
        Returns:
            frames: [batch, seq_len, 9] (3 orthonormal vectors flattened)
        """
        # Extract atoms
        N = coords[:, :, 0, :]   # [batch, seq_len, 3]
        CA = coords[:, :, 1, :]
        C = coords[:, :, 2, :]
        
        # Compute local frame
        # e1: CA -> C direction
        e1 = C - CA
        e1 = F.normalize(e1, dim=-1, eps=1e-8)
        
        # e2: perpendicular to CA-C and CA-N plane
        ca_n = N - CA
        ca_n = F.normalize(ca_n, dim=-1, eps=1e-8)
        e2 = torch.cross(e1, ca_n, dim=-1)
        e2 = F.normalize(e2, dim=-1, eps=1e-8)
        
        # e3: perpendicular to e1 and e2
        e3 = torch.cross(e1, e2, dim=-1)
        e3 = F.normalize(e3, dim=-1, eps=1e-8)
        
        # Concatenate: [batch, seq_len, 9]
        frames = torch.cat([e1, e2, e3], dim=-1)
        return frames


class PairwiseFeatureEmbedding(nn.Module):
    """
    Add pairwise geometric features via attention-like mechanism.
    
    Computes pairwise distances between CA atoms and aggregates
    distance-based features for each residue.
    
    Args:
        hidden_size: Embedding dimension
        num_distance_bins: Number of bins for distance discretization
        max_distance: Maximum distance to consider (Angstroms)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_distance_bins: int = 32,
        max_distance: float = 20.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_distance_bins = num_distance_bins
        self.max_distance = max_distance
        
        # Distance bins (learnable or fixed)
        self.register_buffer(
            'dist_bins',
            torch.linspace(0, max_distance, num_distance_bins)
        )
        
        # Distance embedding
        self.dist_embed = nn.Embedding(num_distance_bins, hidden_size)
        
        # Combine with node features
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        
        logging.info(
            f"PairwiseFeatureEmbedding: bins={num_distance_bins}, "
            f"max_dist={max_distance}"
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add pairwise distance features to node features.
        
        Args:
            node_features: [batch, seq_len, hidden_size]
            coords: [batch, seq_len, 4, 3] backbone coordinates
            attn_mask: [batch, seq_len] attention mask
        
        Returns:
            output: [batch, seq_len, hidden_size] with pairwise features
        """
        batch_size, seq_len = node_features.shape[:2]
        
        # Compute pairwise distances (CA-CA)
        ca_coords = coords[:, :, 1, :]  # [batch, seq_len, 3]
        
        # Pairwise distance matrix
        diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)
        # [batch, seq_len, seq_len, 3]
        
        distances = torch.norm(diff, dim=-1)  # [batch, seq_len, seq_len]
        
        # Bin distances
        dist_bins = torch.bucketize(distances, self.dist_bins)
        dist_bins = torch.clamp(dist_bins, 0, self.num_distance_bins - 1)
        dist_emb = self.dist_embed(dist_bins)  # [batch, seq_len, seq_len, hidden]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1) * attn_mask.unsqueeze(2)
            dist_emb = dist_emb * mask.unsqueeze(-1)
        
        # Aggregate pairwise features (mean pooling)
        pairwise_features = dist_emb.mean(dim=2)  # [batch, seq_len, hidden]
        
        # Combine with node features
        combined = torch.cat([node_features, pairwise_features], dim=-1)
        output = self.combine(combined)
        
        return output


class RelativePositionEmbedding(nn.Module):
    """
    Relative position embeddings for sequence positions.
    
    Instead of absolute positions, encodes relative distances
    between residues, which is more useful for proteins.
    
    Args:
        hidden_size: Embedding dimension
        max_relative_position: Maximum relative position to encode
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_relative_position: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        # Range: [-max_relative_position, max_relative_position]
        num_embeddings = 2 * max_relative_position + 1
        self.rel_pos_embed = nn.Embedding(num_embeddings, hidden_size)
        
        logging.info(
            f"RelativePositionEmbedding: max_rel_pos={max_relative_position}"
        )
    
    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute relative position embeddings.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
        
        Returns:
            rel_pos_emb: [seq_len, seq_len, hidden_size]
        """
        # Create relative position matrix
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clip to max range
        rel_pos = torch.clamp(
            rel_pos,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to positive indices
        rel_pos = rel_pos + self.max_relative_position
        
        # Embed
        rel_pos_emb = self.rel_pos_embed(rel_pos)
        
        return rel_pos_emb


def angles_to_coords_simple(
    angles: torch.Tensor,
    bond_lengths: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Convert backbone angles to approximate coordinates.
    
    This is a simplified version for embedding purposes.
    For accurate reconstruction, use the full NERF algorithm.
    
    Args:
        angles: [batch, seq_len, 6] (phi, psi, omega, tau, CA:C:1N, C:1N:1CA)
        bond_lengths: Optional dict of bond lengths
    
    Returns:
        coords: [batch, seq_len, 4, 3] approximate (N, CA, C, O) coordinates
    """
    if bond_lengths is None:
        # Standard bond lengths (Angstroms)
        bond_lengths = {
            'N-CA': 1.46,
            'CA-C': 1.52,
            'C-N': 1.33,
            'C-O': 1.23,
        }
    
    batch_size, seq_len, _ = angles.shape
    device = angles.device
    
    # Initialize coordinates
    coords = torch.zeros(batch_size, seq_len, 4, 3, device=device)
    
    # Place first residue at origin
    coords[:, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0], device=device)  # N
    coords[:, 0, 1, :] = torch.tensor([bond_lengths['N-CA'], 0.0, 0.0], device=device)  # CA
    coords[:, 0, 2, :] = torch.tensor([
        bond_lengths['N-CA'] + bond_lengths['CA-C'], 0.0, 0.0
    ], device=device)  # C
    coords[:, 0, 3, :] = torch.tensor([
        bond_lengths['N-CA'] + bond_lengths['CA-C'], bond_lengths['C-O'], 0.0
    ], device=device)  # O
    
    # For subsequent residues, use simplified geometry
    # This is approximate - for accurate coords, use NERF
    for i in range(1, seq_len):
        # Use previous C position and angles to place next residue
        prev_c = coords[:, i-1, 2, :]
        
        # Simplified placement (not geometrically accurate)
        # Just for embedding purposes
        coords[:, i, 0, :] = prev_c + torch.tensor([bond_lengths['C-N'], 0, 0], device=device)
        coords[:, i, 1, :] = coords[:, i, 0, :] + torch.tensor([bond_lengths['N-CA'], 0, 0], device=device)
        coords[:, i, 2, :] = coords[:, i, 1, :] + torch.tensor([bond_lengths['CA-C'], 0, 0], device=device)
        coords[:, i, 3, :] = coords[:, i, 2, :] + torch.tensor([0, bond_lengths['C-O'], 0], device=device)
    
    return coords


def compute_secondary_structure_simple(
    angles: torch.Tensor,
) -> torch.Tensor:
    """
    Predict secondary structure from backbone angles (simplified).
    
    Uses simple angle-based heuristics:
    - Helix: phi ~ -60°, psi ~ -45°
    - Sheet: phi ~ -120°, psi ~ 120°
    - Coil: everything else
    
    Args:
        angles: [batch, seq_len, 6] backbone angles
    
    Returns:
        ss_probs: [batch, seq_len, 3] (helix, sheet, coil) probabilities
    """
    phi = angles[:, :, 0]  # [batch, seq_len]
    psi = angles[:, :, 1]
    
    # Convert to degrees for easier thresholds
    phi_deg = phi * 180 / np.pi
    psi_deg = psi * 180 / np.pi
    
    # Helix region: phi ~ -60, psi ~ -45
    helix_score = torch.exp(
        -((phi_deg + 60)**2 + (psi_deg + 45)**2) / 1000
    )
    
    # Sheet region: phi ~ -120, psi ~ 120
    sheet_score = torch.exp(
        -((phi_deg + 120)**2 + (psi_deg - 120)**2) / 1000
    )
    
    # Coil: everything else
    coil_score = 1.0 - helix_score - sheet_score
    coil_score = torch.clamp(coil_score, min=0.0)
    
    # Stack and normalize
    ss_probs = torch.stack([helix_score, sheet_score, coil_score], dim=-1)
    ss_probs = F.softmax(ss_probs, dim=-1)
    
    return ss_probs
