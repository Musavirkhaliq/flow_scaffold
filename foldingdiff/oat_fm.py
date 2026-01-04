"""
Optimal Acceleration Transport for Flow Matching (OAT-FM)

Reference: arXiv:2509.24936
Improves flow matching by optimizing acceleration transport in the product space
of sample and velocity, ensuring smoother and more accurate probability paths.

CRITICAL FIX: OAT-FM must use periodic_diff for angular features to respect
torus geometry. Standard OAT is designed for Euclidean space; applying it to
angular data without geodesic logic causes the model to "flow" through the
middle of the circle (forbidden interior) rather than around the circumference.
"""
import torch
import torch.nn as nn
from typing import Optional, List


def periodic_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute periodic difference for circular angles.
    
    Handles the fact that -179° and +179° are only 2° apart, not 358° apart.
    """
    diff = pred - target
    # Wrap to [-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff


class OptimalAccelerationTransportFM:
    """
    Optimal Acceleration Transport for Flow Matching (OAT-FM)
    
    CRITICAL FIX: Uses periodic_diff for angular features to respect torus geometry.
    Standard OAT acceleration formula applied to angular data without geodesic
    shortest-path logic will try to "flow" through the middle of the circle.
    
    Enhances flow matching by focusing on acceleration transport,
    ensuring smoother and more accurate probability paths.
    
    Args:
        alpha: Acceleration coefficient (default: 0.5)
        is_angular: List indicating which features are angular (circular)
    """
    
    def __init__(self, alpha: float = 0.5, is_angular: Optional[List[bool]] = None):
        self.alpha = alpha
        self.is_angular = is_angular if is_angular is not None else []
    
    def get_oat_interpolant(
        self, 
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        t: torch.Tensor,
        is_angular: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        OAT-FM interpolant with acceleration optimization.
        
        CRITICAL FIX: Uses geodesic interpolation for angular features.
        
        Args:
            x_0: Starting point (data)
            x_1: Ending point (noise)
            t: Time parameter [0, 1]
            is_angular: List indicating which features are angular
        
        Returns:
            Interpolated sample
        """
        if is_angular is None:
            is_angular = self.is_angular
        
        # Ensure t has correct shape
        if t.ndim == 1:
            t = t.view(-1, 1, 1)
        elif t.ndim == 2:
            t = t.view(-1, 1, 1)
        
        # For angular features, use geodesic interpolation (shortest path on circle)
        # For non-angular features, use standard linear interpolation
        if is_angular and any(is_angular):
            is_angular_tensor = torch.tensor(is_angular, device=x_0.device, dtype=torch.bool)
            if is_angular_tensor.ndim == 1:
                is_angular_tensor = is_angular_tensor.view(1, 1, -1)
            angular_mask = is_angular_tensor.expand_as(x_0)
            
            # Geodesic interpolation for angular features
            diff_angular = periodic_diff(x_1, x_0)
            x_t_angular = x_0 + t * diff_angular
            x_t_angular = torch.atan2(torch.sin(x_t_angular), torch.cos(x_t_angular))
            
            # Standard linear interpolation for non-angular features
            x_t_linear = (1 - t) * x_0 + t * x_1
            
            # Combine based on angular mask
            x_t = torch.where(angular_mask, x_t_angular, x_t_linear)
        else:
            # Standard linear interpolant
            x_t = (1 - t) * x_0 + t * x_1
        
        # Add acceleration term for smoother paths (only for non-angular or using geodesic)
        # For angular features, acceleration is already handled by geodesic path
        if not (is_angular and any(is_angular)):
            # Acceleration is proportional to (x_1 - x_0) * (1 - 2t)
            acceleration = (x_1 - x_0) * (1 - 2 * t)
            # Add acceleration component weighted by t(1-t) to ensure
            # it's zero at boundaries (t=0 and t=1)
            x_t = x_t + self.alpha * acceleration * t * (1 - t)
        
        return x_t
    
    def get_oat_velocity(
        self, 
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        t: torch.Tensor,
        is_angular: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        OAT-FM velocity with acceleration component.
        
        CRITICAL FIX: Uses periodic_diff for angular features to respect torus geometry.
        
        Args:
            x_0: Starting point (data)
            x_1: Ending point (noise)
            t: Time parameter [0, 1]
            is_angular: List indicating which features are angular
        
        Returns:
            Velocity vector
        """
        if is_angular is None:
            is_angular = self.is_angular
        
        # Ensure t has correct shape
        if t.ndim == 1:
            t = t.view(-1, 1, 1)
        elif t.ndim == 2:
            t = t.view(-1, 1, 1)
        
        # For angular features, use periodic difference (geodesic velocity)
        # For non-angular features, use standard Euclidean difference
        if is_angular and any(is_angular):
            is_angular_tensor = torch.tensor(is_angular, device=x_0.device, dtype=torch.bool)
            if is_angular_tensor.ndim == 1:
                is_angular_tensor = is_angular_tensor.view(1, 1, -1)
            angular_mask = is_angular_tensor.expand_as(x_0)
            
            # Geodesic velocity: periodic difference (constant along geodesic path)
            v_angular = periodic_diff(x_1, x_0)
            
            # Standard velocity for non-angular
            v_linear = x_1 - x_0
            
            # Combine
            v = torch.where(angular_mask, v_angular, v_linear)
        else:
            # Standard velocity
            v = x_1 - x_0
        
        # Add acceleration component (only for non-angular features)
        # For angular features, geodesic velocity is already constant
        if not (is_angular and any(is_angular)):
            # Derivative of acceleration term: d/dt [alpha * (x_1 - x_0) * (1 - 2t) * t(1-t)]
            # = alpha * (x_1 - x_0) * (1 - 4t + 2t^2)
            acceleration_component = (x_1 - x_0) * (1 - 4 * t + 2 * t ** 2)
            v = v + self.alpha * acceleration_component
        
        return v
    
    def get_oat_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OAT-FM loss.
        
        Args:
            v_pred: Predicted velocity
            v_target: Target velocity (from OAT-FM)
            x_0: Starting point
            x_1: Ending point
            t: Time parameter
        
        Returns:
            Loss value
        """
        # Standard flow matching loss
        loss = torch.mean((v_pred - v_target) ** 2)
        
        return loss

