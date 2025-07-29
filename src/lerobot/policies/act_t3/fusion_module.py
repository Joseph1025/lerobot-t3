#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from lerobot.policies.act_t3.configuration_act_t3 import FusionConfig


class SimpleConcatenationFusion(nn.Module):
    """Simple concatenation fusion module."""
    
    def __init__(self, config: FusionConfig, act_dim: int, t3_dim: int):
        super().__init__()
        self.config = config
        
        # Projection layers to ensure consistent dimensions
        self.act_projection = nn.Linear(act_dim, config.fusion_dim)
        self.t3_projection = nn.Linear(t3_dim, config.fusion_dim)
        
        # Final projection after concatenation
        self.final_projection = nn.Linear(config.fusion_dim * 2, config.fusion_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.fusion_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.fusion_dropout)
        
    def forward(self, act_features: Tensor, t3_features: Tensor,
                act_padding_mask: Optional[Tensor] = None,
                t3_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through simple concatenation fusion.
        
        Args:
            act_features: ACT features (B, L_act, D)
            t3_features: T3 features (B, L_t3, D)
            act_padding_mask: Optional padding mask for ACT features (unused in simple fusion)
            t3_padding_mask: Optional padding mask for T3 features (unused in simple fusion)
            
        Returns:
            Fused features (B, L_act, D)
        """
        # Fix ACT features shape if needed (batch and sequence dimensions might be swapped)
        if act_features.size(0) != t3_features.size(0):
            # Transpose ACT features to correct shape (B, L, D)
            act_features = act_features.transpose(0, 1)
        
        # Project features to ensure consistent dimensions
        act_projected = self.act_projection(act_features)
        t3_projected = self.t3_projection(t3_features)
        
        # Handle different sequence lengths by pooling T3 features to match ACT
        if t3_projected.size(1) != act_projected.size(1):
            # Pool T3 features to match ACT sequence length
            t3_pooled = F.adaptive_avg_pool1d(
                t3_projected.transpose(1, 2), 
                act_projected.size(1)
            ).transpose(1, 2)
        else:
            t3_pooled = t3_projected
        
        # Simple concatenation along feature dimension
        concatenated = torch.cat([act_projected, t3_pooled], dim=-1)
        
        # Project to final dimension
        fused = self.final_projection(concatenated)
        
        # Apply normalization and dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused


class WeightedConcatenationFusion(nn.Module):
    """Weighted concatenation fusion with learnable modality weights."""
    
    def __init__(self, config: FusionConfig, act_dim: int, t3_dim: int):
        super().__init__()
        self.config = config
        
        # Learnable modality weights
        self.act_weight = nn.Parameter(torch.tensor(config.act_weight))
        self.t3_weight = nn.Parameter(torch.tensor(config.t3_weight))
        
        # Projection layers
        self.act_projection = nn.Linear(act_dim, config.fusion_dim)
        self.t3_projection = nn.Linear(t3_dim, config.fusion_dim)
        
        # Final projection
        self.final_projection = nn.Linear(config.fusion_dim * 2, config.fusion_dim)
        
        # Normalization
        self.norm = nn.LayerNorm(config.fusion_dim)
        self.dropout = nn.Dropout(config.fusion_dropout)
        
    def forward(self, act_features: Tensor, t3_features: Tensor,
                act_padding_mask: Optional[Tensor] = None,
                t3_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through weighted concatenation fusion.
        
        Args:
            act_features: ACT features (B, L_act, D)
            t3_features: T3 features (B, L_t3, D)
            act_padding_mask: Optional padding mask (unused)
            t3_padding_mask: Optional padding mask (unused)
            
        Returns:
            Fused features (B, L_act, D)
        """
        # Ensure weights sum to 1
        weights = F.softmax(torch.stack([self.act_weight, self.t3_weight]), dim=0)
        act_weight, t3_weight = weights[0], weights[1]
        
        # Project features
        act_projected = self.act_projection(act_features) * act_weight
        t3_projected = self.t3_projection(t3_features) * t3_weight
        
        # Handle sequence length mismatch
        if t3_projected.size(1) != act_projected.size(1):
            t3_pooled = F.adaptive_avg_pool1d(
                t3_projected.transpose(1, 2), 
                act_projected.size(1)
            ).transpose(1, 2)
        else:
            t3_pooled = t3_projected
        
        # Concatenate weighted features
        concatenated = torch.cat([act_projected, t3_pooled], dim=-1)
        
        # Final projection
        fused = self.final_projection(concatenated)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused


class ACTT3FusionModule(nn.Module):
    """Simplified fusion module using concatenation."""
    
    def __init__(self, config: FusionConfig, act_dim: int, t3_dim: int):
        super().__init__()
        self.config = config
        
        # Choose fusion type
        if config.fusion_type == "concat":
            self.fusion = SimpleConcatenationFusion(config, act_dim, t3_dim)
        elif config.fusion_type == "weighted_concat":
            self.fusion = WeightedConcatenationFusion(config, act_dim, t3_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {config.fusion_type}")
        
    def forward(self, act_features: Tensor, t3_features: Tensor,
                act_padding_mask: Optional[Tensor] = None,
                t3_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through concatenation fusion module.
        
        Args:
            act_features: ACT features (B, L_act, D_act)
            t3_features: T3 features (B, L_t3, D_t3)
            act_padding_mask: Optional padding mask for ACT features
            t3_padding_mask: Optional padding mask for T3 features
            
        Returns:
            Fused features (B, L_act, D_fusion)
        """
        return self.fusion(act_features, t3_features, act_padding_mask, t3_padding_mask) 