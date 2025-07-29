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
from torch import Tensor
from typing import Dict, Any, Optional

from lerobot.policies.act.modeling_act import ACTPolicy


class ACTFeatureExtractor:
    """Extract intermediate features from ACT policy without modifying it."""
    
    def __init__(self, act_policy: ACTPolicy):
        self.act_policy = act_policy
        self.feature_hooks = []
        self.extracted_features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to extract ACT features at different stages."""
        # Hook into ACT encoder output
        self.act_policy.model.encoder.register_forward_hook(
            self._save_encoder_features
        )
        
        # Hook into ACT decoder input
        self.act_policy.model.decoder.register_forward_hook(
            self._save_decoder_features
        )
        
        # Hook into vision backbone if present
        if hasattr(self.act_policy.model, 'backbone'):
            self.act_policy.model.backbone.register_forward_hook(
                self._save_backbone_features
            )
    
    def _save_encoder_features(self, module, input, output):
        """Save encoder features."""
        self.extracted_features['encoder_output'] = output
    
    def _save_decoder_features(self, module, input, output):
        """Save decoder features."""
        self.extracted_features['decoder_output'] = output
    
    def _save_backbone_features(self, module, input, output):
        """Save backbone features."""
        if isinstance(output, dict) and 'feature_map' in output:
            self.extracted_features['backbone_features'] = output['feature_map']
        else:
            self.extracted_features['backbone_features'] = output
    
    def extract_encoder_features(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Extract encoder features from ACT.
        
        Args:
            batch: Input batch for ACT
            
        Returns:
            Encoder features
        """
        # Clear previous features
        self.extracted_features.clear()
        
        # Run ACT forward pass to trigger hooks
        with torch.no_grad():
            _ = self.act_policy.model(batch)
        
        # Return encoder features
        if 'encoder_output' in self.extracted_features:
            features = self.extracted_features['encoder_output']
            return features
        else:
            raise ValueError("Failed to extract encoder features from ACT")
    
    def extract_all_features(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Extract all available features from ACT.
        
        Args:
            batch: Input batch for ACT
            
        Returns:
            Dictionary of extracted features
        """
        # Clear previous features
        self.extracted_features.clear()
        
        # Run ACT forward pass to trigger hooks
        with torch.no_grad():
            _ = self.act_policy.model(batch)
        
        return self.extracted_features.copy()
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of extracted features.
        
        Returns:
            Dictionary mapping feature names to dimensions
        """
        dimensions = {}
        
        for name, features in self.extracted_features.items():
            if isinstance(features, torch.Tensor):
                dimensions[name] = features.size(-1)
        
        return dimensions


class ACTFeatureProjector(nn.Module):
    """Project ACT features to target dimensions for fusion."""
    
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, target_dim)
        self.norm = nn.LayerNorm(target_dim)
        
    def forward(self, features: Tensor) -> Tensor:
        """
        Project features to target dimension.
        
        Args:
            features: Input features
            
        Returns:
            Projected features
        """
        projected = self.projection(features)
        normalized = self.norm(projected)
        return normalized


class ACTT3FeatureManager:
    """Manage feature extraction and projection for ACT-T3 policy."""
    
    def __init__(self, act_policy: ACTPolicy, target_dim: int):
        self.act_extractor = ACTFeatureExtractor(act_policy)
        self.target_dim = target_dim
        self.projectors = {}
        
    def setup_projectors(self, feature_dims: Dict[str, int]):
        """
        Setup projection layers for different feature types.
        
        Args:
            feature_dims: Dictionary mapping feature names to dimensions
        """
        for feature_name, input_dim in feature_dims.items():
            if input_dim != self.target_dim:
                self.projectors[feature_name] = ACTFeatureProjector(input_dim, self.target_dim)
    
    def extract_and_project_features(self, batch: Dict[str, Tensor], 
                                   feature_name: str = 'encoder_output') -> Tensor:
        """
        Extract and project ACT features.
        
        Args:
            batch: Input batch
            feature_name: Name of feature to extract
            
        Returns:
            Projected features
        """
        # Extract features
        all_features = self.act_extractor.extract_all_features(batch)
        
        if feature_name not in all_features:
            raise ValueError(f"Feature '{feature_name}' not found in extracted features")
        
        features = all_features[feature_name]
        
        # Project if needed
        if feature_name in self.projectors:
            features = self.projectors[feature_name](features)
        
        return features
    
    def get_act_features_for_fusion(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Get ACT features ready for fusion.
        
        Args:
            batch: Input batch
            
        Returns:
            ACT features ready for fusion
        """
        # Extract encoder features (most suitable for fusion)
        features = self.extract_and_project_features(batch, 'encoder_output')
        return features 