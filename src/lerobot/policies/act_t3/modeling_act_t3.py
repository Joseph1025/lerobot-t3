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
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot.policies.act_t3.configuration_act_t3 import ACTT3Config
from lerobot.policies.act_t3.t3_encoder import T3FeatureExtractor
from lerobot.policies.act_t3.fusion_module import ACTT3FusionModule
from lerobot.policies.act_t3.act_feature_extractor import ACTT3FeatureManager


class ACTT3Policy(PreTrainedPolicy):
    """
    ACT-T3 Policy that combines ACT and T3 without modifying the original ACT.
    
    This policy:
    1. Uses existing ACT policy (unmodified) for camera + action processing
    2. Adds T3 encoder for tactile data processing
    3. Fuses both modalities using cross-attention and modality mixing
    4. Generates final actions using ACT decoder
    """

    config_class = ACTT3Config
    name = "act_t3"

    def __init__(
        self,
        config: ACTT3Config,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        """
        Initialize ACT-T3 policy.
        
        Args:
            config: ACT-T3 configuration
            dataset_stats: Dataset statistics for normalization
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Normalization layers
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Create ACT policy (unmodified)
        act_config = config.get_act_config()
        self.act_policy = ACTPolicy(act_config, dataset_stats)

        # T3 feature extractor
        self.t3_extractor = T3FeatureExtractor(config.t3_config)
        
        # Set up T3 projection to match fusion dimension
        self.t3_extractor.set_projection(config.fusion_config.fusion_dim)

        # Feature manager for ACT
        self.feature_manager = ACTT3FeatureManager(
            self.act_policy, 
            config.fusion_config.fusion_dim
        )

        # Fusion module - will be updated with correct dimensions after feature extraction
        # For now, use placeholder dimensions that will be updated
        act_dim = config.act_config.dim_model  # ACT model dimension
        t3_dim = config.t3_config.tactile_embed_dim  # T3 embedding dimension
        self.fusion_module = ACTT3FusionModule(config.fusion_config, act_dim, t3_dim)

        # Action head (reuse from ACT)
        self.action_head = self.act_policy.model.action_head

        # Create the main model
        self.model = ACTT3Model(config)

        # Freeze components if specified
        if config.freeze_act:
            self._freeze_act_parameters()
        if config.freeze_t3:
            self._freeze_t3_parameters()

    def get_optim_params(self) -> dict:
        """Get optimizer parameters for ACT-T3 policy."""
        # Return optimizer parameters for the entire model
        return {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk for ACT-T3 policy."""
        # This is the main prediction method for training
        with torch.no_grad():
            action = self.predict_action(batch)
        return action

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for ACT-T3 policy."""
        # For ACT-T3, we use the same logic as predict_action
        return self.predict_action(batch)

    def reset(self):
        """Reset the policy state."""
        # Reset ACT policy
        self.act_policy.reset()
        # Reset feature manager
        self.feature_manager.reset()

    def _freeze_act_parameters(self):
        """Freeze ACT parameters."""
        for param in self.act_policy.parameters():
            param.requires_grad = False
        print("ACT parameters frozen")

    def _freeze_t3_parameters(self):
        """Freeze T3 parameters."""
        for param in self.t3_extractor.parameters():
            param.requires_grad = False
        print("T3 parameters frozen")

    def _prepare_act_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Prepare batch for ACT (remove tactile data).
        
        Args:
            batch: Full batch with tactile data
            
        Returns:
            Batch without tactile data for ACT
        """
        act_batch = {}
        for key, value in batch.items():
            if 'tactile' not in key.lower():
                act_batch[key] = value
        return act_batch

    def _extract_tactile_data(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Extract tactile data from batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Tactile data tensor
        """
        tactile_keys = [k for k in batch.keys() if 'tactile' in k.lower()]
        if not tactile_keys:
            raise ValueError("No tactile data found in batch")
        
        # For now, use the first tactile sensor
        # In the future, this could be extended to handle multiple sensors
        tactile_data = batch[tactile_keys[0]]
        
        # Ensure proper shape (B, C, H, W)
        if tactile_data.dim() == 3:
            tactile_data = tactile_data.unsqueeze(1)  # Add channel dimension
        
        return tactile_data

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Forward pass through ACT-T3 policy.
        
        Args:
            batch: Input batch with images, tactile data, and state
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Normalize inputs
        batch = self.normalize_inputs(batch)
        
        # Handle image features
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Normalize targets
        batch = self.normalize_targets(batch)
        
        # Forward through model
        actions_hat, loss_dict = self.model(batch)

        # Compute losses
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict["l1_loss"] = l1_loss.item()
        
        # Add additional losses from fusion
        if "fusion_loss" in loss_dict:
            total_loss = l1_loss + loss_dict["fusion_loss"] * self.config.loss_weights["fusion_loss"]
        else:
            total_loss = l1_loss

        return total_loss, loss_dict

    def predict_action(self, observation: Dict[str, Tensor], task: Optional[str] = None) -> Tensor:
        """
        Predict action from observation.
        
        Args:
            observation: Input observation
            task: Optional task identifier
            
        Returns:
            Predicted actions
        """
        # Prepare batch
        batch = {**observation}
        if task is not None:
            batch["task"] = task
            
        # Add dummy action for VAE if needed
        if self.config.act_config.use_vae:
            batch["action"] = torch.zeros(1, self.config.act_config.chunk_size, 
                                        self.config.act_config.action_feature.shape[0])
            batch["action_is_pad"] = torch.zeros(1, self.config.act_config.chunk_size, dtype=torch.bool)
        
        # Forward pass
        with torch.no_grad():
            actions, _ = self.model(batch)
            
        # Apply temporal ensembling if enabled
        if hasattr(self.act_policy, 'temporal_ensembler'):
            actions = self.act_policy.temporal_ensembler(actions)
            
        return actions


class ACTT3Model(torch.nn.Module):
    """Main ACT-T3 model implementation."""
    
    def __init__(self, config: ACTT3Config):
        super().__init__()
        self.config = config
        
        # Create ACT policy
        act_config = config.get_act_config()
        self.act_policy = ACTPolicy(act_config)
        
        # T3 feature extractor
        self.t3_extractor = T3FeatureExtractor(config.t3_config)
        self.t3_extractor.set_projection(config.fusion_config.fusion_dim)
        
        # Feature manager for ACT
        self.feature_manager = ACTT3FeatureManager(
            self.act_policy, 
            config.fusion_config.fusion_dim
        )
        
        # Fusion module - will be updated with correct dimensions after feature extraction
        # For now, use placeholder dimensions that will be updated
        act_dim = config.act_config.dim_model  # ACT model dimension
        t3_dim = config.t3_config.tactile_embed_dim  # T3 embedding dimension
        self.fusion_module = ACTT3FusionModule(config.fusion_config, act_dim, t3_dim)
        
        # Action head (reuse from ACT)
        self.action_head = self.act_policy.model.action_head
        
        # Setup feature dimensions
        self._setup_feature_dimensions()
        
    def _setup_feature_dimensions(self):
        """Setup feature dimensions for projection."""
        # Create dummy batch to get feature dimensions
        dummy_batch = self._create_dummy_batch()
        
        # Extract ACT features to get dimensions
        act_features = self.feature_manager.extract_and_project_features(dummy_batch, 'encoder_output')
        
        # Get feature dimensions
        feature_dims = {
            'encoder_output': act_features.size(-1)
        }
        
        # Setup projectors
        self.feature_manager.setup_projectors(feature_dims)
        
    def _create_dummy_batch(self) -> Dict[str, Tensor]:
        """Create dummy batch for feature dimension setup."""
        batch_size = 1
        chunk_size = self.config.act_config.chunk_size
        
        dummy_batch = {}
        
        # Add image data if present
        image_list = []
        for key, feature in self.config.input_features.items():
            if "images" in key:
                img_shape = feature.shape
                img_tensor = torch.randn(batch_size, *img_shape)
                dummy_batch[key] = img_tensor
                image_list.append(img_tensor)
        
        # Add observation.images field (required by ACT)
        if image_list:
            dummy_batch["observation.images"] = image_list
        
        # Add state data
        for key, feature in self.config.input_features.items():
            if "state" in key and "images" not in key:
                state_shape = feature.shape
                dummy_batch[key] = torch.randn(batch_size, *state_shape)
        
        # Add environment state (required by ACT)
        if "observation.state" in dummy_batch:
            dummy_batch["observation.environment_state"] = dummy_batch["observation.state"]
        else:
            # Create default environment state
            dummy_batch["observation.environment_state"] = torch.randn(batch_size, 14)
        
        # Add action data for VAE
        if self.config.act_config.use_vae:
            action_shape = self.config.act_config.action_feature.shape
            dummy_batch["action"] = torch.randn(batch_size, chunk_size, *action_shape)
            dummy_batch["action_is_pad"] = torch.zeros(batch_size, chunk_size, dtype=torch.bool)
        
        return dummy_batch
    
    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Forward pass through ACT-T3 model.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (actions, loss_dict)
        """
        # Prepare ACT batch (remove tactile data)
        act_batch = self._prepare_act_batch(batch)
        
        # Extract ACT features
        act_features = self.feature_manager.get_act_features_for_fusion(act_batch)
        
        # Extract tactile data and features
        tactile_data = self._extract_tactile_data(batch)
        t3_features = self.t3_extractor.extract_features(tactile_data)
        
        # Fuse features
        fused_features = self.fusion_module(act_features, t3_features)
        
        # Generate actions using ACT decoder
        # We need to replace the encoder output in ACT with our fused features
        actions = self._generate_actions_with_fused_features(act_batch, fused_features)
        
        # Compute additional losses
        loss_dict = self._compute_additional_losses(act_features, t3_features, fused_features)
        
        return actions, loss_dict
    
    def _prepare_act_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Prepare batch for ACT (remove tactile data)."""
        act_batch = {}
        for key, value in batch.items():
            if 'tactile' not in key.lower():
                act_batch[key] = value
        return act_batch
    
    def _extract_tactile_data(self, batch: Dict[str, Tensor]) -> Tensor:
        """Extract tactile data from batch."""
        tactile_keys = [k for k in batch.keys() if 'tactile' in k.lower()]
        if not tactile_keys:
            raise ValueError("No tactile data found in batch")
        
        tactile_data = batch[tactile_keys[0]]
        
        # Ensure proper shape (B, C, H, W)
        if tactile_data.dim() == 3:
            tactile_data = tactile_data.unsqueeze(1)
        
        return tactile_data
    
    def _generate_actions_with_fused_features(self, act_batch: Dict[str, Tensor], 
                                            fused_features: Tensor) -> Tensor:
        """
        Generate actions using ACT decoder with fused features.
        
        Args:
            act_batch: Batch for ACT
            fused_features: Fused features from ACT and T3 (B, L, D)
            
        Returns:
            Generated actions
        """
        # Get batch size
        if "observation.images.camera" in act_batch:
            batch_size = act_batch["observation.images.camera"].shape[0]
        else:
            batch_size = act_batch["observation.state"].shape[0]
        
        # Transpose fused features to ACT decoder expected format: (L, B, D)
        fused_features = fused_features.transpose(0, 1)  # (L, B, D)
        
        # Create decoder input
        decoder_in = torch.zeros(
            (self.config.act_config.chunk_size, batch_size, self.config.act_config.dim_model),
            dtype=fused_features.dtype,
            device=fused_features.device,
        )
        
        # Use ACT decoder with fused features
        decoder_out = self.act_policy.model.decoder(
            decoder_in,
            fused_features,  # Use fused features instead of ACT encoder output
            encoder_pos_embed=None,  # No positional embedding for fused features
            decoder_pos_embed=self.act_policy.model.decoder_pos_embed.weight.unsqueeze(1),
        )
        
        # Generate actions
        decoder_out = decoder_out.transpose(0, 1)  # (B, S, C)
        actions = self.action_head(decoder_out)
        
        return actions
    
    def _compute_additional_losses(self, act_features: Tensor, t3_features: Tensor, 
                                 fused_features: Tensor) -> Dict[str, Any]:
        """
        Compute additional losses for training.
        
        Args:
            act_features: ACT features (B, L, D)
            t3_features: T3 features (B, L, D)
            fused_features: Fused features (B, L, D)
            
        Returns:
            Dictionary of additional losses
        """
        loss_dict = {}
        
        # Ensure all features have the same shape for loss computation
        # act_features: (B, L, D) -> (B, L, D)
        # fused_features: (L, B, D) -> (B, L, D) (if transposed)
        if fused_features.size(0) != act_features.size(0):
            # fused_features is in (L, B, D) format, transpose to (B, L, D)
            fused_features = fused_features.transpose(0, 1)
        
        # Consistency loss between ACT and fused features
        if self.config.loss_weights.get("consistency_loss", 0.0) > 0:
            # Ensure both tensors have the same shape
            if act_features.shape != fused_features.shape:
                # Pool the longer sequence to match the shorter one
                if act_features.size(1) > fused_features.size(1):
                    act_features = F.adaptive_avg_pool1d(
                        act_features.transpose(1, 2), 
                        fused_features.size(1)
                    ).transpose(1, 2)
                elif fused_features.size(1) > act_features.size(1):
                    fused_features = F.adaptive_avg_pool1d(
                        fused_features.transpose(1, 2), 
                        act_features.size(1)
                    ).transpose(1, 2)
            
            consistency_loss = F.mse_loss(act_features, fused_features)
            loss_dict["consistency_loss"] = consistency_loss
        
        # Feature diversity loss
        if self.config.loss_weights.get("diversity_loss", 0.0) > 0:
            # Ensure both tensors have the same shape
            if act_features.size(1) != t3_features.size(1):
                # Pool the longer sequence to match the shorter one
                if act_features.size(1) > t3_features.size(1):
                    act_features = F.adaptive_avg_pool1d(
                        act_features.transpose(1, 2), 
                        t3_features.size(1)
                    ).transpose(1, 2)
                elif t3_features.size(1) > act_features.size(1):
                    t3_features = F.adaptive_avg_pool1d(
                        t3_features.transpose(1, 2), 
                        act_features.size(1)
                    ).transpose(1, 2)
            
            # Encourage different modalities to learn different features
            act_mean = act_features.mean(dim=1)  # (B, D)
            t3_mean = t3_features.mean(dim=1)    # (B, D)
            diversity_loss = -F.cosine_similarity(act_mean, t3_mean, dim=1).mean()
            loss_dict["diversity_loss"] = diversity_loss
        
        return loss_dict 