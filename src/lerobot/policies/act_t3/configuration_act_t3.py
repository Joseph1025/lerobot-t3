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

from dataclasses import dataclass, field
from typing import Literal, Dict, Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass
class T3Config:
    """Configuration for T3 tactile encoder component."""
    
    # T3 tactile encoder configuration
    tactile_patch_size: int = 16
    tactile_embed_dim: int = 768
    tactile_depth: int = 3
    tactile_heads: int = 12
    tactile_mlp_ratio: float = 4.0
    tactile_dropout: float = 0.1
    
    # Tactile input configuration
    tactile_input_channels: int = 1  # Default for pressure sensors
    tactile_input_size: int = 320    # Default tactile sensor width
    tactile_input_height: int = 240  # Default tactile sensor height
    
    def __post_init__(self):
        """Validate T3 configuration."""
        if self.tactile_embed_dim % self.tactile_heads != 0:
            raise ValueError(f"tactile_embed_dim ({self.tactile_embed_dim}) must be divisible by tactile_heads ({self.tactile_heads})")


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion module."""
    
    # Fusion type - now supports simple concatenation
    fusion_type: Literal["concat", "weighted_concat"] = "concat"
    
    # Modality weights (for weighted concatenation)
    act_weight: float = 0.6
    t3_weight: float = 0.4
    
    # Feature dimensions
    fusion_dim: int = 512  # Output dimension after fusion
    
    # Dropout for regularization
    fusion_dropout: float = 0.1
    
    def __post_init__(self):
        """Validate fusion configuration."""
        if self.fusion_type not in ["concat", "weighted_concat"]:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        if self.fusion_type == "weighted_concat":
            if not 0.0 <= self.act_weight <= 1.0:
                raise ValueError(f"act_weight must be between 0 and 1, got {self.act_weight}")
            if not 0.0 <= self.t3_weight <= 1.0:
                raise ValueError(f"t3_weight must be between 0 and 1, got {self.t3_weight}")
            if abs(self.act_weight + self.t3_weight - 1.0) > 1e-6:
                raise ValueError(f"act_weight + t3_weight must equal 1.0, got {self.act_weight + self.t3_weight}")


@PreTrainedConfig.register_subclass("act_t3")
@dataclass
class ACTT3Config(PreTrainedConfig):
    """Configuration for ACT-T3 policy that combines ACT and T3 without modifying original ACT."""
    
    # ACT configuration (reused, unmodified)
    act_config: ACTConfig = field(default_factory=ACTConfig)
    
    # T3 configuration (new)
    t3_config: T3Config = field(default_factory=T3Config)
    
    # Fusion configuration (new)
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    
    # Loss weights for different components
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "act_loss": 1.0,
        "t3_loss": 1.0,
        "fusion_loss": 0.5,
        "consistency_loss": 0.1,
    })
    
    # Training configuration
    freeze_act: bool = False  # Whether to freeze ACT parameters during training
    freeze_t3: bool = False   # Whether to freeze T3 parameters during training
    
    # Input/output configuration
    input_shapes: Dict[str, list] = field(default_factory=lambda: {
        "observation.images.cam_high": [3, 224, 224],
        "observation.images.cam_left_wrist": [3, 224, 224],
        "observation.images.cam_right_wrist": [3, 224, 224],
        "observation.state": [14],
        "observation.effort": [14],
        "observation.qvel": [14],
        "observation.tactile1": [1, 64, 64],
        "observation.tactile2": [1, 64, 64],
    })
    
    output_shapes: Dict[str, list] = field(default_factory=lambda: {
        "action": [14],
    })
    
    # Normalization configuration
    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    
    def __post_init__(self):
        """Validate ACT-T3 configuration."""
        super().__post_init__()
        
        # Convert input_shapes and output_shapes to input_features and output_features
        from lerobot.configs.types import PolicyFeature, FeatureType
        
        # Convert input shapes to input features
        for key, shape in self.input_shapes.items():
            # Handle case where shape might already be a PolicyFeature
            if hasattr(shape, 'shape'):
                # shape is already a PolicyFeature, use it directly
                self.input_features[key] = shape
            else:
                # shape is a tuple, create PolicyFeature
                if "images" in key:
                    self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
                elif "tactile" in key:
                    self.input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
                elif "state" in key or "effort" in key or "qvel" in key:
                    self.input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
                else:
                    self.input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        
        # Convert output shapes to output features
        for key, shape in self.output_shapes.items():
            # Handle case where shape might already be a PolicyFeature
            if hasattr(shape, 'shape'):
                # shape is already a PolicyFeature, use it directly
                self.output_features[key] = shape
            else:
                # shape is a tuple, create PolicyFeature
                if "action" in key:
                    self.output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=shape)
                else:
                    self.output_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        
        # Validate input shapes - check for at least one image and one tactile sensor
        has_image = any("images" in key for key in self.input_shapes.keys())
        has_tactile = any("tactile" in key for key in self.input_shapes.keys())
        has_state = any("state" in key for key in self.input_shapes.keys())
        
        if not has_image:
            raise ValueError("At least one image input is required (e.g., observation.images.cam_high)")
        if not has_tactile:
            raise ValueError("At least one tactile input is required (e.g., observation.tactile1)")
        if not has_state:
            raise ValueError("At least one state input is required (e.g., observation.state)")
        
        # Validate output shapes
        if "action" not in self.output_shapes:
            raise ValueError("Required output 'action' not found in output_shapes")
        
        # Validate loss weights
        required_losses = ["act_loss", "t3_loss", "fusion_loss"]
        for required_loss in required_losses:
            if required_loss not in self.loss_weights:
                raise ValueError(f"Required loss weight '{required_loss}' not found in loss_weights")
        
        # Ensure ACT config is compatible - ACTConfig uses input_features/output_features, not input_shapes/output_shapes
        # We need to convert our input_shapes to input_features format for ACT
        act_input_features = {}
        for key, shape in self.input_shapes.items():
            if "tactile" not in key:  # Exclude tactile from ACT input
                # Convert shape to PolicyFeature format
                if hasattr(shape, 'shape'):
                    # shape is already a PolicyFeature, use it directly
                    act_input_features[key] = shape
                else:
                    # shape is a tuple, create PolicyFeature
                    from lerobot.configs.types import PolicyFeature, FeatureType
                    if "images" in key:
                        act_input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
                    elif "state" in key:
                        act_input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
                    else:
                        act_input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        
        act_output_features = {}
        for key, shape in self.output_shapes.items():
            if hasattr(shape, 'shape'):
                # shape is already a PolicyFeature, use it directly
                act_output_features[key] = shape
            else:
                # shape is a tuple, create PolicyFeature
                from lerobot.configs.types import PolicyFeature, FeatureType
                if "action" in key:
                    act_output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=shape)
                else:
                    act_output_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        
        # Update ACT config with our features
        self.act_config.input_features = act_input_features
        self.act_config.output_features = act_output_features
    
    def validate_features(self) -> None:
        """Validate that the input and output features are properly configured."""
        # Validate ACT features
        self.act_config.validate_features()
        
        # Validate tactile features - check for at least one tactile sensor
        tactile_keys = [k for k in self.input_features.keys() if "tactile" in k]
        if not tactile_keys:
            raise ValueError("At least one tactile input is required for ACT-T3 policy")
        
        # Validate each tactile input
        for tactile_key in tactile_keys:
            tactile_feature = self.input_features[tactile_key]
            tactile_shape = tactile_feature.shape
            if len(tactile_shape) != 3:
                raise ValueError(f"Tactile input must be 3D (C, H, W), got {tactile_shape}")
            
            # Validate tactile input size compatibility with patch size
            h, w = tactile_shape[1], tactile_shape[2]
            if h % self.t3_config.tactile_patch_size != 0 or w % self.t3_config.tactile_patch_size != 0:
                raise ValueError(
                    f"Tactile input size ({h}x{w}) must be divisible by patch size ({self.t3_config.tactile_patch_size})"
                )
    
    def get_act_config(self) -> ACTConfig:
        """Get the ACT configuration with updated input shapes."""
        # Create a copy of ACT config with our input shapes (excluding tactile)
        act_input_features = {}
        for key, shape in self.input_shapes.items():
            if "tactile" not in key:  # Exclude tactile from ACT input
                # Convert shape to PolicyFeature format
                if hasattr(shape, 'shape'):
                    # shape is already a PolicyFeature, use it directly
                    act_input_features[key] = shape
                else:
                    # shape is a tuple, create PolicyFeature
                    from lerobot.configs.types import PolicyFeature, FeatureType
                    if "images" in key:
                        act_input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
                    elif "state" in key:
                        act_input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
                    else:
                        act_input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        
        act_output_features = {}
        for key, shape in self.output_shapes.items():
            if hasattr(shape, 'shape'):
                # shape is already a PolicyFeature, use it directly
                act_output_features[key] = shape
            else:
                # shape is a tuple, create PolicyFeature
                from lerobot.configs.types import PolicyFeature, FeatureType
                if "action" in key:
                    act_output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=shape)
                else:
                    act_output_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        
        act_config = ACTConfig(
            input_features=act_input_features,
            output_features=act_output_features,
            normalization_mapping=self.normalization_mapping,
            # Copy other ACT parameters
            n_obs_steps=self.act_config.n_obs_steps,
            chunk_size=self.act_config.chunk_size,
            n_action_steps=self.act_config.n_action_steps,
            vision_backbone=self.act_config.vision_backbone,
            pretrained_backbone_weights=self.act_config.pretrained_backbone_weights,
            use_siglip=self.act_config.use_siglip,
            siglip_model_name=self.act_config.siglip_model_name,
            dim_model=self.act_config.dim_model,
            n_heads=self.act_config.n_heads,
            dim_feedforward=self.act_config.dim_feedforward,
            n_encoder_layers=self.act_config.n_encoder_layers,
            n_decoder_layers=self.act_config.n_decoder_layers,
            use_vae=self.act_config.use_vae,
            latent_dim=self.act_config.latent_dim,
            dropout=self.act_config.dropout,
            kl_weight=self.act_config.kl_weight,
        )
        
        return act_config
    
    @property
    def observation_delta_indices(self) -> list | None:
        """Get observation delta indices for ACT-T3 policy."""
        # For ACT-T3, we use single timestep observations
        return [0.0]
    
    @property
    def action_delta_indices(self) -> list | None:
        """Get action delta indices for ACT-T3 policy."""
        # For ACT-T3, we use action sequences
        fps = 30  # Default fps, can be overridden
        chunk_size = self.act_config.chunk_size
        return [i / fps for i in range(chunk_size)]
    
    @property
    def reward_delta_indices(self) -> list | None:
        """Get reward delta indices for ACT-T3 policy."""
        # ACT-T3 doesn't use rewards directly
        return None
    
    def get_optimizer_preset(self) -> "OptimizerConfig":
        """Get optimizer preset for ACT-T3 policy."""
        from lerobot.optim.optimizers import OptimizerConfig
        
        return OptimizerConfig(
            type="adam",
            lr=1e-4,
            weight_decay=1e-4,
        )
    
    def get_scheduler_preset(self) -> "LRSchedulerConfig | None":
        """Get scheduler preset for ACT-T3 policy."""
        from lerobot.optim.schedulers import LRSchedulerConfig
        
        return LRSchedulerConfig(
            type="cosine",
            warmup_steps=1000,
            max_steps=50000,
        ) 