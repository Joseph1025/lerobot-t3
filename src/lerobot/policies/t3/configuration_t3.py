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
from typing import Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode


@PreTrainedConfig.register_subclass("t3")
@dataclass
class T3Config(PreTrainedConfig):
    """Configuration class for the T3 (Transferable Tactile Transformer) policy.
    
    This extends ACT to support multimodal fusion of camera and tactile data.
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture
    # Vision backbone (ACT branch)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    use_siglip: bool = False
    siglip_model_name: str = "google/siglip-base-patch16-224"
    
    # T3 tactile encoder configuration
    tactile_patch_size: int = 16
    tactile_embed_dim: int = 768
    tactile_depth: int = 3
    tactile_heads: int = 12
    tactile_mlp_ratio: float = 4.0
    
    # Transformer fusion configuration
    fusion_type: Literal["cross_attention", "concat", "weighted_sum"] = "cross_attention"
    fusion_layers: int = 2
    fusion_heads: int = 8
    
    # Transformer layers
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    
    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation
    dropout: float = 0.1
    kl_weight: float = 10.0
    tactile_loss_weight: float = 1.0
    fusion_loss_weight: float = 1.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_lr_tactile: float = 1e-5

    def validate_features(self) -> None:
        """Validate that the input and output features are properly configured."""
        super().validate_features()
        
        # Check for tactile features
        tactile_features = [k for k in self.input_features.keys() if "tactile" in k.lower()]
        if not tactile_features:
            raise ValueError("T3 policy requires tactile input features. Add keys like 'observation.tactile' to input_shapes.")
        
        # Check for image features (ACT branch)
        image_features = [k for k in self.input_features.keys() if "image" in k.lower()]
        if not image_features:
            raise ValueError("T3 policy requires image input features for the ACT branch. Add keys like 'observation.images' to input_shapes.") 