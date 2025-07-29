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

from .configuration_act_t3 import ACTT3Config, T3Config, FusionConfig
from .t3_encoder import T3FeatureExtractor, T3TransformerEncoder
from .fusion_module import ACTT3FusionModule, SimpleConcatenationFusion, WeightedConcatenationFusion
from .act_feature_extractor import ACTT3FeatureManager, ACTFeatureExtractor
from .modeling_act_t3 import ACTT3Policy, ACTT3Model

__all__ = [
    # Configuration
    "ACTT3Config",
    "T3Config", 
    "FusionConfig",
    
    # T3 Components
    "T3FeatureExtractor",
    "T3TransformerEncoder",
    
    # Fusion Components (Simplified)
    "ACTT3FusionModule",
    "SimpleConcatenationFusion",
    "WeightedConcatenationFusion",
    
    # ACT Integration
    "ACTT3FeatureManager",
    "ACTFeatureExtractor",
    
    # Main Policy
    "ACTT3Policy",
    "ACTT3Model",
] 