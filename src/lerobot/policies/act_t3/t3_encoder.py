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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from lerobot.policies.act_t3.configuration_act_t3 import T3Config


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Create 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Grid size (assumes square grid)
        cls_token: Whether to include CLS token
        
    Returns:
        Positional embeddings
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Create 2D sinusoidal positional embeddings from grid."""
    assert embed_dim % 2 == 0
    
    # Use half of dimensions to encode grid_h and grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Create 1D sinusoidal positional embeddings from grid positions.
    
    Args:
        embed_dim: Embedding dimension
        pos: Position values
        
    Returns:
        Positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class T3PatchEmbed(nn.Module):
    """Patch embedding for tactile images."""
    def __init__(self, img_size=(240, 320), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        # Ensure img_size is a tuple (H, W)
        if isinstance(img_size, int):
            img_size = (240, 320)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model {self.img_size}."
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class T3TransformerBlock(nn.Module):
    """Transformer block for tactile processing."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (B, seq_len, embed_dim)
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class T3TransformerEncoder(nn.Module):
    """T3 transformer encoder for processing tactile sensor data."""
    
    def __init__(self, config: T3Config):
        super().__init__()
        # Ensure img_size is a tuple (H, W)
        img_size = getattr(config, 'tactile_img_size', None)
        if img_size is None:
            img_size = (getattr(config, 'tactile_input_height', 64), getattr(config, 'tactile_input_size', 64))
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = config.tactile_patch_size
        self.embed_dim = config.tactile_embed_dim
        self.n_patches = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)
        self.patch_embed = T3PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=config.tactile_input_channels,
            embed_dim=self.embed_dim,
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, config.tactile_embed_dim))
        self.pos_drop = nn.Dropout(p=config.tactile_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            T3TransformerBlock(
                embed_dim=config.tactile_embed_dim,
                num_heads=config.tactile_heads,
                mlp_ratio=config.tactile_mlp_ratio,
                dropout=config.tactile_dropout
            ) for _ in range(config.tactile_depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(config.tactile_embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Regenerate positional embedding with correct shape
        grid_h = self.img_size[0] // self.patch_size
        grid_w = self.img_size[1] // self.patch_size
        
        # Create 2D positional embedding for rectangular grid
        if grid_h == grid_w:
            # Square grid - use the existing function
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_h)
        else:
            # Rectangular grid - create custom embedding
            grid_h_pos = np.arange(grid_h, dtype=np.float32)
            grid_w_pos = np.arange(grid_w, dtype=np.float32)
            grid = np.meshgrid(grid_w_pos, grid_h_pos)
            grid = np.stack(grid, axis=0)
            grid = grid.reshape([2, 1, grid_h, grid_w])
            pos_embed = get_2d_sincos_pos_embed_from_grid(self.embed_dim, grid)
        
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)  # (1, n_patches, embed_dim)
        
        if self.pos_embed.shape != pos_embed.shape:
            self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        else:
            self.pos_embed.data.copy_(pos_embed)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through T3 transformer encoder.
        
        Args:
            x: Tactile input tensor of shape (B, C, H, W)
            
        Returns:
            Encoded tactile features of shape (B, n_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization
        x = self.norm(x)
        
        return x


class T3FeatureExtractor(nn.Module):
    """Feature extractor for tactile data using T3 encoder."""
    
    def __init__(self, config: T3Config):
        super().__init__()
        self.config = config
        self.t3_encoder = T3TransformerEncoder(config)
        
        # Optional projection layer to match ACT feature dimension
        self.projection = None
        
    def set_projection(self, target_dim: int):
        """Set projection layer to match target dimension."""
        if self.config.tactile_embed_dim != target_dim:
            self.projection = nn.Linear(self.config.tactile_embed_dim, target_dim)
    
    def extract_features(self, tactile_data: torch.Tensor) -> torch.Tensor:
        """
        Extract features from tactile data.
        
        Args:
            tactile_data: Tactile data tensor of shape (batch_size, channels, width, height)
            
        Returns:
            Extracted features tensor
        """
        # Handle different input formats
        if tactile_data.dim() == 4:
            batch_size, channels, width, height = tactile_data.shape
            
            # If data is in (batch, channels, width, height) format, transpose to (batch, channels, height, width)
            if width != self.config.tactile_input_height or height != self.config.tactile_input_size:
                # Transpose width and height dimensions
                tactile_data = tactile_data.transpose(2, 3)  # (batch, channels, height, width)
                _, _, height, width = tactile_data.shape
            
            # Verify the shape after potential transpose
            # Note: after transpose, we expect (batch, channels, height, width)
            # where height should match tactile_input_height and width should match tactile_input_size
            assert height == self.config.tactile_input_height, \
                f"Tactile data height {height} does not match expected {self.config.tactile_input_height}"
            assert width == self.config.tactile_input_size, \
                f"Tactile data width {width} does not match expected {self.config.tactile_input_size}"
        else:
            raise ValueError(f"Expected 4D tensor, got {tactile_data.dim()}D tensor")
        
        # Extract features using the transformer encoder
        features = self.t3_encoder(tactile_data)
        return features 