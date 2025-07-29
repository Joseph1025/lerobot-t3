#!/usr/bin/env python
"""
ACT-T3 Policy Training Script for LeRobot dataset.

This script trains the ACT-T3 policy that combines ACT and T3 for multimodal
learning with camera and tactile data.

Usage:
    python train_act_t3.py --dataset_path /path/to/converted/dataset --output_dir /path/to/output
    python train_act_t3.py --dataset_path /path/to/converted/dataset --output_dir /path/to/output --pretrained_model_path /path/to/existing/model
"""

import argparse
from pathlib import Path
import torch
import json
import logging
from typing import Dict, Any, Optional
import time
import numpy as np

# Add src directory to path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act_t3.configuration_act_t3 import ACTT3Config, T3Config, FusionConfig
from lerobot.policies.act_t3.modeling_act_t3 import ACTT3Policy

# Import utility functions
try:
    from .utils import (
        get_delta_timestamps,
        save_checkpoint,
        prepare_act_batch,
        save_training_info,
        validate_training_parameters,
        get_dataset_features
    )
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from train.utils import (
        get_delta_timestamps,
        save_checkpoint,
        prepare_act_batch,
        save_training_info,
        validate_training_parameters,
        get_dataset_features
    )

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_act_t3_config(input_features: dict, output_features: dict, 
                     chunk_size: int = 16, n_action_steps: int = 16,
                     use_siglip: bool = False, siglip_model_name: str = "google/siglip-base-patch16-224",
                     tactile_patch_size: int = 16, tactile_embed_dim: int = 768,
                     fusion_type: str = "concat", fusion_dim: int = 512) -> ACTT3Config:
    """
    Get ACT-T3 configuration.
    
    Args:
        input_features: Input features dictionary
        output_features: Output features dictionary
        chunk_size: Size of action chunks for ACT
        n_action_steps: Number of action steps to execute for ACT
        use_siglip: Whether to use SigLIP vision backbone
        siglip_model_name: SigLIP model name
        tactile_patch_size: Patch size for tactile encoder
        tactile_embed_dim: Embedding dimension for tactile encoder
        fusion_type: Type of fusion ("concat" or "weighted_concat")
        fusion_dim: Dimension for fusion output
        
    Returns:
        ACTT3Config object
    """
    # Determine tactile input configuration from actual dataset features
    tactile_keys = [k for k in input_features.keys() if "tactile" in k]
    if not tactile_keys:
        raise ValueError("No tactile features found in input_features")
    
    # Use the first tactile sensor to determine configuration
    first_tactile_key = tactile_keys[0]
    tactile_shape = input_features[first_tactile_key].shape
    
    # Extract tactile configuration from shape (C, W, H) - dataset format
    if len(tactile_shape) == 3:
        tactile_channels, tactile_width, tactile_height = tactile_shape
        # Note: dataset provides (C, W, H) but T3 expects (C, H, W)
        # So we need to swap width and height for T3 configuration
    else:
        raise ValueError(f"Expected tactile shape to be 3D (C, W, H), got {tactile_shape}")
    
    # Create T3 configuration based on actual data
    # Note: T3 expects (C, H, W) so we use height=tactile_height, width=tactile_width
    t3_config = T3Config(
        tactile_patch_size=tactile_patch_size,
        tactile_embed_dim=tactile_embed_dim,
        tactile_depth=3,
        tactile_heads=12,
        tactile_mlp_ratio=4.0,
        tactile_dropout=0.1,
        tactile_input_channels=tactile_channels,  # Use actual number of channels
        tactile_input_size=tactile_width,         # Use actual width (will become height after transpose)
        tactile_input_height=tactile_height,      # Use actual height (will become width after transpose)
    )
    
    # Create ACT configuration
    from lerobot.policies.act.configuration_act import ACTConfig
    act_config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        use_siglip=use_siglip,
        siglip_model_name=siglip_model_name,
    )
    
    # Create fusion configuration
    fusion_config = FusionConfig(
        fusion_type=fusion_type,
        fusion_dim=fusion_dim,
        fusion_dropout=0.1,
        act_weight=0.6,
        t3_weight=0.4,
    )
    
    # Create ACT-T3 configuration with actual input shapes from dataset
    config = ACTT3Config(
        act_config=act_config,
        t3_config=t3_config,
        fusion_config=fusion_config,
        input_shapes=input_features,
        output_shapes=output_features,
        loss_weights={
            "act_loss": 1.0,
            "t3_loss": 1.0,
            "fusion_loss": 0.5,
            "consistency_loss": 0.1,
        },
        freeze_act=False,
        freeze_t3=False,
    )
    
    return config


def get_act_t3_model(config: ACTT3Config, dataset_stats: Dict[str, Dict[str, torch.Tensor]], 
                    pretrained_model_path: Optional[Path] = None) -> ACTT3Policy:
    """
    Get ACT-T3 model.
    
    Args:
        config: ACT-T3 configuration
        dataset_stats: Dataset statistics for normalization
        pretrained_model_path: Path to pretrained model to load (optional)
        
    Returns:
        ACTT3Policy object
    """
    if pretrained_model_path is not None:
        policy = ACTT3Policy.from_pretrained(pretrained_model_path, dataset_stats=dataset_stats)
    else:
        policy = ACTT3Policy(config, dataset_stats=dataset_stats)
    
    return policy


def prepare_act_t3_batch(batch: Dict[str, Any], policy: ACTT3Policy, device_torch: torch.device) -> Dict[str, Any]:
    """
    Prepare batch for ACT-T3 training.
    
    Args:
        batch: Input batch
        policy: ACT-T3 policy
        device_torch: Target device
        
    Returns:
        Prepared batch
    """
    # First prepare batch for ACT (handles image features, state, action)
    batch = prepare_act_batch(batch, policy.act_policy, device_torch)
    
    # Add tactile data if present
    tactile_keys = [k for k in batch.keys() if 'tactile' in k.lower()]
    if not tactile_keys:
        logger.warning("No tactile data found in batch. Using zero tensor as placeholder.")
        # Create zero tensor for tactile data
        batch_size = next(iter(batch.values())).shape[0] if batch else 1
        tactile_shape = policy.config.input_shapes.get("observation.tactile.sensor", [1, 64, 64])
        batch["observation.tactile.sensor"] = torch.zeros(batch_size, *tactile_shape, device=device_torch)
    
    return batch


def save_simplified_dataset_stats(dataset_stats: Dict[str, Dict[str, torch.Tensor]], 
                                output_dir: Path) -> None:
    """
    Save simplified dataset statistics for inference use.
    
    Args:
        dataset_stats: Full dataset statistics
        output_dir: Output directory
    """
    simplified_stats = {}
    
    # Keep only the essential statistics for inference
    for feature_name, stats in dataset_stats.items():
        simplified_stats[feature_name] = {
            'mean': stats.get('mean', torch.zeros(1)).tolist(),
            'std': stats.get('std', torch.ones(1)).tolist(),
            'min': stats.get('min', torch.zeros(1)).tolist(),
            'max': stats.get('max', torch.ones(1)).tolist(),
        }
    
    # Save simplified stats
    stats_path = output_dir / "simplified_dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(simplified_stats, f, indent=2)
    
    logger.info(f"Saved simplified dataset stats to: {stats_path}")


def get_act_t3_delta_timestamps(dataset_metadata: LeRobotDatasetMetadata, chunk_size: int = 16) -> Dict[str, list]:
    """
    Get delta timestamps for ACT-T3 policy.
    
    Args:
        dataset_metadata: Dataset metadata
        chunk_size: Number of actions in the sequence
        
    Returns:
        Delta timestamps dictionary
    """
    fps = dataset_metadata.fps
    
    return {
        "observation.images.cam_high": [0.0],
        "observation.images.cam_left_wrist": [0.0],
        "observation.images.cam_right_wrist": [0.0],
        "observation.state": [0.0],
        "observation.effort": [0.0],
        "observation.qvel": [0.0],
        "observation.tactile1": [0.0],  # Use actual tactile sensor names
        "observation.tactile2": [0.0],  # Use actual tactile sensor names
        "action": [i / fps for i in range(chunk_size)],
    }


def train_act_t3(
    dataset_path: Path,
    output_dir: Path,
    training_steps: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    log_freq: int = 100,
    device: str = "cuda",
    chunk_size: int = 16,
    n_action_steps: int = 16,
    video_backend: str = "torchcodec",
    pretrained_model_path: Optional[Path] = None,
    use_siglip: bool = False,
    siglip_model_name: str = "google/siglip-base-patch16-224",
    tactile_patch_size: int = 16,
    tactile_embed_dim: int = 768,
    fusion_type: str = "concat",
    fusion_dim: int = 512,
    freeze_act: bool = False,
    freeze_t3: bool = False,
):
    """
    Train an ACT-T3 policy on the converted dataset.
    
    Args:
        dataset_path: Path to the converted LeRobot dataset
        output_dir: Directory to save the trained policy
        training_steps: Number of training steps
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        log_freq: Frequency of logging
        device: Device to train on ("cuda" or "cpu")
        chunk_size: Size of action chunks for ACT
        n_action_steps: Number of action steps to execute for ACT
        video_backend: Video backend to use ("opencv", "torchcodec", "ffmpeg")
        pretrained_model_path: Path to pretrained model to continue training (optional)
        use_siglip: Whether to use SigLIP vision backbone
        siglip_model_name: SigLIP model name
        tactile_patch_size: Patch size for tactile encoder
        tactile_embed_dim: Embedding dimension for tactile encoder
        fusion_type: Type of fusion ("concat" or "weighted_concat")
        fusion_dim: Dimension for fusion output
        freeze_act: Whether to freeze ACT parameters
        freeze_t3: Whether to freeze T3 parameters
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Print training configuration
    print("=" * 80)
    print("ACT-T3 POLICY TRAINING")
    print("=" * 80)
    print(f"Device: {device_torch}")
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Training steps: {training_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Log frequency: {log_freq}")
    print()
    
    # Get dataset features using utility function
    input_features, output_features, dataset_metadata = get_dataset_features(dataset_path)
    
    # Print dataset information
    print("DATASET INFORMATION")
    print("-" * 40)
    print(f"Total episodes: {dataset_metadata.total_episodes}")
    print(f"Total frames: {dataset_metadata.total_frames}")
    print(f"FPS: {dataset_metadata.fps}")
    print(f"Robot type: {dataset_metadata.robot_type}")
    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")
    print()
    
    # Get policy configuration and model
    if pretrained_model_path is not None:
        # Load existing model - configuration will be loaded from the model
        policy = get_act_t3_model(config=None, dataset_stats=dataset_metadata.stats, 
                                pretrained_model_path=pretrained_model_path)
        
        # Update configuration if needed
        if use_siglip:
            policy.config.act_config.use_siglip = use_siglip
            policy.config.act_config.siglip_model_name = siglip_model_name
        
        chunk_size = policy.config.act_config.chunk_size
        n_action_steps = policy.config.act_config.n_action_steps
        delta_timestamps = get_act_t3_delta_timestamps(dataset_metadata, chunk_size=chunk_size)
        print(f"Loaded pretrained model from: {pretrained_model_path}")
    else:
        # Create new model from scratch
        config = get_act_t3_config(
            input_features, output_features, chunk_size, n_action_steps,
            use_siglip, siglip_model_name, tactile_patch_size, tactile_embed_dim,
            fusion_type, fusion_dim
        )
        
        # Set freezing options
        config.freeze_act = freeze_act
        config.freeze_t3 = freeze_t3
        
        delta_timestamps = get_act_t3_delta_timestamps(dataset_metadata, chunk_size=chunk_size)
        policy = get_act_t3_model(config, dataset_metadata.stats)
    
    # Print model architecture
    print("MODEL ARCHITECTURE")
    print("-" * 40)
    print(f"Policy type: ACT-T3")
    print(f"ACT chunk size: {chunk_size}")
    print(f"ACT action steps: {n_action_steps}")
    print(f"T3 patch size: {tactile_patch_size}")
    print(f"T3 embed dimension: {tactile_embed_dim}")
    print(f"Fusion type: {fusion_type}")
    print(f"Fusion dimension: {fusion_dim}")
    print(f"Vision backbone: {'SigLIP' if use_siglip else 'ResNet'}")
    if use_siglip:
        print(f"SigLIP model: {siglip_model_name}")
    print(f"Freeze ACT: {freeze_act}")
    print(f"Freeze T3: {freeze_t3}")
    print()
    
    policy.train()
    policy.to(device_torch)
    
    # Load dataset with delta_timestamps and video_backend
    print("Loading dataset...")
    dataset = LeRobotDataset(str(dataset_path), delta_timestamps=delta_timestamps, video_backend=video_backend)
    print(f"Dataset loaded successfully!")
    print()
    
    # Create optimizer and dataloader
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=str(device_torch) != "cpu",
        drop_last=True,
    )
    
    # Training loop
    print("STARTING TRAINING")
    print("-" * 40)
    print(f"Total training steps: {training_steps}")
    print(f"Steps per epoch: {len(dataloader)}")
    print(f"Estimated epochs: {training_steps / len(dataloader):.1f}")
    print()
    
    step = 0
    done = False
    best_loss = float('inf')
    best_step = 0  # Track which step had the best loss
    
    # Training progress tracking
    start_time = time.time()
    step_times = []
    
    while not done:
        for batch in dataloader:
            step_start_time = time.time()
            
            # Move batch to device
            batch = {k: (v.to(device_torch) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # Prepare batch for ACT-T3
            batch = prepare_act_t3_batch(batch, policy, device_torch)
            
            # Forward pass
            loss, loss_dict = policy.forward(batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Check if this is the best loss so far
            current_loss = loss.item()
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
                best_step = step
            
            # Save checkpoints using utility function (saves every 1000 steps + best model)
            save_checkpoint(policy, optimizer, step, current_loss, output_dir, is_best=is_best)
            
            # Calculate step time
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            # Logging (prints loss every log_freq steps)
            if step % log_freq == 0:
                avg_step_time = np.mean(step_times[-log_freq:]) if len(step_times) >= log_freq else step_time
                elapsed_time = time.time() - start_time
                eta = (training_steps - step) * avg_step_time
                
                print(f"Step {step:4d}/{training_steps} | "
                      f"Loss: {current_loss:.6f} | "
                      f"Best: {best_loss:.6f} | "
                      f"Time: {avg_step_time:.3f}s/step | "
                      f"ETA: {eta/60:.1f}min")
                
                # Log additional losses if available
                if loss_dict:
                    for loss_name, loss_value in loss_dict.items():
                        if isinstance(loss_value, (int, float)):
                            print(f"  {loss_name}: {loss_value:.6f}")
                
                print()
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # Training completion
    total_time = time.time() - start_time
    print("TRAINING COMPLETED")
    print("-" * 40)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best loss: {best_loss:.6f} at step {best_step}")
    print(f"Average step time: {np.mean(step_times):.3f} seconds")
    print()
    
    # Save final trained policy
    print("Saving final model...")
    policy.save_pretrained(output_dir)
    
    # Save simplified dataset stats for inference
    save_simplified_dataset_stats(dataset_metadata.stats, output_dir)
    
    # Save final training info using utility function
    save_training_info(
        output_dir=output_dir,
        step=step,
        current_loss=current_loss,
        best_loss=best_loss,
        policy_type="act_t3",
        training_steps=training_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        best_step=best_step
    )
    
    print(f"Training completed successfully!")
    print(f"Model saved to: {output_dir}")
    print("=" * 80)


def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train ACT-T3 policy on LeRobot dataset")
    
    # Required arguments
    parser.add_argument("--dataset_path", type=Path, required=True,
                       help="Path to the converted LeRobot dataset")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Directory to save the trained policy")
    
    # Training parameters
    parser.add_argument("--training_steps", type=int, default=5000,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for optimizer")
    parser.add_argument("--log_freq", type=int, default=100,
                       help="Frequency of logging")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on ('cuda' or 'cpu')")
    
    # ACT parameters
    parser.add_argument("--chunk_size", type=int, default=16,
                       help="Size of action chunks for ACT")
    parser.add_argument("--n_action_steps", type=int, default=16,
                       help="Number of action steps to execute for ACT")
    
    # Vision backbone parameters
    parser.add_argument("--use_siglip", action="store_true",
                       help="Use SigLIP vision backbone")
    parser.add_argument("--siglip_model_name", type=str, 
                       default="google/siglip-base-patch16-224",
                       help="SigLIP model name")
    
    # T3 parameters
    parser.add_argument("--tactile_patch_size", type=int, default=16,
                       help="Patch size for tactile encoder")
    parser.add_argument("--tactile_embed_dim", type=int, default=768,
                       help="Embedding dimension for tactile encoder")
    
    # Fusion parameters
    parser.add_argument("--fusion_type", type=str, default="concat",
                       choices=["concat", "weighted_concat"],
                       help="Type of fusion to use")
    parser.add_argument("--fusion_dim", type=int, default=512,
                       help="Dimension for fusion output")
    
    # Training options
    parser.add_argument("--freeze_act", action="store_true",
                       help="Freeze ACT parameters during training")
    parser.add_argument("--freeze_t3", action="store_true",
                       help="Freeze T3 parameters during training")
    
    # Data loading parameters
    parser.add_argument("--video_backend", type=str, default="torchcodec",
                       choices=["opencv", "torchcodec", "ffmpeg"],
                       help="Video backend to use")
    
    # Model loading
    parser.add_argument("--pretrained_model_path", type=Path, default=None,
                       help="Path to pretrained model to continue training")
    
    args = parser.parse_args()
    
    # Validate parameters
    validate_training_parameters(
        policy_type="act_t3",
        pretrained_model_path=args.pretrained_model_path,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
    )
    
    # Start training
    train_act_t3(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_freq=args.log_freq,
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        video_backend=args.video_backend,
        pretrained_model_path=args.pretrained_model_path,
        use_siglip=args.use_siglip,
        siglip_model_name=args.siglip_model_name,
        tactile_patch_size=args.tactile_patch_size,
        tactile_embed_dim=args.tactile_embed_dim,
        fusion_type=args.fusion_type,
        fusion_dim=args.fusion_dim,
        freeze_act=args.freeze_act,
        freeze_t3=args.freeze_t3,
    )


if __name__ == "__main__":
    main() 