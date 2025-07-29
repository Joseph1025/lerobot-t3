#!/usr/bin/env python
"""
Unified Policy Training Script for LeRobot dataset.

This script can train both ACT and Diffusion policies on our converted HDF5 dataset.
Based on the examples from examples/3_train_policy.py but adapted for our dataset structure.

Usage:
    python train_policy.py --dataset_path /path/to/converted/dataset --output_dir /path/to/output --policy_type diffusion
    python train_policy.py --dataset_path /path/to/converted/dataset --output_dir /path/to/output --policy_type act
    python train_policy.py --dataset_path /path/to/converted/dataset --output_dir /path/to/output --policy_type act --pretrained_model_path /path/to/existing/model
"""

import argparse
from pathlib import Path
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
import logging

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


def get_policy_config(policy_type: str, input_features: dict, output_features: dict, chunk_size: int = 16, n_action_steps: int = 16):
    """
    Get policy configuration based on policy type.
    
    Args:
        policy_type: Type of policy ("diffusion" or "act")
        input_features: Input features dictionary
        output_features: Output features dictionary
        chunk_size: Size of action chunks for ACT policy
        n_action_steps: Number of action steps to execute for ACT policy
        
    Returns:
        Policy configuration object
    """
    if policy_type.lower() == "diffusion":
        logger.info("Creating Diffusion Policy configuration...")
        diffusion_config = DiffusionConfig(input_features=input_features, output_features=output_features)
        # Note: horizon must be compatible with U-Net downsampling (2^len(down_dims))
        # Default down_dims = (512, 1024, 2048) means 2^3 = 8x downsampling
        # So horizon must be divisible by 8
        diffusion_config.horizon = 16  # Keep default for now
        diffusion_config.n_action_steps = 8  # Keep default for now
        diffusion_config.n_obs_steps = 2  # Keep default for now
        return diffusion_config

    elif policy_type.lower() == "act":
        logger.info("Creating ACT Policy configuration...")
        config = ACTConfig(input_features=input_features, output_features=output_features)
        # Config chunk size here
        config.chunk_size = chunk_size
        config.n_action_steps = n_action_steps
        logger.info(f"ACT config: chunk_size={chunk_size}, n_action_steps={n_action_steps}")
        return config
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}. Supported types: 'diffusion', 'act'")


def get_policy_model(policy_type: str, config, dataset_stats, pretrained_model_path: Path | None = None):
    """
    Get policy model based on policy type.
    
    Args:
        policy_type: Type of policy ("diffusion" or "act")
        config: Policy configuration
        dataset_stats: Dataset statistics
        pretrained_model_path: Path to pretrained model to load (optional)
        
    Returns:
        Policy model
    """
    if pretrained_model_path is not None:
        logger.info(f"Loading pretrained {policy_type.upper()} model from: {pretrained_model_path}")
        if policy_type.lower() == "diffusion":
            policy = DiffusionPolicy.from_pretrained(pretrained_model_path)
        elif policy_type.lower() == "act":
            policy = ACTPolicy.from_pretrained(pretrained_model_path)
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
        
        # Set the model to training mode for continued training
        policy.train()
        logger.info(f"Successfully loaded pretrained {policy_type.upper()} model")
        return policy
    else:
        # Create new model from scratch
        if policy_type.lower() == "diffusion":
            logger.info("Initializing Diffusion Policy...")
            return DiffusionPolicy(config, dataset_stats=dataset_stats)
        elif policy_type.lower() == "act":
            logger.info("Initializing ACT Policy...")
            return ACTPolicy(config, dataset_stats=dataset_stats)
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")


def train_policy(
    dataset_path: Path,
    output_dir: Path,
    policy_type: str = "diffusion",
    training_steps: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    log_freq: int = 100,
    device: str = "cuda",
    chunk_size: int = 16,
    n_action_steps: int = 16,
    horizon: int = 16,
    diffusion_n_action_steps: int = 8,
    diffusion_n_obs_steps: int = 2,
    video_backend: str = "torchcodec",
    pretrained_model_path: Path | None = None,
    use_siglip: bool = False,
    siglip_model_name: str = "google/siglip-base-patch16-224"
):
    """
    Train a policy on the converted dataset.
    
    Args:
        dataset_path: Path to the converted LeRobot dataset
        output_dir: Directory to save the trained policy
        policy_type: Type of policy ("diffusion" or "act")
        training_steps: Number of training steps
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        log_freq: Frequency of logging
        device: Device to train on ("cuda" or "cpu")
        chunk_size: Size of action chunks for ACT policy
        n_action_steps: Number of action steps to execute for ACT policy
        horizon: Horizon size for diffusion policy (must be divisible by 8)
        diffusion_n_action_steps: Number of action steps to execute for diffusion policy
        diffusion_n_obs_steps: Number of observation steps for diffusion policy
        video_backend: Video backend to use ("opencv", "torchcodec", "ffmpeg")
        pretrained_model_path: Path to pretrained model to continue training (optional)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device_torch}")
    logger.info(f"Training {policy_type.upper()} policy...")
    logger.info(f"Using video backend: {video_backend}")
    
    # Get dataset features using utility function
    input_features, output_features, dataset_metadata = get_dataset_features(dataset_path)
    
    # Get policy configuration and model
    if pretrained_model_path is not None:
        # Load existing model - configuration will be loaded from the model
        policy = get_policy_model(policy_type, config=None, dataset_stats=dataset_metadata.stats, pretrained_model_path=pretrained_model_path)
        
        # Update SigLIP settings if provided
        if use_siglip:
            policy.config.use_siglip = use_siglip
            policy.config.siglip_model_name = siglip_model_name
        
        # Get delta_timestamps based on the loaded model's configuration
        if policy_type.lower() == "act":
            chunk_size = policy.config.chunk_size
            n_action_steps = policy.config.n_action_steps
            delta_timestamps = get_delta_timestamps(policy_type, dataset_metadata, chunk_size=chunk_size)
            logger.info(f"Using loaded ACT config: chunk_size={chunk_size}, n_action_steps={n_action_steps}")
        else:  # diffusion
            horizon = policy.config.horizon
            diffusion_n_action_steps = policy.config.n_action_steps
            diffusion_n_obs_steps = policy.config.n_obs_steps
            delta_timestamps = get_delta_timestamps(policy_type, horizon=horizon, n_obs_steps=diffusion_n_obs_steps)
            logger.info(f"Using loaded Diffusion config: horizon={horizon}, n_action_steps={diffusion_n_action_steps}, n_obs_steps={diffusion_n_obs_steps}")
    else:
        # Create new model from scratch
        if policy_type.lower() == "act":
            config = get_policy_config(policy_type, input_features, output_features, chunk_size, n_action_steps)
            # Set SigLIP configuration
            config.use_siglip = use_siglip
            config.siglip_model_name = siglip_model_name
            delta_timestamps = get_delta_timestamps(policy_type, dataset_metadata, chunk_size=chunk_size)
        else:  # diffusion
            config = get_policy_config(policy_type, input_features, output_features)
            # Override diffusion config with command line arguments
            config.horizon = horizon
            config.n_action_steps = diffusion_n_action_steps
            config.n_obs_steps = diffusion_n_obs_steps
            # Set SigLIP configuration
            config.use_siglip = use_siglip
            config.siglip_model_name = siglip_model_name
            delta_timestamps = get_delta_timestamps(policy_type, horizon=config.horizon, n_obs_steps=config.n_obs_steps)
        policy = get_policy_model(policy_type, config, dataset_metadata.stats)
    
    policy.train()
    policy.to(device_torch)
    
    # Load dataset with delta_timestamps and video_backend
    logger.info("Loading dataset with delta_timestamps...")
    dataset = LeRobotDataset(str(dataset_path), delta_timestamps=delta_timestamps, video_backend=video_backend)
    logger.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    
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
    logger.info(f"Starting training for {training_steps} steps...")
    step = 0
    done = False
    best_loss = float('inf')
    best_step = 0  # Track which step had the best loss
    
    while not done:
        for batch in dataloader:
            # Move batch to device
            batch = {k: (v.to(device_torch) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # For ACT policy, prepare batch using utility function
            if policy_type.lower() == "act":
                batch = prepare_act_batch(batch, policy, device_torch)
            
            # Forward pass
            loss, _ = policy.forward(batch)
            
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
            
            # Logging (prints loss every log_freq steps)
            if step % log_freq == 0:
                logger.info(f"Step: {step}/{training_steps}, Loss: {current_loss:.6f}, Best Loss: {best_loss:.6f}")
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # Save final trained policy
    logger.info(f"Saving final trained {policy_type} policy to: {output_dir}")
    policy.save_pretrained(output_dir)
    
    # Save final training info using utility function
    save_training_info(
        output_dir=output_dir,
        step=step,
        current_loss=current_loss,
        best_loss=best_loss,
        policy_type=policy_type,
        training_steps=training_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        best_step=best_step
    )
    
    logger.info(f"{policy_type.upper()} policy training completed successfully!")
    logger.info(f"Final loss: {current_loss:.6f}")
    logger.info(f"Best model saved at step {best_step} with loss: {best_loss:.6f}")
    logger.info(f"Best model location: {output_dir}/checkpoints/best_model")


def main():
    parser = argparse.ArgumentParser(description="Train policies on converted LeRobot dataset")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to converted LeRobot dataset")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for trained policy")
    parser.add_argument("--policy_type", type=str, default="diffusion", choices=["diffusion", "act"], 
                       help="Type of policy to train")
    parser.add_argument("--training_steps", type=int, default=5000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency (prints loss every N steps)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--chunk_size", type=int, default=128, help="Size of action chunks for ACT policy")
    parser.add_argument("--n_action_steps", type=int, default=64, help="Number of action steps to execute for ACT policy")
    parser.add_argument("--horizon", type=int, default=16, help="Horizon size for diffusion policy (must be divisible by 8)")
    parser.add_argument("--diffusion_n_action_steps", type=int, default=8, help="Number of action steps to execute for diffusion policy")
    parser.add_argument("--diffusion_n_obs_steps", type=int, default=2, help="Number of observation steps for diffusion policy")
    parser.add_argument("--video_backend", type=str, default="torchcodec", choices=["opencv", "torchcodec", "ffmpeg"],
                       help="Video backend to use for loading dataset")
    parser.add_argument("--pretrained_model_path", type=Path, default=None, 
                       help="Path to pretrained model to continue training (optional)")
    parser.add_argument("--use_siglip", type=bool, default=False, 
                       help="Whether to use SigLIP as vision backbone")
    parser.add_argument("--siglip_model_name", type=str, default="google/siglip-base-patch16-224", 
                       help="SigLIP model name from HuggingFace")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")
    
    # Validate training parameters using utility function
    validate_training_parameters(
        policy_type=args.policy_type,
        pretrained_model_path=args.pretrained_model_path,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        horizon=args.horizon,
        diffusion_n_action_steps=args.diffusion_n_action_steps,
        diffusion_n_obs_steps=args.diffusion_n_obs_steps
    )
    
    # Train policy
    train_policy(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        policy_type=args.policy_type,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_freq=args.log_freq,
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        horizon=args.horizon,
        diffusion_n_action_steps=args.diffusion_n_action_steps,
        diffusion_n_obs_steps=args.diffusion_n_obs_steps,
        video_backend=args.video_backend,
        pretrained_model_path=args.pretrained_model_path,
        use_siglip=args.use_siglip,
        siglip_model_name=args.siglip_model_name
    )


if __name__ == "__main__":
    main() 