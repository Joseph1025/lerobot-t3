#!/usr/bin/env python
"""
Resume Training from Checkpoint

This script allows you to resume training from a saved checkpoint.

Usage:
    python resume_training.py --checkpoint_path /path/to/checkpoint.pt --output_dir /path/to/output --training_steps 2000
"""

import argparse
from pathlib import Path
import torch
import logging
try:
    from .train_policy import get_policy_model
    from .utils import (
        load_checkpoint,
        get_model_config_from_checkpoint,
        prepare_act_batch,
        save_checkpoint,
        save_training_info
    )
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from train.train_policy import get_policy_model
    from train.utils import (
        load_checkpoint,
        get_model_config_from_checkpoint,
        prepare_act_batch,
        save_checkpoint,
        save_training_info
    )
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resume_training(
    checkpoint_path: Path,
    dataset_path: Path,
    output_dir: Path,
    training_steps: int,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    log_freq: int = 100,
    device: str = "cuda",
    video_backend: str = "torchcodec"
):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        dataset_path: Path to the dataset
        output_dir: Output directory for continued training
        training_steps: Total number of training steps (including previous steps)
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        log_freq: Frequency of logging
        device: Device to train on
        video_backend: Video backend to use
    """
    # Handle checkpoint path - could be a .pt file or a model directory
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.pt':
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        policy_type = checkpoint['policy_type']
        initial_step = checkpoint['step']
        logger.info(f"Loading from checkpoint file: {checkpoint_path}")
    elif checkpoint_path.is_dir():
        # Load from model directory (best_model or saved model)
        policy_type = None  # Will be determined from the model
        initial_step = 0    # Start from step 0 when loading from model directory
        logger.info(f"Loading from model directory: {checkpoint_path}")
    else:
        raise ValueError(f"Checkpoint path must be a .pt file or a model directory. Got: {checkpoint_path}")
    
    logger.info(f"Resuming {policy_type} training from step {initial_step}")
    logger.info(f"Target total steps: {training_steps}")
    logger.info(f"Remaining steps: {training_steps - initial_step}")
    
    # Load dataset metadata
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    logger.info(f"Loading dataset metadata from: {dataset_path}")
    dataset_metadata = LeRobotDatasetMetadata(str(dataset_path))
    
    # Get model configuration and delta_timestamps using utility function
    if policy_type is None:
        # For model directories, we need to determine policy type from the model
        # Try to load the model first to get its type
        try:
            # Try ACT first
            policy = get_policy_model("act", config=None, dataset_stats=dataset_metadata.stats, pretrained_model_path=checkpoint_path)
            policy_type = "act"
        except:
            try:
                # Try diffusion
                policy = get_policy_model("diffusion", config=None, dataset_stats=dataset_metadata.stats, pretrained_model_path=checkpoint_path)
                policy_type = "diffusion"
            except:
                raise ValueError(f"Could not determine policy type from model directory: {checkpoint_path}")
        
        # Get delta_timestamps based on the loaded model's configuration
        if policy_type.lower() == "act":
            chunk_size = policy.config.chunk_size
            n_action_steps = policy.config.n_action_steps
            from train.utils import get_delta_timestamps
            delta_timestamps = get_delta_timestamps(policy_type, dataset_metadata, chunk_size=chunk_size)
            logger.info(f"Using ACT config: chunk_size={chunk_size}, n_action_steps={n_action_steps}")
        else:  # diffusion
            horizon = policy.config.horizon
            diffusion_n_action_steps = policy.config.n_action_steps
            diffusion_n_obs_steps = policy.config.n_obs_steps
            from train.utils import get_delta_timestamps
            delta_timestamps = get_delta_timestamps(policy_type, horizon=horizon, n_obs_steps=diffusion_n_obs_steps)
            logger.info(f"Using Diffusion config: horizon={horizon}, n_action_steps={diffusion_n_action_steps}, n_obs_steps={diffusion_n_obs_steps}")
    else:
        # For checkpoint files, use the utility function
        policy, delta_timestamps = get_model_config_from_checkpoint(
            checkpoint_path, policy_type, 
            dataset_metadata=dataset_metadata,
            get_policy_model_func=get_policy_model
        )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device_torch}")
    
    # Set model to training mode and move to device
    policy.train()
    policy.to(device_torch)
    
    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Load checkpoint state if it's a checkpoint file
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.pt':
        step, loss, _ = load_checkpoint(checkpoint_path, policy, optimizer)
        best_loss = loss  # Start with the loss from checkpoint
    else:
        # For model directories, start fresh
        step = initial_step
        best_loss = float('inf')
        logger.info("Starting with fresh optimizer state (no checkpoint file)")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = LeRobotDataset(str(dataset_path), delta_timestamps=delta_timestamps, video_backend=video_backend)
    logger.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=str(device_torch) != "cpu",
        drop_last=True,
    )
    
    # Training loop
    logger.info(f"Resuming training for {training_steps - step} more steps...")
    done = False
    best_step = step  # Track which step had the best loss
    
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
        best_step=best_step,
        resumed_from=str(checkpoint_path)
    )
    
    logger.info(f"{policy_type.upper()} policy training completed successfully!")
    logger.info(f"Final loss: {current_loss:.6f}")
    logger.info(f"Best model saved at step {best_step} with loss: {best_loss:.6f}")
    logger.info(f"Best model location: {output_dir}/checkpoints/best_model")


def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=Path, required=True, 
                       help="Path to checkpoint file (.pt) or model directory (best_model)")
    parser.add_argument("--dataset_path", type=Path, required=True, 
                       help="Path to the dataset")
    parser.add_argument("--output_dir", type=Path, required=True, 
                       help="Output directory for continued training")
    parser.add_argument("--training_steps", type=int, required=True, 
                       help="Total number of training steps (including previous steps)")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--log_freq", type=int, default=100, 
                       help="Logging frequency")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to train on")
    parser.add_argument("--video_backend", type=str, default="torchcodec", 
                       choices=["opencv", "torchcodec", "ffmpeg"],
                       help="Video backend to use")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {args.checkpoint_path}")
    
    if not args.dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")
    
    # Validate checkpoint path and get initial step
    if args.checkpoint_path.is_file() and args.checkpoint_path.suffix == '.pt':
        # Load checkpoint file to get initial step
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
        initial_step = checkpoint['step']
        
        if args.training_steps <= initial_step:
            raise ValueError(f"training_steps ({args.training_steps}) must be greater than initial step ({initial_step})")
    elif args.checkpoint_path.is_dir():
        # For model directories, start from step 0
        initial_step = 0
        logger.info("Loading from model directory - will start training from step 0")
    else:
        raise ValueError(f"Checkpoint path must be a .pt file or a model directory. Got: {args.checkpoint_path}")
    
    # Resume training
    resume_training(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_freq=args.log_freq,
        device=args.device,
        video_backend=args.video_backend
    )


if __name__ == "__main__":
    main() 