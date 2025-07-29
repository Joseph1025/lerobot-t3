#!/usr/bin/env python
"""
Training Utilities

Common utility functions for training and resuming LeRobot policies.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features

logger = logging.getLogger(__name__)


def get_delta_timestamps(policy_type: str, dataset_metadata=None, chunk_size=10, horizon=16, n_obs_steps=2):
    """
    Get delta_timestamps configuration based on policy type.
    
    Args:
        policy_type: Type of policy ("diffusion" or "act")
        dataset_metadata: LeRobotDatasetMetadata (needed for ACT fps)
        chunk_size: Number of actions in the sequence for ACT
        horizon: Horizon size for diffusion policy
        n_obs_steps: Number of observation steps for diffusion policy
    Returns:
        Delta timestamps dictionary
    """
    if policy_type.lower() == "diffusion":
        # Diffusion policy: previous + current observations, future actions
        # For horizon=16, we need 16 action timestamps
        # The pattern is: [previous_obs, current_obs, future_actions...]
        # With n_obs_steps=2, we have 2 observation timestamps and horizon action timestamps
        action_timestamps = [i * 0.1 for i in range(-n_obs_steps + 1, horizon - n_obs_steps + 1)]
        obs_timestamps = [i * 0.1 for i in range(-n_obs_steps + 1, 1)]
        return {
            "observation.images.cam_high": obs_timestamps,
            "observation.images.cam_left_wrist": obs_timestamps,
            "observation.images.cam_right_wrist": obs_timestamps,
            "observation.state": obs_timestamps,
            "observation.effort": obs_timestamps,
            "action": action_timestamps,
        }
    elif policy_type.lower() == "act":
        # ACT policy: provide action sequence, single timestep for others
        if dataset_metadata is None:
            raise ValueError("dataset_metadata must be provided for ACT policy delta_timestamps.")
        fps = dataset_metadata.fps
        # Use a reasonable chunk_size (default 10, can be changed)
        return {
            "observation.images.cam_high": [0.0],
            "observation.images.cam_left_wrist": [0.0],
            "observation.images.cam_right_wrist": [0.0],
            "observation.state": [0.0],
            "observation.effort": [0.0],
            "action": [i / fps for i in range(chunk_size)],
        }
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")


def save_checkpoint(policy, optimizer, step: int, loss: float, output_dir: Path, is_best: bool = False):
    """
    Save a checkpoint of the model.
    
    Args:
        policy: The policy model to save
        optimizer: The optimizer state to save
        step: Current training step
        loss: Current loss value
        output_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    # Create checkpoints directory
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Save regular checkpoint every 1000 steps (not every step)
    if step % 1000 == 0:
        checkpoint_path = checkpoints_dir / f"checkpoint_step_{step}.pt"
        torch.save({
            'step': step,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'policy_type': policy.name,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")
    
    # Save best model (silently, will log at end of training)
    if is_best:
        best_model_path = checkpoints_dir / "best_model"
        best_model_path.mkdir(exist_ok=True)
        
        # Save the model using the policy's save_pretrained method
        policy.save_pretrained(best_model_path)
        
        # Save additional training info
        training_info = {
            'step': step,
            'loss': loss,
            'policy_type': policy.name,
            'is_best': True
        }
        with open(best_model_path / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)


def load_checkpoint(checkpoint_path: Path, policy, optimizer):
    """
    Load a checkpoint and restore model and optimizer state.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        policy: The policy model
        optimizer: The optimizer
        
    Returns:
        step: The step number from the checkpoint
        loss: The loss value from the checkpoint
        policy_type: The policy type from the checkpoint
    """
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    policy.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint['step']
    loss = checkpoint['loss']
    policy_type = checkpoint['policy_type']
    
    logger.info(f"Loaded checkpoint from step {step} with loss {loss:.6f}")
    
    return step, loss, policy_type


def get_model_config_from_checkpoint(checkpoint_path: Path, policy_type: str, dataset_metadata, get_policy_model_func):
    """
    Get model configuration from checkpoint or create a new one with default settings.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        policy_type: Type of policy
        dataset_metadata: Dataset metadata (can be None, will be loaded if needed)
        get_policy_model_func: Function to create policy model
        
    Returns:
        policy: The policy model with loaded configuration
        delta_timestamps: Delta timestamps for dataset loading
    """
    # Load dataset metadata if not provided
    if dataset_metadata is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        # Try to find dataset path from checkpoint directory structure
        checkpoint_dir = checkpoint_path.parent
        # Look for dataset in parent directories or use a default path
        # For now, we'll need the dataset path to be passed separately
        raise ValueError("dataset_metadata must be provided for get_model_config_from_checkpoint")
    
    # Try to find a best_model directory in the same parent directory as the checkpoint
    checkpoint_dir = checkpoint_path.parent
    best_model_dir = checkpoint_dir / "best_model"
    
    # Also check if there's a config.json in the checkpoint directory
    config_file = checkpoint_dir / "config.json"
    
    if best_model_dir.exists():
        # Load configuration from best_model directory
        logger.info(f"Loading model configuration from best_model directory: {best_model_dir}")
        policy = get_policy_model_func(policy_type, config=None, dataset_stats=dataset_metadata.stats, pretrained_model_path=best_model_dir)
    elif config_file.exists():
        # Load configuration from config.json in checkpoint directory
        logger.info(f"Loading model configuration from config file: {config_file}")
        policy = get_policy_model_func(policy_type, config=None, dataset_stats=dataset_metadata.stats, pretrained_model_path=checkpoint_dir)
    else:
        # Create a new model with default configuration
        logger.info(f"No configuration found, creating new model with default settings")
        policy = get_policy_model_func(policy_type, config=None, dataset_stats=dataset_metadata.stats)
    
    # Get delta_timestamps based on the loaded model's configuration
    if policy_type.lower() == "act":
        chunk_size = policy.config.chunk_size
        n_action_steps = policy.config.n_action_steps
        delta_timestamps = get_delta_timestamps(policy_type, dataset_metadata, chunk_size=chunk_size)
        logger.info(f"Using ACT config: chunk_size={chunk_size}, n_action_steps={n_action_steps}")
    else:  # diffusion
        horizon = policy.config.horizon
        diffusion_n_action_steps = policy.config.n_action_steps
        diffusion_n_obs_steps = policy.config.n_obs_steps
        delta_timestamps = get_delta_timestamps(policy_type, horizon=horizon, n_obs_steps=diffusion_n_obs_steps)
        logger.info(f"Using Diffusion config: horizon={horizon}, n_action_steps={diffusion_n_action_steps}, n_obs_steps={diffusion_n_obs_steps}")
    
    return policy, delta_timestamps


def prepare_act_batch(batch: Dict[str, Any], policy, device_torch):
    """
    Prepare batch for ACT policy training.
    
    Args:
        batch: Input batch dictionary
        policy: ACT policy model
        device_torch: Torch device
        
    Returns:
        Prepared batch dictionary
    """
    if "action" in batch:
        # Ensure action has shape [B, S, D] where S is chunk_size
        if batch["action"].dim() == 2:  # [B, D]
            # Repeat the action to create a sequence
            batch["action"] = batch["action"].unsqueeze(1).repeat(1, policy.config.chunk_size, 1)
        
        # For ACT, we need to ensure observation.state has the correct shape [B, D]
        if "observation.state" in batch:
            if batch["observation.state"].ndim == 3 and batch["observation.state"].shape[1] == 1:
                batch["observation.state"] = batch["observation.state"].squeeze(1)
            elif batch["observation.state"].ndim == 3 and batch["observation.state"].shape[1] > 1:
                batch["observation.state"] = batch["observation.state"][:, 0, :]
        
        # Add action_is_pad tensor if not present
        if "action_is_pad" not in batch:
            batch_size = batch["action"].shape[0]
            batch["action_is_pad"] = torch.zeros(batch_size, policy.config.chunk_size, dtype=torch.bool, device=device_torch)
    
    return batch


def save_training_info(output_dir: Path, step: int, current_loss: float, best_loss: float, 
                      policy_type: str, training_steps: int, learning_rate: float, 
                      batch_size: int, best_step: int = 0, resumed_from: Optional[str] = None):
    """
    Save training information to JSON file.
    
    Args:
        output_dir: Output directory
        step: Current training step
        current_loss: Current loss value
        best_loss: Best loss achieved
        policy_type: Type of policy
        training_steps: Total training steps
        learning_rate: Learning rate used
        batch_size: Batch size used
        best_step: Step number where best loss was achieved
        resumed_from: Path to checkpoint if resuming (optional)
    """
    final_info = {
        'final_step': step,
        'final_loss': current_loss,
        'best_loss': best_loss,
        'best_step': best_step,
        'policy_type': policy_type,
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
    }
    
    if resumed_from:
        final_info['resumed_from'] = resumed_from
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(final_info, f, indent=2)


def validate_training_parameters(policy_type: str, pretrained_model_path: Optional[Path] = None,
                               chunk_size: int = 16, n_action_steps: int = 16,
                               horizon: int = 16, diffusion_n_action_steps: int = 8,
                               diffusion_n_obs_steps: int = 2):
    """
    Validate training parameters.
    
    Args:
        policy_type: Type of policy ("diffusion", "act", or "act_t3")
        pretrained_model_path: Path to pretrained model
        chunk_size: Size of action chunks for ACT
        n_action_steps: Number of action steps to execute for ACT
        horizon: Horizon size for diffusion policy
        diffusion_n_action_steps: Number of action steps to execute for diffusion policy
        diffusion_n_obs_steps: Number of observation steps for diffusion policy
    """
    if policy_type.lower() not in ["diffusion", "act", "act_t3"]:
        raise ValueError(f"Unsupported policy type: {policy_type}. Supported types: 'diffusion', 'act', 'act_t3'")
    
    if pretrained_model_path is not None and not pretrained_model_path.exists():
        raise ValueError(f"Pretrained model path does not exist: {pretrained_model_path}")
    
    if policy_type.lower() == "act" or policy_type.lower() == "act_t3":
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
        if n_action_steps > chunk_size:
            raise ValueError(f"n_action_steps ({n_action_steps}) cannot be greater than chunk_size ({chunk_size})")
    
    elif policy_type.lower() == "diffusion":
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if horizon % 8 != 0:
            raise ValueError(f"horizon ({horizon}) must be divisible by 8 for diffusion policy")
        if diffusion_n_action_steps <= 0:
            raise ValueError(f"diffusion_n_action_steps must be positive, got {diffusion_n_action_steps}")
        if diffusion_n_obs_steps <= 0:
            raise ValueError(f"diffusion_n_obs_steps must be positive, got {diffusion_n_obs_steps}")
        if diffusion_n_action_steps > horizon:
            raise ValueError(f"diffusion_n_action_steps ({diffusion_n_action_steps}) cannot be greater than horizon ({horizon})")


def get_dataset_features(dataset_path: Path):
    """
    Get dataset features from metadata.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        tuple: (input_features, output_features, dataset_metadata)
    """
    logger.info(f"Loading dataset metadata from: {dataset_path}")
    dataset_metadata = LeRobotDatasetMetadata(str(dataset_path))
    
    # Convert dataset features to policy features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    logger.info(f"Input features: {list(input_features.keys())}")
    logger.info(f"Output features: {list(output_features.keys())}")
    
    return input_features, output_features, dataset_metadata 