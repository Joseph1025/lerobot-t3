"""
Evaluates a given model on a given dataset. Log its performance metrics to a file.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# Import LeRobot components
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('evaluation.log')
        ]
    )


def load_model(model_path: str) -> nn.Module:
    """
    Loads a model from a given path.
    
    Args:
        model_path: Path to the model checkpoint or directory
        
    Returns:
        Loaded model in evaluation mode
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    logging.info(f"Loading model from: {model_path}")
    
    # Check if it's a directory (LeRobot format) or file
    if model_path.is_dir():
        # Try to load as LeRobot policy
        try:
            # Check for config.json to determine policy type
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                policy_type = config.get('type', 'diffusion')
                
                if policy_type.lower() == 'diffusion':
                    model = DiffusionPolicy.from_pretrained(str(model_path))
                elif policy_type.lower() == 'act':
                    model = ACTPolicy.from_pretrained(str(model_path))
                else:
                    raise ValueError(f"Unsupported policy type: {policy_type}")
            else:
                # Default to diffusion if no config found
                model = DiffusionPolicy.from_pretrained(str(model_path))
                
        except Exception as e:
            logging.error(f"Failed to load as LeRobot policy: {e}")
            raise
    else:
        # Load as regular PyTorch checkpoint
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Standard checkpoint format
                model_state_dict = checkpoint['model_state_dict']
                # You would need to know the model architecture here
                # For now, we'll raise an error
                raise NotImplementedError("Regular PyTorch checkpoint loading not implemented yet")
            else:
                raise ValueError("Unknown checkpoint format")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise
    
    model.eval()
    logging.info(f"Successfully loaded model: {type(model).__name__}")
    return model


def load_dataset(dataset_path: str) -> LeRobotDataset:
    """
    Loads a dataset from a given path.
    
    Args:
        dataset_path: Path to the dataset or dataset repo_id
        
    Returns:
        Loaded LeRobot dataset
    """
    logging.info(f"Loading dataset from: {dataset_path}")
    
    try:
        # Check if it's a local path or repo_id
        if os.path.exists(dataset_path):
            # Local path
            dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path)
        else:
            # Assume it's a repo_id
            dataset = LeRobotDataset(repo_id=dataset_path)
        
        logging.info(f"Successfully loaded dataset with {dataset.num_episodes} episodes and {dataset.num_frames} frames")
        return dataset
        
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise


def evaluate(model: nn.Module, dataset: LeRobotDataset, output_path: str) -> Dict[str, Any]:
    """
    Evaluates a model on a given dataset.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        output_path: Path to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logging.info("Starting evaluation...")
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    logging.info(f"Model device: {device}")
    
    # Initialize metrics
    total_loss = 0.0
    total_samples = 0
    action_errors = []
    predictions = []
    targets = []
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on a subset of the dataset for efficiency
    eval_size = min(1000, len(dataset))  # Evaluate on max 1000 samples
    eval_indices = random.sample(range(len(dataset)), eval_size)
    
    logging.info(f"Evaluating on {eval_size} samples...")
    
    # Log dataset structure for debugging
    if len(dataset) > 0:
        sample = dataset[0]
        logging.info("Dataset sample keys:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                logging.info(f"  {k}: {v.shape} ({v.dtype})")
            else:
                logging.info(f"  {k}: {type(v)}")
    
    with torch.no_grad():
        for i, idx in enumerate(eval_indices):
            if i % 100 == 0:
                logging.info(f"Progress: {i}/{eval_size}")
            
            try:
                # Get sample from dataset
                sample = dataset[idx]
                
                # Prepare inputs for the model
                # This depends on the model type and dataset structure
                if hasattr(model, 'select_action'):
                    # For policy models that have select_action method
                    observation = {}
                    
                    # Process observation data
                    for k, v in sample.items():
                        if k.startswith('observation'):
                            if isinstance(v, torch.Tensor):
                                # For diffusion policy, we need to preprocess images before passing to select_action
                                if k.startswith('observation.images') and hasattr(model, 'config') and hasattr(model.config, 'crop_shape'):
                                    # Apply center cropping to match the model's expected input size
                                    crop_shape = model.config.crop_shape
                                    if crop_shape is not None and v.shape[-2:] != crop_shape:
                                        # Center crop the image
                                        h, w = v.shape[-2:]
                                        crop_h, crop_w = crop_shape
                                        start_h = (h - crop_h) // 2
                                        start_w = (w - crop_w) // 2
                                        v = v[..., start_h:start_h + crop_h, start_w:start_w + crop_w]
                                        logging.debug(f"Cropped {k} from {v.shape} to {v.shape}")
                                
                                # Ensure images are in [0, 1] range
                                if k.startswith('observation.images'):
                                    if v.max() > 1.0:
                                        v = v.float() / 255.0
                                    else:
                                        v = v.float()
                                    # Log the shape for debugging
                                    logging.info(f"Image tensor {k} shape before stacking: {v.shape}")
                                
                                observation[k] = v.to(device)
                            else:
                                observation[k] = v
                    
                    # If this is a diffusion policy, log the shape of the stacked images tensor
                    if hasattr(model, 'config') and hasattr(model.config, 'image_features') and hasattr(model, 'select_action'):
                        image_keys = list(model.config.image_features)
                        if all(k in observation for k in image_keys):
                            stacked = torch.stack([observation[k] for k in image_keys], dim=-4)
                            logging.info(f"Stacked images tensor shape (before select_action): {stacked.shape}")
                    # Get model prediction
                    predicted_action = model.select_action(observation)
                    
                    # Get target action
                    target_action = sample['action'].to(device)
                    
                    # Calculate action error
                    action_error = torch.mean((predicted_action - target_action) ** 2).item()
                    action_errors.append(action_error)
                    
                    predictions.append(predicted_action.cpu().numpy())
                    targets.append(target_action.cpu().numpy())
                    
                else:
                    # For regular models, use forward pass
                    # This is a simplified version - you might need to adapt based on your model
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in sample.items() if k != 'action'}
                    
                    outputs = model(inputs)
                    
                    # Calculate loss (this depends on your model's output format)
                    if isinstance(outputs, torch.Tensor):
                        target = sample['action'].to(device)
                        loss = nn.functional.mse_loss(outputs, target)
                        total_loss += loss.item()
                        total_samples += 1
                
            except Exception as e:
                logging.warning(f"Error processing sample {idx}: {e}")
                continue
    
    # Calculate metrics
    metrics = {
        'total_samples': total_samples,
        'eval_samples': len(action_errors),
    }
    
    if action_errors:
        metrics.update({
            'mean_action_error': np.mean(action_errors),
            'std_action_error': np.std(action_errors),
            'min_action_error': np.min(action_errors),
            'max_action_error': np.max(action_errors),
        })
    
    if total_samples > 0:
        metrics['mean_loss'] = total_loss / total_samples
    
    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and targets if available
    if predictions and targets:
        np.save(output_dir / 'predictions.npy', np.array(predictions))
        np.save(output_dir / 'targets.npy', np.array(targets))
    
    logging.info(f"Evaluation completed. Results saved to {output_dir}")
    logging.info(f"Metrics: {metrics}")
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model checkpoint or directory")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the dataset or dataset repo_id")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save evaluation results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    # Setup
    args = parse_args()
    setup_logging()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    try:
        # Load model and dataset
        model = load_model(args.model_path)
        dataset = load_dataset(args.dataset_path)
        
        # Evaluate
        metrics = evaluate(model, dataset, args.output_path)
        
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
