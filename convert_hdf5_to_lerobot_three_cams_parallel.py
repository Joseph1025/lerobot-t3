#!/usr/bin/env python
"""
Parallel HDF5 -> LeRobot converter for three cameras (high + both wrists).

This version parallelizes episode loading from HDF5 while the main writer thread
streams frames to the LeRobot dataset using its async image writer.

Usage:
    python convert_hdf5_to_lerobot_three_cams_parallel.py \
        --input_dir /path/to/hdf5 \
        --output_dir /path/to/out \
        --fps 30 \
        --num_workers 4 \
        --prefetch 4 \
        --image_writer_processes 0 \
        --image_writer_threads 12
"""

import argparse
import concurrent.futures
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_features_dict() -> Dict[str, Dict[str, Any]]:
    features = {
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.effort": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    return features


def validate_hdf5_structure(file_path: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        with h5py.File(file_path, "r") as f:
            for req in ["action", "observations"]:
                if req not in f:
                    errors.append(f"Missing dataset: {req}")
            if errors:
                return False, errors

            obs = f["observations"]
            for req in ["qpos", "effort", "images"]:
                if req not in obs:
                    errors.append(f"Missing observation: {req}")
            if errors:
                return False, errors

            imgs = obs["images"]
            for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
                if cam not in imgs:
                    errors.append(f"Missing camera: {cam}")
            if errors:
                return False, errors
    except Exception as e:
        errors.append(str(e))
        return False, errors

    return True, []


def load_hdf5_episode(file_path: Path) -> Dict[str, np.ndarray]:
    """Load a single HDF5 episode file with logging for parallelization verification."""
    thread_name = threading.current_thread().name
    logger.debug(f"[{thread_name}] Starting to load {file_path.name}")
    
    data: Dict[str, np.ndarray] = {}
    with h5py.File(file_path, "r") as f:
        data["action"] = f["action"][:]
        obs_group = f["observations"]
        data["observation.state"] = obs_group["qpos"][:]
        data["observation.effort"] = obs_group["effort"][:]
        images_group = obs_group["images"]
        data["observation.images.cam_high"] = images_group["cam_high"][:]
        data["observation.images.cam_left_wrist"] = images_group["cam_left_wrist"][:]
        data["observation.images.cam_right_wrist"] = images_group["cam_right_wrist"][:]
    
    logger.debug(f"[{thread_name}] Finished loading {file_path.name}")
    return data


def convert_parallel(
    input_dir: Path,
    output_dir: Path,
    fps: int,
    repo_id: str,
    task_name: str,
    robot_type: str,
    overwrite: bool,
    num_workers: int,
    prefetch: int,
    image_writer_processes: int,
    image_writer_threads: int,
    debug: bool = False,
    quiet_svt: bool = False,
    clean_output: bool = False,
) -> None:
    if clean_output:
        # Minimal logging - only show tqdm and essential info
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)
        quiet_svt = True  # Always suppress SVT in clean mode
        logger.warning("Clean output mode - minimal logging enabled")
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info(f"Debug logging enabled. Workers: {num_workers}, Prefetch: {prefetch}")
    
    # Filter out SVT/ffmpeg noise if requested
    if quiet_svt:
        # Suppress ffmpeg/libav logging
        logging.getLogger("libav").setLevel(logging.ERROR)
        logging.getLogger("av").setLevel(logging.ERROR)
        
        # Suppress subprocess output that might contain SVT info
        import os
        os.environ['FFREPORT'] = '0'  # Disable ffmpeg reports
        os.environ['AV_LOG_LEVEL'] = 'error'  # Set libav to error only
        
        if not clean_output:
            logger.info("SVT/ffmpeg logging suppressed - only tqdm and parallelization info will be shown")
    
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found in {input_dir}")

    if not clean_output:
        logger.info(f"Found {len(hdf5_files)} HDF5 files")
        logger.info(f"Parallelization: {num_workers} workers, {prefetch} prefetch, {image_writer_threads} image writer threads")

    ok, errs = validate_hdf5_structure(hdf5_files[0])
    if not ok:
        raise ValueError("Validation failed: " + "; ".join(errs))

    if output_dir.exists():
        if overwrite:
            import shutil
            logger.info(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(
                f"Output directory {output_dir} already exists. Use --overwrite to overwrite it."
            )

    features = create_features_dict()
    logger.info("Creating LeRobot dataset with async image writer...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    total_frames = 0
    start_time = time.time()

    # Producer pool to load episodes concurrently
    max_workers = max(1, num_workers)
    queue_size = max(1, prefetch)

    # Iterate in submission order but consume as episodes complete, keeping limited in-flight episodes
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hdf5") as ex:
        in_flight: List[Tuple[concurrent.futures.Future, int]] = []  # (future, episode_index)
        submitted_idx = 0
        completed_idx = 0

        # Prime the queue
        logger.info(f"Priming queue with {min(queue_size, len(hdf5_files))} episodes...")
        while submitted_idx < len(hdf5_files) and len(in_flight) < queue_size:
            future = ex.submit(load_hdf5_episode, hdf5_files[submitted_idx])
            in_flight.append((future, submitted_idx))  # Track which episode this future represents
            submitted_idx += 1

        episode_idx = 0
        
        # Create progress bars
        episode_pbar = tqdm(
            total=len(hdf5_files),
            desc="Processing episodes",
            unit="episode",
            position=0,
            leave=True,
            ncols=100,  # Fixed width for consistent display
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        frame_pbar = tqdm(
            total=0,  # Will be updated per episode
            desc="Processing frames",
            unit="frame",
            position=1,
            leave=True,
            ncols=100,  # Fixed width for consistent display
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Process episodes in submission order (not completion order)
        while completed_idx < len(hdf5_files):
            # Wait for the next completed future in completion order
            done_futures = [fut for fut, _ in in_flight if fut.done()]
            if not done_futures:
                # Wait for any future to complete
                done, _ = concurrent.futures.wait([fut for fut, _ in in_flight], return_when=concurrent.futures.FIRST_COMPLETED)
                done_futures = list(done)
            
            # Find the episode that should be processed next (in submission order)
            next_episode_idx = None
            next_future = None
            
            for future, ep_idx in in_flight:
                if future.done() and ep_idx == episode_idx:
                    next_episode_idx = ep_idx
                    next_future = future
                    break
            
            if next_episode_idx is None:
                # Wait for the next episode in sequence to complete
                continue
            
            # Process this episode
            episode_data = next_future.result()
            
            # Remove from in-flight
            in_flight.remove((next_future, next_episode_idx))
            
            # Write this episode sequentially to ensure consistent indices on disk
            dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_idx)
            ep_len = int(episode_data["action"].shape[0])
            
            # Update frame progress bar for this episode
            frame_pbar.reset(total=ep_len)
            frame_pbar.set_description(f"Episode {episode_idx + 1}/{len(hdf5_files)}")
            
            for frame_idx in range(ep_len):
                frame = {
                    "action": episode_data["action"][frame_idx],
                    "observation.state": episode_data["observation.state"][frame_idx],
                    "observation.effort": episode_data["observation.effort"][frame_idx],
                    "observation.images.cam_high": episode_data["observation.images.cam_high"][frame_idx],
                    "observation.images.cam_left_wrist": episode_data["observation.images.cam_left_wrist"][frame_idx],
                    "observation.images.cam_right_wrist": episode_data["observation.images.cam_right_wrist"][frame_idx],
                    "next.done": np.array([frame_idx == ep_len - 1], dtype=np.bool_),
                }
                dataset.add_frame(frame, task=task_name)
                frame_pbar.update(1)

            dataset.save_episode()
            total_frames += ep_len
            episode_idx += 1
            completed_idx += 1
            
            # Update episode progress
            episode_pbar.update(1)
            episode_pbar.set_postfix({
                'frames': total_frames,
                'avg_frames_per_ep': total_frames // (episode_idx),
                'remaining_episodes': len(hdf5_files) - episode_idx
            })

            # Top up the in-flight queue
            if submitted_idx < len(hdf5_files):
                future = ex.submit(load_hdf5_episode, hdf5_files[submitted_idx])
                in_flight.append((future, submitted_idx))
                submitted_idx += 1

        # Close progress bars
        episode_pbar.close()
        frame_pbar.close()

    dataset.stop_image_writer()
    
    # Calculate performance stats
    total_time = time.time() - start_time
    episodes_per_second = len(hdf5_files) / total_time
    frames_per_second = total_frames / total_time

    video_files = list(output_dir.rglob("*.mp4"))
    
    if clean_output:
        # Clean summary with just the essential info
        print(f"\nConversion Complete!")
        print(f"Output: {output_dir}")
        print(f"Episodes: {len(hdf5_files)} | Frames: {total_frames:,}")
        print(f"Performance: {episodes_per_second:.1f} ep/s | {frames_per_second:.0f} frames/s")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Videos: {len(video_files)} MP4 files created")
    else:
        # Standard logging
        logger.info(f"Created {len(video_files)} video files")
        logger.info(
            f"Conversion complete. Episodes: {len(hdf5_files)}, Frames: {total_frames}, Output: {output_dir}"
        )
        logger.info(f"Performance: {episodes_per_second:.2f} episodes/sec, {frames_per_second:.2f} frames/sec")
        logger.info(f"Total time: {total_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel converter: HDF5 -> LeRobot (three cameras)"
    )
    parser.add_argument("--input_dir", type=Path, help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", type=Path, help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo_id", type=str, default="my_robot_dataset_three_cams")
    parser.add_argument("--task_name", type=str, default="pick_blue_cube")
    parser.add_argument("--robot_type", type=str, default="agilex")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4, help="Episode loader threads")
    parser.add_argument("--prefetch", type=int, default=4, help="Max in-flight episodes")
    parser.add_argument("--image_writer_processes", type=int, default=0)
    parser.add_argument("--image_writer_threads", type=int, default=12)
    parser.add_argument("--debug", action="store_true", help="Enable detailed logging")
    parser.add_argument("--quiet-svt", action="store_true", help="Suppress SVT/ffmpeg logging")
    parser.add_argument("--clean-output", action="store_true", help="Run in clean output mode (minimal logging)")

    args = parser.parse_args()

    # Validate required arguments for actual conversion
    if not args.input_dir:
        parser.error("--input_dir is required for conversion")
    if not args.output_dir:
        parser.error("--output_dir is required for conversion")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    convert_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        repo_id=args.repo_id,
        task_name=args.task_name,
        robot_type=args.robot_type,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        prefetch=args.prefetch,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        debug=args.debug,
        quiet_svt=args.quiet_svt,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()


