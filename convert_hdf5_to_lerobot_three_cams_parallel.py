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
) -> None:
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found in {input_dir}")

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

    # Producer pool to load episodes concurrently
    max_workers = max(1, num_workers)
    queue_size = max(1, prefetch)

    def _submit_all(executor: concurrent.futures.Executor) -> List[concurrent.futures.Future]:
        futures: List[concurrent.futures.Future] = []
        for fpath in hdf5_files:
            futures.append(executor.submit(load_hdf5_episode, fpath))
        return futures

    # Iterate in submission order but consume as episodes complete, keeping limited in-flight episodes
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hdf5") as ex:
        in_flight: List[concurrent.futures.Future] = []
        submitted_idx = 0
        completed_idx = 0

        # Prime the queue
        while submitted_idx < len(hdf5_files) and len(in_flight) < queue_size:
            in_flight.append(ex.submit(load_hdf5_episode, hdf5_files[submitted_idx]))
            submitted_idx += 1

        episode_idx = 0
        
        # Create progress bars
        episode_pbar = tqdm(
            total=len(hdf5_files),
            desc="Processing episodes",
            unit="episode",
            position=0,
            leave=True
        )
        
        frame_pbar = tqdm(
            total=0,  # Will be updated per episode
            desc="Processing frames",
            unit="frame",
            position=1,
            leave=True
        )
        
        while completed_idx < len(hdf5_files):
            # Wait for the next completed future in completion order
            done, _ = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in list(done):
                in_flight.remove(fut)
                episode_data = fut.result()

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
                    in_flight.append(ex.submit(load_hdf5_episode, hdf5_files[submitted_idx]))
                    submitted_idx += 1

        # Close progress bars
        episode_pbar.close()
        frame_pbar.close()

    dataset.stop_image_writer()

    video_files = list(output_dir.rglob("*.mp4"))
    logger.info(f"Created {len(video_files)} video files")
    logger.info(
        f"Conversion complete. Episodes: {len(hdf5_files)}, Frames: {total_frames}, Output: {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parallel converter: HDF5 -> LeRobot (three cameras)"
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo_id", type=str, default="my_robot_dataset_three_cams")
    parser.add_argument("--task_name", type=str, default="pick_blue_cube")
    parser.add_argument("--robot_type", type=str, default="agilex")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4, help="Episode loader threads")
    parser.add_argument("--prefetch", type=int, default=4, help="Max in-flight episodes")
    parser.add_argument("--image_writer_processes", type=int, default=0)
    parser.add_argument("--image_writer_threads", type=int, default=12)

    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()


