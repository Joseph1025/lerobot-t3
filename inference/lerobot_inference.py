#!/usr/bin/env python3
"""
LeRobot Inference Script - Fixed Version
Properly handles normalization differences between custom and LeRobot formats
"""

import torch
import numpy as np
import os
import pickle
import argparse
from pathlib import Path
import time
import threading
from collections import deque

# ROS imports
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# LeRobot imports
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

# Global variables for temporal aggregation (from original inference.py)
inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


class LeRobotInferenceFixed:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize data queues FIRST (before ROS callbacks)
        self.img_left_deque = deque(maxlen=100)
        self.img_right_deque = deque(maxlen=100)
        self.img_front_deque = deque(maxlen=100)
        self.puppet_arm_left_deque = deque(maxlen=100)
        self.puppet_arm_right_deque = deque(maxlen=100)
        
        # Initialize ROS AFTER data queues (this sets up publishers)
        self.init_ros()
        
        # Load policy and stats
        self.policy = self.load_policy()
        self.stats = self.load_stats()
        
        # Calculate optimal action scale once
        self.optimal_action_scale = self.calculate_optimal_action_scale()
        
        print("LeRobot Inference initialized")

    def apply_temporal_aggregation(self, all_time_actions, t, chunk_size, action_dim):
        """Apply temporal aggregation with exponential weighting (from original inference.py)"""
        actions_for_curr_step = all_time_actions[:, t]
        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        
        if len(actions_for_curr_step) == 0:
            return None
        
        # Exponential weighting with k = 0.01 (from original inference.py)
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights[:, np.newaxis]
        
        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
        return raw_action

    def inference_process(self, t, pre_action):
        """Inference process that runs in a separate thread (from original inference.py)"""
        global inference_lock, inference_actions, inference_timestep
        
        try:
            # Get latest observation
            data = self.get_latest_data()
            if data is None:
                return
            
            img_left, img_right, img_front, puppet_arm_left, puppet_arm_right = data
            
            # Preprocess observation
            observation = self.preprocess_observation(
                img_left, img_right, img_front, puppet_arm_left, puppet_arm_right
            )
            
            # Get action from policy
            with torch.inference_mode():
                action = self.policy.select_action(observation)
            
            # Convert to numpy and add temporal dimension
            action_np = action.cpu().numpy()
            if len(action_np.shape) == 2:  # (batch, action_dim)
                action_np = action_np[np.newaxis, :, :]  # (1, batch, action_dim)
            
            # Store in global variable
            inference_lock.acquire()
            inference_actions = action_np
            inference_timestep = t
            inference_lock.release()
            
        except Exception as e:
            print(f"Error in inference process: {e}")
            import traceback
            traceback.print_exc()

    def init_ros(self):
        """Initialize ROS node and subscribers/publishers"""
        try:
            rospy.init_node('lerobot_inference_fixed', anonymous=True)
        except rospy.exceptions.ROSException:
            pass  # Node already initialized
        
        # Subscribers
        rospy.Subscriber("/camera1/color/image_raw", Image, self.img_left_callback, queue_size=10)
        rospy.Subscriber("/camera2/color/image_raw", Image, self.img_right_callback, queue_size=10)
        rospy.Subscriber("/camera3/color/image_raw", Image, self.img_front_callback, queue_size=10)
        rospy.Subscriber("/puppet/joint_left", JointState, self.puppet_arm_left_callback, queue_size=10)
        rospy.Subscriber("/puppet/joint_right", JointState, self.puppet_arm_right_callback, queue_size=10)
        
        # Publishers
        self.puppet_arm_left_publisher = rospy.Publisher("/left_joint_states", JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher("/right_joint_states", JointState, queue_size=10)

    def load_policy(self):
        """Load LeRobot policy from checkpoint directory"""
        if self.args.policy_type.lower() == "diffusion":
            policy = DiffusionPolicy.from_pretrained(self.args.ckpt_dir)
        elif self.args.policy_type.lower() == "act":
            # Load ACT policy - configuration will be loaded from config.json
            policy = ACTPolicy.from_pretrained(self.args.ckpt_dir)
            
            # Use the chunk_size and n_action_steps from the model's config.json
            chunk_size = policy.config.chunk_size
            n_action_steps = policy.config.n_action_steps
            
            print(f"Using ACT configuration from model:")
            print(f"  - chunk_size: {chunk_size}")
            print(f"  - n_action_steps: {n_action_steps}")
            
            # Only override if explicitly specified and different from config
            if hasattr(self.args, 'chunk_size') and self.args.chunk_size is not None:
                if self.args.chunk_size != chunk_size:
                    policy.config.chunk_size = self.args.chunk_size
                    print(f"  - Overriding chunk_size to: {self.args.chunk_size}")
                else:
                    print(f"  - chunk_size already matches command line argument")
            
            if hasattr(self.args, 'n_action_steps') and self.args.n_action_steps is not None:
                if self.args.n_action_steps != n_action_steps:
                    policy.config.n_action_steps = self.args.n_action_steps
                    print(f"  - Overriding n_action_steps to: {self.args.n_action_steps}")
                else:
                    print(f"  - n_action_steps already matches command line argument")
            
            # Reset the policy to apply configuration
            policy.reset()
            
        else:
            raise ValueError(f"Unsupported policy type: {self.args.policy_type}")
        
        policy.to(self.device)
        policy.eval()
        print(f"Loaded {self.args.policy_type} policy from {self.args.ckpt_dir}")
        return policy

    def load_stats(self):
        """Load and convert stats to LeRobot format"""
        stats_path = os.path.join(self.args.ckpt_dir, "dataset_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                custom_stats = pickle.load(f)
            
            # Convert custom stats format to LeRobot format
            lerobot_stats = {}
            
            # Convert qpos stats
            if 'qpos_mean' in custom_stats and 'qpos_std' in custom_stats:
                lerobot_stats["observation.state"] = {
                    "mean": torch.from_numpy(custom_stats['qpos_mean']).float(),
                    "std": torch.from_numpy(custom_stats['qpos_std']).float(),
                }
            
            # Convert action stats
            if 'action_mean' in custom_stats and 'action_std' in custom_stats:
                lerobot_stats["action"] = {
                    "mean": torch.from_numpy(custom_stats['action_mean']).float(),
                    "std": torch.from_numpy(custom_stats['action_std']).float(),
                }
            
            print(f"Converted stats from {stats_path}")
            return lerobot_stats
        else:
            print(f"Warning: No stats file found at {stats_path}")
            return None

    def preprocess_observation(self, img_left, img_right, img_front, puppet_arm_left, puppet_arm_right):
        """Convert ROS data to LeRobot observation format with proper normalization"""
        # Convert images to tensors
        bridge = CvBridge()
        
        try:
            # Convert ROS Image messages to numpy arrays
            img_left_np = bridge.imgmsg_to_cv2(img_left, 'rgb8')
            img_right_np = bridge.imgmsg_to_cv2(img_right, 'rgb8')
            img_front_np = bridge.imgmsg_to_cv2(img_front, 'rgb8')
        except Exception as e:
            print(f"Error converting images: {e}")
            print("This might be due to NumPy version compatibility. Try downgrading NumPy to <2.0")
            raise
        
        # Convert to tensors and normalize to [0,1]
        img_left_tensor = torch.from_numpy(img_left_np).float() / 255.0
        img_right_tensor = torch.from_numpy(img_right_np).float() / 255.0
        img_front_tensor = torch.from_numpy(img_front_np).float() / 255.0
        
        # Rearrange to channel-first format
        img_left_tensor = img_left_tensor.permute(2, 0, 1)
        img_right_tensor = img_right_tensor.permute(2, 0, 1)
        img_front_tensor = img_front_tensor.permute(2, 0, 1)
        
        # Concatenate joint positions
        qpos = np.concatenate([
            np.array(puppet_arm_left.position),
            np.array(puppet_arm_right.position)
        ])
        
        # Normalize qpos if stats available
        if self.stats is not None and "observation.state" in self.stats:
            qpos_mean = self.stats["observation.state"]["mean"]
            qpos_std = self.stats["observation.state"]["std"]
            qpos = (qpos - qpos_mean.cpu().numpy()) / qpos_std.cpu().numpy()
        
        qpos_tensor = torch.from_numpy(qpos).float()
        
        # Add batch dimension
        img_left_tensor = img_left_tensor.unsqueeze(0)
        img_right_tensor = img_right_tensor.unsqueeze(0)
        img_front_tensor = img_front_tensor.unsqueeze(0)
        qpos_tensor = qpos_tensor.unsqueeze(0)
        
        # Move to device
        img_left_tensor = img_left_tensor.to(self.device)
        img_right_tensor = img_right_tensor.to(self.device)
        img_front_tensor = img_front_tensor.to(self.device)
        qpos_tensor = qpos_tensor.to(self.device)
        
        # Create observation dictionary matching LeRobot format
        observation = {
            "observation.images.cam_high": img_front_tensor,
            "observation.images.cam_left_wrist": img_left_tensor,
            "observation.images.cam_right_wrist": img_right_tensor,
            "observation.state": qpos_tensor,
        }
        
        return observation

    def postprocess_action(self, action):
        """Convert policy action to robot commands with proper denormalization"""
        # Remove batch dimension
        action = action.squeeze(0)
        
        # Move to CPU and convert to numpy
        action = action.cpu().numpy()
        
        # Denormalize if stats available
        if self.stats is not None and "action" in self.stats:
            action_mean = self.stats["action"]["mean"]
            action_std = self.stats["action"]["std"]
            action = action * action_std.cpu().numpy() + action_mean.cpu().numpy()
        
        # Scale up actions to make movements more visible
        action = action * self.optimal_action_scale
        
        # Split into left and right arm actions
        left_action = action[:7]
        right_action = action[7:14]
        
        return left_action, right_action

    def publish_arm_commands(self, left_action, right_action):
        """Publish joint commands to robot arms"""
        # Convert to lists if they're numpy arrays
        if hasattr(left_action, 'tolist'):
            left_action = left_action.tolist()
        if hasattr(right_action, 'tolist'):
            right_action = right_action.tolist()
        
        # Left arm
        left_msg = JointState()
        left_msg.header = Header()
        left_msg.header.stamp = rospy.Time.now()
        left_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        left_msg.position = left_action
        self.puppet_arm_left_publisher.publish(left_msg)
        
        # Right arm
        right_msg = JointState()
        right_msg.header = Header()
        right_msg.header.stamp = rospy.Time.now()
        right_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        right_msg.position = right_action
        self.puppet_arm_right_publisher.publish(right_msg)

    def get_latest_data(self):
        """Get latest synchronized data from all sensors"""
        if (len(self.img_left_deque) == 0 or 
            len(self.img_right_deque) == 0 or 
            len(self.img_front_deque) == 0 or
            len(self.puppet_arm_left_deque) == 0 or
            len(self.puppet_arm_right_deque) == 0):
            return None
        
        return (
            self.img_left_deque[-1],
            self.img_right_deque[-1], 
            self.img_front_deque[-1],
            self.puppet_arm_left_deque[-1],
            self.puppet_arm_right_deque[-1]
        )

    def puppet_arm_publish_continuous(self, left, right):
        """Continuous publishing to move robot to target position (from original inference.py)"""
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        
        # Wait for current arm positions
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        
        # Calculate movement direction
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        
        # Step size for each joint (adjust as needed)
        arm_steps_length = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]
        
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            
            # Move left arm
            for i in range(len(left)):
                if left_diff[i] < arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * arm_steps_length[i]
                    flag = True
            
            # Move right arm
            for i in range(len(right)):
                if right_diff[i] < arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * arm_steps_length[i]
                    flag = True
            
            # Publish commands
            self.publish_arm_commands(left_arm, right_arm)
            step += 1
            print(f"Initialization step: {step}")
            rate.sleep()

    def run_inference(self):
        """Main inference loop"""
        print("Starting inference loop...")
        
        # Initialization similar to original inference.py
        print("Performing initialization...")
        
        # Initial positions (from original inference.py)
        left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
        right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
        
        # Move to first position
        self.puppet_arm_publish_continuous(left0, right0)
        input("Enter any key to continue: ")
        
        # Move to second position
        self.puppet_arm_publish_continuous(left1, right1)
        
        print("Initialization complete")
        
        rate = rospy.Rate(self.args.publish_rate)
        
        # Wait for initial data
        print("Waiting for sensor data...")
        while not rospy.is_shutdown():
            data = self.get_latest_data()
            if data is not None:
                break
            rate.sleep()
        
        print("Starting robot control...")
        if self.args.policy_type.lower() == "act":
            print(f"ACT Policy Configuration:")
            print(f"  - chunk_size: {self.policy.config.chunk_size}")
            print(f"  - n_action_steps: {self.policy.config.n_action_steps}")
            print(f"  - temporal_ensemble_coeff: {self.policy.config.temporal_ensemble_coeff}")
        print(f"Stats available: {self.stats is not None}")
        if self.stats:
            print(f"Stats keys: {list(self.stats.keys())}")
        
        # ACT-specific debugging
        if self.args.policy_type.lower() == "act":
            print(f"ACT Policy Details:")
            print(f"  - Has temporal ensembler: {hasattr(self.policy, 'temporal_ensembler')}")
            if hasattr(self.policy, 'temporal_ensembler') and self.policy.temporal_ensembler is not None:
                print(f"  - Temporal ensemble coeff: {self.policy.config.temporal_ensemble_coeff}")
            print(f"  - Action queue length: {len(self.policy._action_queue) if hasattr(self.policy, '_action_queue') else 'N/A'}")
        
        # Initialize temporal aggregation variables
        global inference_thread, inference_lock, inference_actions, inference_timestep
        t = 0
        max_t = 0
        action = None
        
        # Get configuration parameters
        chunk_size = self.policy.config.chunk_size if hasattr(self.policy.config, 'chunk_size') else 100
        action_dim = 14  # 7 joints for left arm + 7 joints for right arm
        
        if self.args.temporal_agg:
            print(f"Temporal aggregation enabled with chunk_size={chunk_size}, action_dim={action_dim}")
            all_time_actions = np.zeros([self.args.max_publish_step, self.args.max_publish_step + chunk_size, action_dim])
        else:
            print("Temporal aggregation disabled")
        
        with torch.inference_mode():
            while t < self.args.max_publish_step and not rospy.is_shutdown():
                # Query policy (similar to original inference.py)
                if self.args.policy_type.lower() == "act":
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=self.inference_process, args=(t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + self.args.pos_lookahead_step
                            if self.args.temporal_agg:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                        inference_lock.release()
                    
                    if self.args.temporal_agg:
                        raw_action = self.apply_temporal_aggregation(all_time_actions, t, chunk_size, action_dim)
                        if raw_action is None:
                            # Fallback to direct action if temporal aggregation fails
                            if self.args.pos_lookahead_step != 0:
                                raw_action = all_actions[:, t % self.args.pos_lookahead_step]
                            else:
                                raw_action = all_actions[:, t % chunk_size]
                    else:
                        if self.args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % self.args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    # For non-ACT policies, use direct inference
                    data = self.get_latest_data()
                    if data is None:
                        rate.sleep()
                        continue
                    
                    img_left, img_right, img_front, puppet_arm_left, puppet_arm_right = data
                    observation = self.preprocess_observation(
                        img_left, img_right, img_front, puppet_arm_left, puppet_arm_right
                    )
                    action = self.policy.select_action(observation)
                    raw_action = action.cpu().numpy()
                    if len(raw_action.shape) == 2:
                        raw_action = raw_action[np.newaxis, :, :]
                
                # Postprocess action
                if len(raw_action.shape) == 3:
                    action = raw_action[0, 0]  # Take first batch and first timestep
                else:
                    action = raw_action[0]
                
                # Convert to torch tensor for postprocessing
                action_tensor = torch.from_numpy(action).float()
                left_action, right_action = self.postprocess_action(action_tensor)
                
                # Debug output
                if t % 10 == 0:
                    print(f"Step {t}: Action shape={action.shape}, Left range=[{min(left_action):.3f}, {max(left_action):.3f}], Right range=[{min(right_action):.3f}, {max(right_action):.3f}]")
                
                # Publish commands
                self.publish_arm_commands(left_action, right_action)
                
                t += 1
                rate.sleep()

    def calculate_optimal_action_scale(self):
        """Calculate optimal action scale based on robot joint limits and training data"""
        if self.stats is None or "action" not in self.stats:
            print("No action stats available, using default scale")
            return self.args.action_scale
        
        action_std = self.stats["action"]["std"].cpu().numpy()
        action_mean = self.stats["action"]["mean"].cpu().numpy()
        
        # Method 1: Based on action standard deviation
        # This represents the typical magnitude of action changes in training data
        std_based_scale = np.mean(np.abs(action_std))
        
        # Method 2: Based on action range
        # This represents the full range of actions in training data
        action_range = np.max(action_mean) - np.min(action_mean)
        range_based_scale = action_range / 2.0  # Use half range as scale
        
        # Method 3: Based on robot joint limits (if known)
        # Typical robot joint limits are around ±π radians (±180 degrees)
        joint_limit_scale = np.pi / 2.0  # Half turn as reasonable movement (more aggressive)
        
        # Choose the most conservative scale
        optimal_scale = min(std_based_scale, range_based_scale, joint_limit_scale)
        
        # Apply additional multiplier for more visible movements
        # This makes the robot movements more noticeable
        movement_multiplier = self.args.movement_multiplier # Use the argument
        
        final_scale = optimal_scale * movement_multiplier
        
        print(f"Action scale calculation:")
        print(f"  - Std-based scale: {std_based_scale:.6f}")
        print(f"  - Range-based scale: {range_based_scale:.6f}")
        print(f"  - Joint-limit scale: {joint_limit_scale:.6f}")
        print(f"  - Optimal scale: {optimal_scale:.6f}")
        print(f"  - Movement multiplier: {movement_multiplier:.1f}x")
        print(f"  - Final scale: {final_scale:.6f}")
        
        return final_scale * self.args.action_scale

    # ROS callbacks
    def img_left_callback(self, msg):
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        self.img_front_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        self.puppet_arm_right_deque.append(msg)


def get_arguments():
    parser = argparse.ArgumentParser(description="LeRobot Inference Fixed for Real Robot Control")
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to policy checkpoint directory')
    parser.add_argument('--policy_type', type=str, default='diffusion', choices=['diffusion', 'act'], 
                       help='Type of policy to use')
    parser.add_argument('--publish_rate', type=int, default=10, help='Control frequency (Hz)')
    parser.add_argument('--chunk_size', type=int, default=None, help='ACT chunk size (overrides model config if specified)')
    parser.add_argument('--n_action_steps', type=int, default=None, help='ACT n_action_steps (overrides model config if specified)')
    parser.add_argument('--action_scale', type=float, default=1.0, help='Scale factor for actions (default: 1.0)')
    parser.add_argument('--movement_multiplier', type=float, default=3.0, help='Multiplier for movement visibility (default: 3.0)')
    parser.add_argument('--temporal_agg', action='store_true', default=True, help='Enable temporal aggregation (default: True)')
    parser.add_argument('--max_publish_step', type=int, default=1000, help='Maximum number of steps to run (default: 1000)')
    parser.add_argument('--pos_lookahead_step', type=int, default=0, help='Position lookahead steps (default: 0)')
    return parser.parse_args()


def main():
    args = get_arguments()
    
    # Create inference object
    inference = LeRobotInferenceFixed(args)
    
    try:
        # Run inference
        inference.run_inference()
    except KeyboardInterrupt:
        print("Inference stopped by user")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down...")


if __name__ == '__main__':
    main() 