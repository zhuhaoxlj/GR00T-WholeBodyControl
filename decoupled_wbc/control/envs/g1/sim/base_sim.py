import argparse
from collections import deque
from datetime import datetime
import json
import os
import pathlib
from pathlib import Path
import re
import threading
from threading import Lock, Thread
from typing import Dict

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import yaml

from decoupled_wbc.control.envs.g1.sim.image_publish_utils import ImagePublishProcess
from decoupled_wbc.control.envs.g1.sim.metric_utils import check_contact, check_height
from decoupled_wbc.control.envs.g1.sim.sim_utilts import get_subtree_body_names
from decoupled_wbc.control.envs.g1.sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

DECOUPLED_WBC_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        # global_view is only set up for this specifc scene for now.
        if config["ROBOT_SCENE"] == "decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml":
            camera_configs["global_view"] = {
                "height": 400,
                "width": 400,
            }
        self.config = config
        self.env_name = env_name
        self.num_body_dof = self.config["NUM_JOINTS"]
        self.num_hand_dof = self.config["NUM_HAND_JOINTS"]
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.config["motor_effort_limit_list"])
        self.camera_configs = camera_configs

        # Thread safety lock
        self.reward_lock = Lock()

        # Unitree bridge will be initialized by the simulator
        self.unitree_bridge = None

        # Store display mode
        self.onscreen = onscreen

        # Initialize scene (defined in subclasses)
        self.init_scene()
        self.last_reward = 0
        self.foot_trajectory_enabled = self.config.get("ENABLE_FOOT_TRAJECTORY", True)
        self.foot_trajectory_sample_stride = self.config.get("FOOT_TRAJECTORY_SAMPLE_STRIDE", 4)
        self.foot_trajectory_ground_threshold = self.config.get(
            "FOOT_TRAJECTORY_GROUND_THRESHOLD", 0.025
        )
        self.foot_trajectory_min_distance = self.config.get("FOOT_TRAJECTORY_MIN_DISTANCE", 0.01)
        self.foot_trajectory_z_offset = self.config.get("FOOT_TRAJECTORY_Z_OFFSET", 0.012)
        self.foot_trajectory_counter = 0
        self.foot_trajectory_event_enabled = self.config.get(
            "FOOT_TRAJECTORY_EVENT_ENABLED", True
        )
        self.foot_trajectory_recording_active = not self.foot_trajectory_event_enabled
        self.foot_trajectory_event_file = Path(
            os.environ.get(
                "G1_DANCE_TRAJECTORY_EVENT_FILE",
                self.config.get(
                    "FOOT_TRAJECTORY_EVENT_FILE", "/tmp/g1_dance_trajectory_event.json"
                ),
            )
        )
        self.foot_trajectory_image_dir = Path(
            self.config.get(
                "FOOT_TRAJECTORY_IMAGE_DIR",
                Path(DECOUPLED_WBC_ROOT) / "outputs" / "foot_trajectories",
            )
        )
        self.foot_trajectory_image_dpi = self.config.get("FOOT_TRAJECTORY_IMAGE_DPI", 160)
        self.foot_trajectory_last_event_id = None
        self.foot_trajectory_current_event = {}
        try:
            existing_event = json.loads(self.foot_trajectory_event_file.read_text())
            self.foot_trajectory_last_event_id = (
                existing_event.get("timestamp_ms"),
                existing_event.get("sequence"),
                existing_event.get("event"),
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        self.foot_trajectory = {
            "left": deque(maxlen=self.config.get("FOOT_TRAJECTORY_MAX_POINTS", 20000)),
            "right": deque(maxlen=self.config.get("FOOT_TRAJECTORY_MAX_POINTS", 20000)),
        }
        self.foot_body_ids = {
            "left": mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
            ),
            "right": mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
            ),
        }
        self.foot_local_contact_points = np.array(
            [
                [-0.05, 0.025, -0.03],
                [-0.05, -0.025, -0.03],
                [0.12, 0.03, -0.03],
                [0.12, -0.03, -0.03],
            ],
            dtype=np.float64,
        )

        # Setup offscreen rendering if needed
        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        # Use spawn method for better GIL isolation, or configured method
        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def init_scene(self):
        """Initialize the default robot scene"""
        self.mj_model = mujoco.MjModel.from_xml_path(
            str(pathlib.Path(DECOUPLED_WBC_ROOT) / self.config["ROBOT_SCENE"])
        )
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            # viewer camera
            self.viewer.cam.azimuth = 120  # Horizontal rotation in degrees
            self.viewer.cam.elevation = -30  # Vertical tilt in degrees
            self.viewer.cam.distance = 2.0  # Distance from camera to target
            self.viewer.cam.lookat = np.array([0, 0, 0.5])  # Point the camera is looking at

        # Note that the actuator order is the same as the joint order in the mujoco model.
        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.config["NUM_JOINTS"]
        assert len(self.left_hand_index) == self.config["NUM_HAND_JOINTS"]
        assert len(self.right_hand_index) == self.config["NUM_HAND_JOINTS"]

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

    def init_renderers(self):
        # Initialize camera renderers
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        """Compute body torques based on the current robot state"""
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + 7 - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + 6 - 1]
                        )
                    )
        return body_torques

    def compute_hand_torques(self) -> np.ndarray:
        """Compute hand torques based on the current robot state"""
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + 7 - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + 6 - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + 7 - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + 6 - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        """Compute body joint positions based on the current command"""
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        """Compute hand joint positions based on the current command"""
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        """Prepare observation dictionary from the current robot state"""
        obs = {}
        obs["floating_base_pose"] = self.mj_data.qpos[:7]
        obs["floating_base_vel"] = self.mj_data.qvel[:6]
        obs["floating_base_acc"] = self.mj_data.qacc[:6]
        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]
        obs["secondary_imu_vel"] = self.mj_data.cvel[self.torso_index]
        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + 7 - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + 6 - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + 6 - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + 7 - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + 6 - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + 6 - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                # Get Cartesian pose and velocity of the band_attached_link
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],  # link position in world
                        self.mj_data.xquat[
                            self.band_attached_link
                        ],  # link quaternion in world [w,x,y,z]
                        np.zeros(6),  # placeholder for velocity
                    ]
                )

                # Get velocity in world frame
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,  # 0 for world frame
                )

                # Reorder velocity from [ang, lin] to [lin, ang]
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                # explicitly resetting the force when the band is not enabled
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.handle_foot_trajectory_events()
        self.update_foot_trajectory()
        # self.check_self_collision()

    def kinematics_step(self):
        """
        Run kinematics only: compute the qpos of the robot and directly set the qpos.
        For debugging purposes.
        """
        if self.unitree_bridge is not None:
            self.unitree_bridge.PublishLowState(self.prepare_obs())
            if self.unitree_bridge.joystick:
                self.unitree_bridge.PublishWirelessController()

        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                # Get Cartesian pose and velocity of the band_attached_link
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],  # link position in world
                        self.mj_data.xquat[
                            self.band_attached_link
                        ],  # link quaternion in world [w,x,y,z]
                        np.zeros(6),  # placeholder for velocity
                    ]
                )

                # Get velocity in world frame
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,  # 0 for world frame
                )

                # Reorder velocity from [ang, lin] to [lin, ang]
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()

                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                # explicitly resetting the force when the band is not enabled
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)

        body_qpos = self.compute_body_qpos()  # (num_body_dof,)
        hand_qpos = self.compute_hand_qpos()  # (num_hand_dof * 2,)

        self.mj_data.qpos[self.body_joint_index + 7 - 1] = body_qpos
        self.mj_data.qpos[self.left_hand_index + 7 - 1] = hand_qpos[: self.num_hand_dof]
        self.mj_data.qpos[self.right_hand_index + 7 - 1] = hand_qpos[self.num_hand_dof :]

        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)
        self.handle_foot_trajectory_events()
        self.update_foot_trajectory()

    def get_foot_contact_position(self, side: str):
        body_id = self.foot_body_ids[side]
        if body_id < 0:
            return None

        body_pos = self.mj_data.xpos[body_id]
        body_rot = self.mj_data.xmat[body_id].reshape(3, 3)
        foot_points = body_pos + self.foot_local_contact_points @ body_rot.T
        near_ground_points = foot_points[
            foot_points[:, 2] <= self.foot_trajectory_ground_threshold
        ]
        if len(near_ground_points) == 0:
            return None

        contact_pos = near_ground_points.mean(axis=0)
        contact_pos[2] = self.foot_trajectory_z_offset
        return contact_pos

    def update_foot_trajectory(self):
        if not self.foot_trajectory_enabled:
            return
        if not self.foot_trajectory_recording_active:
            return

        self.foot_trajectory_counter += 1
        if self.foot_trajectory_counter % self.foot_trajectory_sample_stride != 0:
            return

        for side in ("left", "right"):
            contact_pos = self.get_foot_contact_position(side)
            if contact_pos is None:
                continue

            trajectory = self.foot_trajectory[side]
            if (
                len(trajectory) == 0
                or np.linalg.norm(contact_pos[:2] - trajectory[-1][:2])
                >= self.foot_trajectory_min_distance
            ):
                trajectory.append(contact_pos.copy())

    def reset_foot_trajectory(self):
        self.foot_trajectory_counter = 0
        for trajectory in self.foot_trajectory.values():
            trajectory.clear()

    def handle_foot_trajectory_events(self):
        if not self.foot_trajectory_enabled or not self.foot_trajectory_event_enabled:
            return
        try:
            event = json.loads(self.foot_trajectory_event_file.read_text())
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError) as exc:
            if self.config.get("verbose", False):
                print(f"Failed to read foot trajectory event: {exc}")
            return

        event_id = (event.get("timestamp_ms"), event.get("sequence"), event.get("event"))
        if event_id == self.foot_trajectory_last_event_id:
            return

        self.foot_trajectory_last_event_id = event_id
        event_type = event.get("event")
        if event_type == "start":
            self.reset_foot_trajectory()
            self.foot_trajectory_current_event = event
            self.foot_trajectory_recording_active = True
            print(
                "Foot trajectory recording started for "
                f"{event.get('motion_name', 'unknown_motion')}"
            )
        elif event_type == "end":
            if not self.foot_trajectory_recording_active:
                return
            self.update_foot_trajectory()
            saved_path = self.save_foot_trajectory_image(event)
            self.foot_trajectory_recording_active = False
            if saved_path is not None:
                print(f"Foot trajectory image saved: {saved_path}")

    def save_foot_trajectory_image(self, event):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"Cannot save foot trajectory image; matplotlib is unavailable: {exc}")
            return None

        self.foot_trajectory_image_dir.mkdir(parents=True, exist_ok=True)
        motion_name = event.get(
            "motion_name", self.foot_trajectory_current_event.get("motion_name", "unknown_motion")
        )
        motion_index = event.get(
            "motion_index", self.foot_trajectory_current_event.get("motion_index", -1)
        )
        safe_motion_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(motion_name)).strip("_")
        if not safe_motion_name:
            safe_motion_name = "unknown_motion"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            self.foot_trajectory_image_dir
            / f"{timestamp}_motion{motion_index}_{safe_motion_name}_foot_trajectory.png"
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = {"left": "#1f9ed6", "right": "#f26b2e"}
        labels = {"left": "left foot", "right": "right foot"}
        has_points = False

        for side, trajectory in self.foot_trajectory.items():
            if not trajectory:
                continue

            points = np.asarray(list(trajectory), dtype=np.float64)
            has_points = True
            ax.plot(points[:, 0], points[:, 1], color=colors[side], linewidth=1.8, alpha=0.75)
            ax.scatter(points[:, 0], points[:, 1], color=colors[side], s=18, alpha=0.55)
            ax.scatter(
                points[0, 0],
                points[0, 1],
                color=colors[side],
                edgecolors="black",
                marker="o",
                s=95,
                zorder=5,
                label=f"{labels[side]} start",
            )
            ax.scatter(
                points[-1, 0],
                points[-1, 1],
                color=colors[side],
                edgecolors="black",
                marker="X",
                s=115,
                zorder=5,
                label=f"{labels[side]} end",
            )
            ax.annotate(
                f"{side} start",
                xy=(points[0, 0], points[0, 1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )
            ax.annotate(
                f"{side} end",
                xy=(points[-1, 0], points[-1, 1]),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=9,
            )

        title = f"Foot trajectory: {motion_name}"
        if motion_index is not None:
            title += f" (index {motion_index})"
        ax.set_title(title)
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
        if has_points:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "No foot contact trajectory points recorded",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.foot_trajectory_image_dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def draw_foot_trajectory(self):
        if not self.foot_trajectory_enabled or self.viewer is None:
            return

        colors = {
            "left": np.array([0.1, 0.8, 1.0, 0.85], dtype=np.float32),
            "right": np.array([1.0, 0.45, 0.1, 0.85], dtype=np.float32),
        }
        for side, trajectory in self.foot_trajectory.items():
            for point_index, point in enumerate(trajectory):
                geom_index = self.viewer.user_scn.ngeom
                if geom_index >= self.viewer.user_scn.maxgeom:
                    return

                color = colors[side].copy()
                color[3] = 0.25 + 0.6 * (point_index + 1) / max(1, len(trajectory))
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[geom_index],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.018, 0.0, 0.0], dtype=np.float64),
                    pos=point,
                    mat=np.eye(3, dtype=np.float64).flatten(),
                    rgba=color,
                )
                self.viewer.user_scn.ngeom += 1

    def apply_perturbation(self, key):
        """Apply perturbation to the robot"""
        # Add velocity perturbations in body frame
        perturbation_x_body = 0.0  # forward/backward in body frame
        perturbation_y_body = 0.0  # left/right in body frame
        if key == "up":
            perturbation_x_body = 1.0  # forward
        elif key == "down":
            perturbation_x_body = -1.0  # backward
        elif key == "left":
            perturbation_y_body = 1.0  # left
        elif key == "right":
            perturbation_y_body = -1.0  # right

        # Transform body frame velocity to world frame using MuJoCo's rotation
        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]  # [w, x, y, z] quaternion

        # Use MuJoCo's robust quaternion rotation (handles invalid quaternions automatically)
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        # Apply to base linear velocity in world frame
        self.mj_data.qvel[0] += vel_world[0]  # world X velocity
        self.mj_data.qvel[1] += vel_world[1]  # world Y velocity

        # Update dynamics after velocity change
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.user_scn.ngeom = 0
            self.draw_foot_trajectory()
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        """Calculate reward. Should be implemented by subclasses."""
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        """Thread-safe way to get the last calculated reward."""
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        """Set the unitree bridge from the simulator"""
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        """Get privileged observation. Should be implemented by subclasses."""
        return {}

    def update_render_caches(self):
        """Update render cache and shared memory for subprocess."""
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        # Update shared memory if image publishing process is available
        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band is not None:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "r":
            for trajectory in self.foot_trajectory.values():
                trajectory.clear()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        """Check if the robot has fallen"""
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        """Check for self-collision of the robot"""
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for trajectory in self.foot_trajectory.values():
            trajectory.clear()
        self.foot_trajectory_counter = 0


class CubeEnv(DefaultEnv):
    """Environment with a cube object for pick and place tasks"""

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        # Override the robot scene
        config = config.copy()  # Create a copy to avoid modifying the original
        config["ROBOT_SCENE"] = "decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml"
        super().__init__(config, "cube", {}, onscreen, offscreen, enable_image_publish)

    def update_reward(self):
        """Calculate reward based on gripper contact with cube and cube height"""
        right_hand_body = [
            "right_hand_thumb_2_link",
            "right_hand_middle_1_link",
            "right_hand_index_1_link",
        ]
        gripper_cube_contact = check_contact(
            self.mj_model, self.mj_data, right_hand_body, "cube_body"
        )
        cube_lifted = check_height(self.mj_model, self.mj_data, "cube", 0.85, 2.0)

        with self.reward_lock:
            self.last_reward = gripper_cube_contact & cube_lifted


class BoxEnv(DefaultEnv):
    """Environment with a box object for manipulation tasks"""

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        # Override the robot scene
        config = config.copy()  # Create a copy to avoid modifying the original
        config["ROBOT_SCENE"] = "decoupled_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml"
        super().__init__(config, "box", {}, onscreen, offscreen, enable_image_publish)

    def reward(self):
        """Calculate reward based on gripper contact with cube and cube height"""
        left_hand_body = [
            "left_hand_thumb_2_link",
            "left_hand_middle_1_link",
            "left_hand_index_1_link",
        ]
        right_hand_body = [
            "right_hand_thumb_2_link",
            "right_hand_middle_1_link",
            "right_hand_index_1_link",
        ]
        gripper_box_contact = check_contact(self.mj_model, self.mj_data, left_hand_body, "box_body")
        gripper_box_contact &= check_contact(
            self.mj_model, self.mj_data, right_hand_body, "box_body"
        )
        box_lifted = check_height(self.mj_model, self.mj_data, "box", 0.92, 2.0)

        print("gripper_box_contact: ", gripper_box_contact, "box_lifted: ", box_lifted)

        with self.reward_lock:
            self.last_reward = gripper_box_contact & box_lifted
            return self.last_reward


class BottleEnv(DefaultEnv):
    """Environment with a cylinder object for manipulation tasks"""

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        # Override the robot scene
        config = config.copy()  # Create a copy to avoid modifying the original
        config["ROBOT_SCENE"] = "decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml"
        camera_configs = {
            "egoview": {
                "height": 400,
                "width": 400,
            },
        }
        super().__init__(
            config, "cylinder", camera_configs, onscreen, offscreen, enable_image_publish
        )

        self.bottle_body = self.mj_model.body("bottle_body")
        self.bottle_geom = self.mj_model.geom("bottle")

        if self.viewer is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.mj_model.camera("egoview").id

    def update_reward(self):
        """Calculate reward based on gripper contact with cylinder and cylinder height"""
        pass

    def get_privileged_obs(self):
        obs_pos = self.mj_data.xpos[self.bottle_body.id]
        obs_quat = self.mj_data.xquat[self.bottle_body.id]
        return {"bottle_pos": obs_pos, "bottle_quat": obs_quat}


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(self, config: Dict[str, any], env_name: str = "default", **kwargs):
        self.config = config
        self.env_name = env_name

        # Initialize ROS 2 node
        if not rclpy.ok():
            rclpy.init()
            self.node = rclpy.create_node("sim_mujoco")
            self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            self.thread.start()
        else:
            self.thread = None
            executor = rclpy.get_global_executor()
            self.node = executor.get_nodes()[0]  # will only take the first node

        # Create rate objects for different update frequencies
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self.rate = self.node.create_rate(1 / self.sim_dt)

        # Create the appropriate environment based on name
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        elif env_name == "pnp_cube":
            self.sim_env = CubeEnv(config, **kwargs)
        elif env_name == "lift_box":
            self.sim_env = BoxEnv(config, **kwargs)
        elif env_name == "pnp_bottle":
            self.sim_env = BottleEnv(config, **kwargs)
        else:
            raise ValueError(f"Invalid environment name: {env_name}")

        # Initialize the DDS communication layer - should be safe to call multiple times

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            # If it fails because it's already initialized, that's okay
            print(f"Note: Channel factory initialization attempt: {e}")

        # Initialize the unitree bridge and pass it to the environment
        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        # Initialize additional components
        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        # Create simulation thread
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        """Start the image publish subprocess"""
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        """Initialize subscribers. Can be overridden by subclasses."""
        pass

    def init_publisher(self):
        """Initialize publishers. Can be overridden by subclasses."""
        pass

    def init_unitree_bridge(self):
        """Initialize the unitree SDK bridge"""
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0

        try:
            while (
                self.sim_env.viewer and self.sim_env.viewer.is_running()
            ) or self.sim_env.viewer is None:
                # Run simulation step
                self.sim_env.sim_step()

                # Update viewer at viewer rate
                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                # Calculate reward at reward rate
                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                # Update render caches at image rate
                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Sleep to maintain correct rate
                self.rate.sleep()

                sim_cnt += 1
        except rclpy.exceptions.ROSInterruptException:
            # This is expected when ROS shuts down - exit cleanly
            pass
        except Exception:
            self.close()

    def __del__(self):
        """Clean up resources when simulator is deleted"""
        self.close()

    def reset(self):
        """Reset the simulation. Can be overridden by subclasses."""
        self.sim_env.reset()

    def close(self):
        """Close the simulation. Can be overridden by subclasses."""
        try:
            # Stop image publishing subprocess
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()

            # Close viewer
            if hasattr(self.sim_env, "viewer") and self.sim_env.viewer is not None:
                self.sim_env.viewer.close()

            # Shutdown ROS
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        obs = self.sim_env.get_privileged_obs()
        # TODO: add ros2 topic to get privileged obs
        return obs

    def handle_keyboard_button(self, key):
        # Only handles keyboard buttons for default env.
        if self.env_name == "default":
            self.sim_env.handle_keyboard_button(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--config",
        type=str,
        default="./decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml",
        help="config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])

    simulation = BaseSimulator(config)
    simulation.start_as_thread()
