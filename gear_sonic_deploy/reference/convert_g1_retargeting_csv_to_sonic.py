#!/usr/bin/env python3
"""Convert a flat G1 retargeting CSV into a Sonic deploy reference motion.

Input format:
  Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, 29 G1 joint DOFs

The flat CSV uses centimeters for root translation and degrees for all angles.
The output directory matches the deploy CSV bundle consumed by
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


FPS_DEFAULT = 50

BODY_INDEXES = np.array([0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29], dtype=np.int64)

# Source CSV joint columns are in MuJoCo/MJCF order. Sonic deploy reference
# joint_pos.csv is stored in IsaacLab order; deploy code maps it back to MuJoCo.
G1_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)

G1_ISAACLAB_TO_MUJOCO_DOF = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)

G1_MUJOCO_JOINT_COLUMNS = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply for wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[1:]


def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / axis_norm
    half = 0.5 * angle
    return np.array([math.cos(half), *(math.sin(half) * axis)], dtype=np.float64)


def quat_to_angle_axis(q: np.ndarray) -> tuple[float, np.ndarray]:
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    sin_half = np.linalg.norm(q[1:])
    if sin_half < 1e-12:
        return 0.0, np.array([1.0, 0.0, 0.0], dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, q[0])
    return angle, q[1:] / sin_half


@dataclass
class FKNode:
    name: str
    parent: int
    children: list[int]
    axis: np.ndarray
    translation: np.ndarray
    rest_rotation: np.ndarray


class G1FK:
    """Minimal Python port of the deploy-side RobotFK implementation."""

    def __init__(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"Could not find <worldbody> in {xml_path}")
        root_body = worldbody.find("body")
        if root_body is None:
            raise ValueError(f"Could not find root <body> in {xml_path}")

        self.nodes: list[FKNode] = []
        self._add_node(root_body, parent=-1)
        if len(self.nodes) != 30:
            raise ValueError(f"Expected 30 G1 bodies including pelvis, got {len(self.nodes)}")

    def _add_node(self, body: ET.Element, parent: int) -> int:
        node_idx = len(self.nodes)
        name = body.attrib.get("name", "<unnamed>")

        joint = body.find("joint")
        axis = np.zeros(3, dtype=np.float64)
        if joint is not None and "axis" in joint.attrib:
            axis = np.fromstring(joint.attrib["axis"], sep=" ", dtype=np.float64)

        translation = np.zeros(3, dtype=np.float64)
        if "pos" in body.attrib:
            translation = np.fromstring(body.attrib["pos"], sep=" ", dtype=np.float64)

        rest_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if "quat" in body.attrib:
            rest_rotation = np.fromstring(body.attrib["quat"], sep=" ", dtype=np.float64)

        self.nodes.append(
            FKNode(
                name=name,
                parent=parent,
                children=[],
                axis=axis,
                translation=translation,
                rest_rotation=quat_normalize(rest_rotation),
            )
        )
        if parent >= 0:
            self.nodes[parent].children.append(node_idx)

        for child in body.findall("body"):
            self._add_node(child, parent=node_idx)
        return node_idx

    def compute(
        self, root_translation: np.ndarray, root_rotation_wxyz: np.ndarray, joint_angles_isaaclab: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.zeros((len(self.nodes), 3), dtype=np.float64)
        rotations = np.zeros((len(self.nodes), 4), dtype=np.float64)
        positions[0] = root_translation
        rotations[0] = quat_normalize(root_rotation_wxyz)
        self._compute_children(0, positions, rotations, joint_angles_isaaclab)
        return positions, rotations

    def _compute_children(
        self, parent_idx: int, positions: np.ndarray, rotations: np.ndarray, joint_angles_isaaclab: np.ndarray
    ) -> None:
        parent_rot = rotations[parent_idx]
        parent_pos = positions[parent_idx]
        for child_idx in self.nodes[parent_idx].children:
            node = self.nodes[child_idx]
            positions[child_idx] = parent_pos + quat_rotate(parent_rot, node.translation)

            source_joint_idx = G1_ISAACLAB_TO_MUJOCO_DOF[child_idx - 1]
            child_axis = quat_rotate(node.rest_rotation, node.axis)
            child_rot = quat_from_angle_axis(joint_angles_isaaclab[source_joint_idx], child_axis)
            rotations[child_idx] = quat_normalize(quat_mul(parent_rot, quat_mul(child_rot, node.rest_rotation)))

            self._compute_children(child_idx, positions, rotations, joint_angles_isaaclab)


def load_flat_csv(csv_path: Path, start_frame: int, end_frame: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header")

        missing = [
            c
            for c in [
                "root_translateX",
                "root_translateY",
                "root_translateZ",
                "root_rotateX",
                "root_rotateY",
                "root_rotateZ",
                *G1_MUJOCO_JOINT_COLUMNS,
            ]
            if c not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{csv_path} is missing expected columns: {missing}")

        root_pos_cm = []
        root_euler_deg = []
        joint_deg = []
        for row in reader:
            root_pos_cm.append([float(row["root_translateX"]), float(row["root_translateY"]), float(row["root_translateZ"])])
            root_euler_deg.append([float(row["root_rotateX"]), float(row["root_rotateY"]), float(row["root_rotateZ"])])
            joint_deg.append([float(row[col]) for col in G1_MUJOCO_JOINT_COLUMNS])

    root_pos_cm = np.asarray(root_pos_cm, dtype=np.float64)
    root_euler_deg = np.asarray(root_euler_deg, dtype=np.float64)
    joint_deg = np.asarray(joint_deg, dtype=np.float64)

    root_pos_cm = root_pos_cm[start_frame:end_frame]
    root_euler_deg = root_euler_deg[start_frame:end_frame]
    joint_deg = joint_deg[start_frame:end_frame]
    if len(root_pos_cm) < 2:
        raise ValueError("Need at least two frames after slicing")

    root_pos_m = root_pos_cm / 100.0
    root_quat_xyzw = Rotation.from_euler("xyz", root_euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_pos_mujoco = np.deg2rad(joint_deg)
    joint_pos_isaaclab = joint_pos_mujoco[:, G1_MUJOCO_TO_ISAACLAB_DOF]

    return root_pos_m, root_quat_wxyz, joint_pos_isaaclab


def enforce_quat_continuity(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    flat = out.reshape(out.shape[0], -1, 4)
    for body in range(flat.shape[1]):
        for frame in range(1, flat.shape[0]):
            if np.dot(flat[frame - 1, body], flat[frame, body]) < 0:
                flat[frame, body] *= -1.0
    return out


def compute_body_kinematics(
    fk: G1FK, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, joint_pos_isaaclab: np.ndarray, fps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_mujoco_body_indexes = np.array(
        [0 if idx == 0 else G1_MUJOCO_TO_ISAACLAB_DOF[idx - 1] + 1 for idx in BODY_INDEXES], dtype=np.int64
    )

    num_frames = len(joint_pos_isaaclab)
    body_pos = np.zeros((num_frames, len(BODY_INDEXES), 3), dtype=np.float64)
    body_quat = np.zeros((num_frames, len(BODY_INDEXES), 4), dtype=np.float64)

    for frame in range(num_frames):
        full_pos, full_quat = fk.compute(root_pos[frame], root_quat_wxyz[frame], joint_pos_isaaclab[frame])
        body_pos[frame] = full_pos[selected_mujoco_body_indexes]
        body_quat[frame] = full_quat[selected_mujoco_body_indexes]

    body_quat = enforce_quat_continuity(body_quat)

    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros((num_frames, len(BODY_INDEXES), 3), dtype=np.float64)
    for frame in range(num_frames):
        f0 = max(frame - 1, 0)
        f1 = min(num_frames - 1, frame + 1)
        frame_dt = max(1, f1 - f0)
        body_lin_vel[frame] = (body_pos[f1] - body_pos[f0]) * fps / frame_dt

        q1 = body_quat[f1]
        q0 = body_quat[max(0, f1 - 1)]
        for body_idx in range(len(BODY_INDEXES)):
            dq = quat_mul(q1[body_idx], quat_conjugate(q0[body_idx]))
            angle, axis = quat_to_angle_axis(dq)
            body_ang_vel[frame, body_idx] = axis * angle * fps

    body_lin_vel = gaussian_filter1d(body_lin_vel, sigma=2.0, axis=0, mode="nearest")
    body_ang_vel = gaussian_filter1d(body_ang_vel, sigma=2.0, axis=0, mode="nearest")
    return body_pos, body_quat, body_lin_vel, body_ang_vel


def compute_joint_velocity(joint_pos: np.ndarray, fps: int) -> np.ndarray:
    return np.gradient(joint_pos, 1.0 / fps, axis=0, edge_order=1)


def save_array_csv(path: Path, array: np.ndarray, headers: list[str]) -> None:
    flat = array.reshape(array.shape[0], -1)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows([f"{value:.9f}" for value in row] for row in flat)


def write_metadata(path: Path, motion_name: str, num_frames: int) -> None:
    with path.open("w") as f:
        f.write(f"Metadata for: {motion_name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write("[ " + " ".join(str(int(i)) for i in BODY_INDEXES) + "]\n\n")
        f.write(f"Total timesteps: {num_frames}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: ({num_frames}, 29) (float32)\n")
        f.write(f"  joint_vel: ({num_frames}, 29) (float32)\n")
        f.write(f"  body_pos_w: ({num_frames}, 14, 3) (float32)\n")
        f.write(f"  body_quat_w: ({num_frames}, 14, 4) (float32)\n")
        f.write(f"  body_lin_vel_w: ({num_frames}, 14, 3) (float32)\n")
        f.write(f"  body_ang_vel_w: ({num_frames}, 14, 3) (float32)\n")
        f.write("  _body_indexes: (14,) (int64)\n")
        f.write("  time_step_total: () (int64)\n")


def write_info(
    path: Path,
    motion_name: str,
    arrays: dict[str, np.ndarray],
    source_csv: Path,
    fps: int,
    start_frame: int,
    end_frame: int | None,
) -> None:
    with path.open("w") as f:
        f.write(f"Motion Information: {motion_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"source_csv: {source_csv}\n")
        f.write(f"fps: {fps}\n")
        f.write(f"source_slice: [{start_frame}:{'' if end_frame is None else end_frame}]\n\n")
        for key, value in arrays.items():
            f.write(f"{key}:\n")
            f.write(f"  Shape: {value.shape}\n")
            f.write("  Dtype: float32\n")
            f.write(f"  Range: [{value.min():.3f}, {value.max():.3f}]\n")
            f.write(f"  Sample: {value.reshape(-1)[:5]}\n\n")
        f.write("_body_indexes:\n")
        f.write(f"  Shape: {BODY_INDEXES.shape}\n")
        f.write("  Dtype: int64\n")
        f.write(f"  Range: [{BODY_INDEXES.min():.3f}, {BODY_INDEXES.max():.3f}]\n")
        f.write(f"  Sample: {BODY_INDEXES[:5]}\n\n")


def convert(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    xml_path = Path(args.xml)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_pos, root_quat, joint_pos = load_flat_csv(input_csv, args.start_frame, args.end_frame)
    fk = G1FK(xml_path)
    body_pos, body_quat, body_lin_vel, body_ang_vel = compute_body_kinematics(
        fk, root_pos, root_quat, joint_pos, args.fps
    )
    joint_vel = compute_joint_velocity(joint_pos, args.fps)

    arrays = {
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }

    save_array_csv(output_dir / "joint_pos.csv", arrays["joint_pos"], [f"joint_{i}" for i in range(29)])
    save_array_csv(output_dir / "joint_vel.csv", arrays["joint_vel"], [f"joint_vel_{i}" for i in range(29)])
    save_array_csv(
        output_dir / "body_pos.csv",
        arrays["body_pos_w"],
        [f"body_{i}_{axis}" for i in range(14) for axis in "xyz"],
    )
    save_array_csv(
        output_dir / "body_quat.csv",
        arrays["body_quat_w"],
        [f"body_{i}_{axis}" for i in range(14) for axis in "wxyz"],
    )
    save_array_csv(
        output_dir / "body_lin_vel.csv",
        arrays["body_lin_vel_w"],
        [f"body_{i}_vel_{axis}" for i in range(14) for axis in "xyz"],
    )
    save_array_csv(
        output_dir / "body_ang_vel.csv",
        arrays["body_ang_vel_w"],
        [f"body_{i}_angvel_{axis}" for i in range(14) for axis in "xyz"],
    )
    write_metadata(output_dir / "metadata.txt", args.motion_name, len(joint_pos))
    write_info(output_dir / "info.txt", args.motion_name, arrays, input_csv, args.fps, args.start_frame, args.end_frame)

    print(f"Converted {len(joint_pos)} frames")
    print(f"Wrote Sonic reference motion: {output_dir}")
    print("Files: joint_pos.csv, joint_vel.csv, body_pos.csv, body_quat.csv, body_lin_vel.csv, body_ang_vel.csv")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(repo_root / "qt_dance_data" / "Take_017_G1_Retargeting.csv"),
        help="Flat G1 retargeting CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "gear_sonic_deploy" / "reference" / "self" / "qt_take_017"),
        help="Output Sonic deploy motion directory",
    )
    parser.add_argument(
        "--xml",
        default=str(repo_root / "gear_sonic_deploy" / "g1" / "g1_29dof.xml"),
        help="G1 MuJoCo XML used for FK",
    )
    parser.add_argument("--motion-name", default="qt_take_017", help="Motion name written to metadata/info")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT, help="Motion frame rate")
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive Python slice start frame")
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive Python slice end frame. Use -1 to drop the last frame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    convert(parse_args())
