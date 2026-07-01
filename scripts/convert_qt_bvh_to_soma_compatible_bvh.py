#!/usr/bin/env python3
"""Adapt a QT-style BVH to the SOMA joint names expected by soma-retargeter.

This is a pragmatic bridge for QT skeleton BVHs that contain a compact 23-joint
human hierarchy. It keeps the official SOMA sample hierarchy as the output
template, copies over matching trunk/limb rotations, and leaves missing face and
finger joints at zero rotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


QT_TO_SOMA_JOINT = {
    "Hips": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Chest": "Spine3",
    "Neck1": "Neck",
    "Neck2": "Neck",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "LeftToeEnd": "LeftToeBase",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
    "RightToeEnd": "RightToeBase",
}


def split_bvh(path: Path) -> tuple[list[str], int, float, list[list[float]]]:
    lines = path.read_text().splitlines()
    motion_idx = lines.index("MOTION")
    frames = int(lines[motion_idx + 1].split(":", 1)[1].strip())
    frame_time = float(lines[motion_idx + 2].split(":", 1)[1].strip())
    rows = [[float(v) for v in line.split()] for line in lines[motion_idx + 3 :] if line.strip()]
    if len(rows) != frames:
        raise ValueError(f"{path} declares {frames} frames but contains {len(rows)} motion rows")
    return lines[:motion_idx], frames, frame_time, rows


def channel_layout(header: list[str]) -> list[tuple[str, list[str]]]:
    current_joint = None
    layout: list[tuple[str, list[str]]] = []
    for line in header:
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            current_joint = stripped.split(maxsplit=1)[1]
        elif stripped.startswith("CHANNELS "):
            if current_joint is None:
                raise ValueError("CHANNELS entry found before ROOT/JOINT")
            parts = stripped.split()
            layout.append((current_joint, parts[2:]))
    return layout


def rows_to_frames(layout: list[tuple[str, list[str]]], rows: list[list[float]]) -> list[dict[str, dict[str, float]]]:
    expected_width = sum(len(channels) for _, channels in layout)
    frames = []
    for row_idx, row in enumerate(rows):
        if len(row) != expected_width:
            raise ValueError(f"frame {row_idx} has {len(row)} values; expected {expected_width}")
        cursor = 0
        frame: dict[str, dict[str, float]] = {}
        for joint, channels in layout:
            frame[joint] = {}
            for channel in channels:
                frame[joint][channel] = row[cursor]
                cursor += 1
        frames.append(frame)
    return frames


def build_output_rows(
    source_layout: list[tuple[str, list[str]]],
    target_layout: list[tuple[str, list[str]]],
    source_rows: list[list[float]],
) -> list[list[float]]:
    source_frames = rows_to_frames(source_layout, source_rows)
    output_rows = []

    for source_frame in source_frames:
        output_row = []
        for target_joint, target_channels in target_layout:
            source_joint = QT_TO_SOMA_JOINT.get(target_joint)
            source_values = source_frame.get(source_joint, {}) if source_joint else {}

            for channel in target_channels:
                if "position" in channel:
                    value = source_values.get(channel, 0.0) if target_joint == "Hips" else 0.0
                elif "rotation" in channel:
                    value = source_values.get(channel, 0.0)
                else:
                    value = 0.0
                output_row.append(value)
        output_rows.append(output_row)

    return output_rows


def write_bvh(template_header: list[str], frames: int, frame_time: float, rows: list[list[float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("\n".join(template_header))
        f.write("\nMOTION\n")
        f.write(f"Frames: {frames}\n")
        f.write(f"Frame Time: {frame_time:.8f}\n")
        for row in rows:
            f.write(" ".join(f"{value:.6f}" for value in row))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="QT BVH input path")
    parser.add_argument("--template", required=True, type=Path, help="SOMA BVH template path")
    parser.add_argument("--output", required=True, type=Path, help="Adapted SOMA-compatible BVH output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_header, frames, frame_time, source_rows = split_bvh(args.input)
    template_header, _, _, _ = split_bvh(args.template)

    source_layout = channel_layout(source_header)
    target_layout = channel_layout(template_header)
    rows = build_output_rows(source_layout, target_layout, source_rows)
    write_bvh(template_header, frames, frame_time, rows, args.output)
    print(f"Wrote {frames} frames to {args.output}")
    print(f"Output channels per frame: {len(rows[0]) if rows else 0}")


if __name__ == "__main__":
    main()
