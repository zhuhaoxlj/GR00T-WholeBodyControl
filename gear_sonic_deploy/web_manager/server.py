from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_MOTION_DIR = PROJECT_ROOT / "reference" / "example"
MOTION_DIR = Path(os.environ.get("SONIC_MOTION_DIR", DEFAULT_MOTION_DIR)).resolve()
SONIC_STATUS_FILE = Path(
    os.environ.get("SONIC_STATUS_FILE", MOTION_DIR / ".sonic_status.json")
).resolve()
SONIC_ZMQ_URL = os.environ.get("SONIC_ZMQ_URL", "tcp://127.0.0.1:5557")
SONIC_ZMQ_TOPIC = os.environ.get("SONIC_ZMQ_TOPIC", "g1_debug")
SONIC_STATUS_STALE_SECONDS = float(os.environ.get("SONIC_STATUS_STALE_SECONDS", "3.0"))
SIM_MODEL_ROOT = Path(
    os.environ.get(
        "SONIC_SIM_MODEL_ROOT",
        REPO_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1",
    )
).resolve()
SIM_SCENE_RELATIVE_PATH = os.environ.get("SONIC_SIM_SCENE", "scene_43dof.xml")
SIM_WBC_CONFIG_PATH = (
    REPO_ROOT
    / "decoupled_wbc"
    / "sim2mujoco"
    / "resources"
    / "robots"
    / "g1"
    / "g1_gear_wbc.yaml"
).resolve()
SIM_WBC_RESOURCE_ROOT = (
    REPO_ROOT / "decoupled_wbc" / "sim2mujoco" / "resources" / "robots" / "g1"
).resolve()
SIM_WBC_POLICY_FILES = {
    "balance": "policy/GR00T-WholeBodyControl-Balance.onnx",
    "walk": "policy/GR00T-WholeBodyControl-Walk.onnx",
}
FRONTEND_DIST_DIR = APP_DIR / "frontend" / "dist"
FRONTEND_DIST_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

REQUIRED_FILES = {
    "joint_pos.csv",
    "joint_vel.csv",
    "body_pos.csv",
    "body_quat.csv",
    "body_lin_vel.csv",
    "body_ang_vel.csv",
}
OPTIONAL_FILES = {"metadata.txt", "info.txt", "smpl_joint.csv", "smpl_pose.csv"}
ALLOWED_FILES = REQUIRED_FILES | OPTIONAL_FILES
EXPECTED_COLUMNS = {
    "joint_pos.csv": 29,
    "joint_vel.csv": 29,
    "body_pos.csv": 42,
    "body_quat.csv": 56,
    "body_lin_vel.csv": 42,
    "body_ang_vel.csv": 42,
}
MOTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MOTION_ALIAS_FILE = MOTION_DIR / ".motion_aliases.json"
MOTION_GROUPS_FILE = MOTION_DIR / ".motion_groups.json"
MOTION_PLAYBACK_REQUEST_FILE = MOTION_DIR / ".motion_playback_request.json"
MOTION_ALIAS_MAX_LENGTH = 80
MOTION_GROUP_NAME_MAX_LENGTH = 80
PRESET_MOTION_NAMES = {
    "dance_in_da_party_001__A464",
    "dance_in_da_party_001__A464_M",
    "forward_lunge_R_001__A359_M",
    "macarena_001__A545",
    "macarena_001__A545_M",
    "neutral_kick_R_001__A543",
    "neutral_kick_R_001__A543_M",
    "squat_001__A359",
    "tired_forward_lunge_R_001__A359_M",
    "tired_one_leg_jumping_R_001__A359",
    "tired_one_leg_jumping_R_001__A359_M",
    "walking_quip_360_R_002__A428",
    "walking_quip_360_R_002__A428_M",
}

PLANNER_MODE_GROUPS = [
    {"name": "standing", "display_name": "站立/行走", "actions": [
        "SLOW_WALK", "WALK", "RUN", "FORWARD_JUMP", "STEALTH_WALK", "INJURED_WALK",
    ]},
    {"name": "squat", "display_name": "下蹲/爬行", "actions": [
        "IDEL_SQUAT", "IDEL_KNEEL_TWO_LEGS", "IDEL_KNEEL", "CRAWLING", "ELBOW_CRAWLING",
    ]},
    {"name": "boxing", "display_name": "拳击", "actions": [
        "IDEL_BOXING", "WALK_BOXING", "LEFT_PUNCH", "RIGHT_PUNCH", "RANDOM_PUNCH",
        "LEFT_HOOK", "RIGHT_HOOK",
    ]},
    {"name": "styled_walking", "display_name": "风格化行走", "actions": [
        "LEDGE_WALKING", "OBJECT_CARRYING", "STEALTH_WALK_2", "HAPPY_DANCE_WALK",
        "ZOMBIE_WALK", "GUN_WALK", "SCARE_WALK",
    ]},
]

LOCOMOTION_MODE_NAMES = {
    0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN", 4: "IDEL_SQUAT",
    5: "IDEL_KNEEL_TWO_LEGS", 6: "IDEL_KNEEL", 7: "IDEL_LYING_FACE_DOWN",
    8: "CRAWLING", 9: "IDEL_BOXING", 10: "WALK_BOXING", 11: "LEFT_PUNCH",
    12: "RIGHT_PUNCH", 13: "RANDOM_PUNCH", 14: "ELBOW_CRAWLING",
    15: "LEFT_HOOK", 16: "RIGHT_HOOK", 17: "FORWARD_JUMP", 18: "STEALTH_WALK",
    19: "INJURED_WALK", 20: "LEDGE_WALKING", 21: "OBJECT_CARRYING",
    22: "STEALTH_WALK_2", 23: "HAPPY_DANCE_WALK", 24: "ZOMBIE_WALK",
    25: "GUN_WALK", 26: "SCARE_WALK",
}

_status_lock = threading.Lock()
_realtime_status: dict[str, Any] = {}
_status_thread_started = False

app = FastAPI(title="Sonic Motion Web Manager")
static_dir = APP_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if FRONTEND_DIST_ASSETS_DIR.exists():
    app.mount(
        "/sim/assets",
        StaticFiles(directory=FRONTEND_DIST_ASSETS_DIR),
        name="sim-assets",
    )


def ensure_motion_dir() -> None:
    MOTION_DIR.mkdir(parents=True, exist_ok=True)
    (MOTION_DIR / ".upload_tmp").mkdir(exist_ok=True)
    (MOTION_DIR / ".trash").mkdir(exist_ok=True)


def safe_motion_name(name: str) -> str:
    cleaned = name.strip().strip("/\\")
    if not MOTION_NAME_RE.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail=f"Invalid motion name: {name!r}")
    return cleaned


def clean_motion_alias(alias: str | None) -> str | None:
    if alias is None:
        return None
    cleaned = " ".join(alias.strip().split())
    if not cleaned:
        return None
    if len(cleaned) > MOTION_ALIAS_MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Motion alias must be <= {MOTION_ALIAS_MAX_LENGTH} characters",
        )
    return cleaned


def clean_motion_group_name(name: str | None) -> str:
    cleaned = " ".join((name or "").strip().split())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Motion group name is required")
    if len(cleaned) > MOTION_GROUP_NAME_MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Motion group name must be <= {MOTION_GROUP_NAME_MAX_LENGTH} characters",
        )
    return cleaned


def read_motion_aliases() -> dict[str, str]:
    try:
        data = json.loads(MOTION_ALIAS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if isinstance(value, str)}


def write_motion_aliases(aliases: dict[str, str]) -> None:
    ensure_motion_dir()
    MOTION_ALIAS_FILE.write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def set_motion_alias(motion_name: str, alias: str | None) -> None:
    aliases = read_motion_aliases()
    if alias:
        aliases[motion_name] = alias
    else:
        aliases.pop(motion_name, None)
    write_motion_aliases(aliases)


def read_motion_groups() -> list[dict[str, Any]]:
    try:
        data = json.loads(MOTION_GROUPS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    raw_groups = data.get("groups") if isinstance(data, dict) else data
    if not isinstance(raw_groups, list):
        return []
    groups: list[dict[str, Any]] = []
    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id") or "").strip()
        name = str(group.get("name") or "").strip()
        motion_names = group.get("motion_names")
        if not group_id or not name or not isinstance(motion_names, list):
            continue
        cleaned_motion_names: list[str] = []
        for motion_name in motion_names:
            try:
                cleaned_motion_names.append(safe_motion_name(str(motion_name)))
            except HTTPException:
                continue
        groups.append({
            "id": group_id,
            "name": name,
            "motion_names": cleaned_motion_names,
            "created_at": group.get("created_at"),
            "updated_at": group.get("updated_at"),
        })
    return groups


def write_motion_groups(groups: list[dict[str, Any]]) -> None:
    ensure_motion_dir()
    MOTION_GROUPS_FILE.write_text(
        json.dumps({"groups": groups}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def motion_exists(motion_name: str) -> bool:
    path = assert_under_motion_dir(MOTION_DIR / motion_name)
    return path.exists() and path.is_dir() and not path.is_symlink()


def validate_motion_names(motion_names: list[Any]) -> list[str]:
    if not isinstance(motion_names, list):
        raise HTTPException(status_code=400, detail="motion_names must be a list")
    cleaned: list[str] = []
    for raw_name in motion_names:
        motion_name = safe_motion_name(str(raw_name))
        if not motion_exists(motion_name):
            raise HTTPException(status_code=400, detail=f"Motion not found: {motion_name}")
        cleaned.append(motion_name)
    return cleaned


def group_with_motion_details(group: dict[str, Any], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    aliases = aliases if aliases is not None else read_motion_aliases()
    motions = []
    valid_motion_names = []
    for motion_name in group["motion_names"]:
        if not motion_exists(motion_name):
            continue
        motion_path = MOTION_DIR / motion_name
        validation = validate_motion_dir(motion_path, strict=False)
        motions.append({
            "name": motion_name,
            "alias": aliases.get(motion_name),
            "display_name": aliases.get(motion_name) or motion_name,
            "valid": validation["valid"],
            "errors": validation["errors"],
            "timesteps": validation["metadata_timesteps"] or next(iter(validation["rows"].values()), None),
        })
        valid_motion_names.append(motion_name)
    return {
        **group,
        "motion_names": valid_motion_names,
        "motions": motions,
    }


def find_motion_group(group_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = read_motion_groups()
    for group in groups:
        if group["id"] == group_id:
            return groups, group
    raise HTTPException(status_code=404, detail="Motion group not found")


def remove_motion_from_groups(motion_name: str) -> None:
    groups = read_motion_groups()
    changed = False
    timestamp = datetime.now().isoformat(timespec="seconds")
    for group in groups:
        next_motion_names = [name for name in group["motion_names"] if name != motion_name]
        if len(next_motion_names) != len(group["motion_names"]):
            group["motion_names"] = next_motion_names
            group["updated_at"] = timestamp
            changed = True
    if changed:
        write_motion_groups(groups)


def write_motion_playback_request(group: dict[str, Any]) -> dict[str, Any]:
    if not group["motion_names"]:
        raise HTTPException(status_code=400, detail="Motion group is empty")
    motion_names = validate_motion_names(group["motion_names"])
    request_id = uuid.uuid4().hex
    payload = {
        "request_id": request_id,
        "type": "motion_group",
        "group_id": group["id"],
        "group_name": group["name"],
        "motion_names": motion_names,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    ensure_motion_dir()
    MOTION_PLAYBACK_REQUEST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return payload


def is_user_uploaded_motion(motion_name: str) -> bool:
    return motion_name not in PRESET_MOTION_NAMES


def assert_under_motion_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != MOTION_DIR and MOTION_DIR not in resolved.parents:
        raise HTTPException(status_code=400, detail="Path escapes motion directory")
    return resolved


def iter_motion_dirs() -> Iterable[Path]:
    ensure_motion_dir()
    for child in sorted(MOTION_DIR.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith(".") or not child.is_dir() or child.is_symlink():
            continue
        yield child


def parse_metadata(path: Path) -> tuple[int | None, list[int]]:
    total_timesteps = None
    body_indexes: list[int] = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None, []
    for index, line in enumerate(lines):
        if "Total timesteps:" in line:
            match = re.search(r"Total timesteps:\s*(\d+)", line)
            if match:
                total_timesteps = int(match.group(1))
        if "Body part indexes:" in line and index + 1 < len(lines):
            body_indexes = [int(v) for v in re.findall(r"\d+", lines[index + 1])]
    return total_timesteps, body_indexes


def validate_csv(path: Path, expected_cols: int | None) -> tuple[int, int]:
    rows = 0
    cols_seen = None
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path.name} is empty") from exc
        for row_index, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                raise ValueError(f"{path.name}:{row_index} is empty")
            if cols_seen is None:
                cols_seen = len(row)
                if expected_cols is not None and cols_seen != expected_cols:
                    raise ValueError(f"{path.name} expected {expected_cols} columns, got {cols_seen}")
            elif len(row) != cols_seen:
                raise ValueError(f"{path.name}:{row_index} column count changed")
            for cell in row:
                value = float(cell)
                if not math.isfinite(value):
                    raise ValueError(f"{path.name}:{row_index} contains NaN/Inf")
            rows += 1
    if rows <= 0:
        raise ValueError(f"{path.name} has no data rows")
    return rows, cols_seen or 0


def validate_motion_dir(path: Path, strict: bool = True) -> dict:
    files = {p.name for p in path.iterdir() if p.is_file() and not p.is_symlink()}
    missing = sorted(REQUIRED_FILES - files)
    unknown = sorted(files - ALLOWED_FILES)
    errors: list[str] = []
    if missing:
        errors.append("Missing required files: " + ", ".join(missing))
    if strict and unknown:
        errors.append("Unknown files: " + ", ".join(unknown))
    row_counts: dict[str, int] = {}
    columns: dict[str, int] = {}
    if not missing:
        for filename in sorted((REQUIRED_FILES | (files & OPTIONAL_FILES)) & files):
            if not filename.endswith(".csv"):
                continue
            try:
                rows, cols = validate_csv(path / filename, EXPECTED_COLUMNS.get(filename))
                row_counts[filename] = rows
                columns[filename] = cols
            except Exception as exc:
                errors.append(str(exc))
        if row_counts and len(set(row_counts.values())) != 1:
            errors.append("CSV row counts are inconsistent")
    total_timesteps, body_indexes = parse_metadata(path / "metadata.txt")
    if body_indexes and len(body_indexes) != 14:
        errors.append(f"metadata Body part indexes expected 14 values, got {len(body_indexes)}")
    if row_counts:
        csv_timesteps = next(iter(row_counts.values()))
        if total_timesteps is not None and total_timesteps != csv_timesteps:
            errors.append(f"metadata Total timesteps {total_timesteps} != CSV rows {csv_timesteps}")
    return {
        "valid": not errors,
        "errors": errors,
        "missing": missing,
        "unknown": unknown,
        "rows": row_counts,
        "columns": columns,
        "metadata_timesteps": total_timesteps,
        "body_indexes": body_indexes,
    }


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def assert_under_sim_model_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != SIM_MODEL_ROOT and SIM_MODEL_ROOT not in resolved.parents:
        raise HTTPException(status_code=400, detail="Path escapes simulation model root")
    return resolved


def assert_under_wbc_resource_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != SIM_WBC_RESOURCE_ROOT and SIM_WBC_RESOURCE_ROOT not in resolved.parents:
        raise HTTPException(status_code=400, detail="Path escapes WBC resource root")
    return resolved


def sim_asset_url(relative_path: str) -> str:
    return "/api/sim/assets/" + relative_path.replace("\\", "/")


def wbc_asset_url(relative_path: str) -> str:
    return "/api/sim/wbc/assets/" + relative_path.replace("\\", "/")


def relative_to_sim_model_root(path: Path) -> str:
    return path.resolve().relative_to(SIM_MODEL_ROOT).as_posix()


def relative_to_wbc_resource_root(path: Path) -> str:
    return path.resolve().relative_to(SIM_WBC_RESOURCE_ROOT).as_posix()


def iter_sim_asset_files() -> Iterable[Path]:
    if not SIM_MODEL_ROOT.exists():
        return []
    return (
        path
        for path in sorted(SIM_MODEL_ROOT.rglob("*"), key=lambda candidate: candidate.as_posix())
        if path.is_file() and not path.is_symlink()
    )


def read_xml_asset_references(xml_path: Path) -> dict[str, Any]:
    try:
        xml_text = xml_path.read_text(errors="replace")
    except OSError:
        return {"includes": [], "meshes": [], "textures": [], "meshdir": None}

    include_files = sorted(set(re.findall(r"<include\s+[^>]*file=[\"']([^\"']+)[\"']", xml_text)))
    mesh_files = sorted(set(re.findall(r"<mesh\s+[^>]*file=[\"']([^\"']+)[\"']", xml_text)))
    texture_files = sorted(set(re.findall(r"<texture\s+[^>]*file=[\"']([^\"']+)[\"']", xml_text)))
    meshdir_match = re.search(r"<compiler\s+[^>]*meshdir=[\"']([^\"']+)[\"']", xml_text)
    return {
        "includes": include_files,
        "meshes": mesh_files,
        "textures": texture_files,
        "meshdir": meshdir_match.group(1) if meshdir_match else None,
    }


def build_sim_asset_manifest() -> dict[str, Any]:
    scene_path = assert_under_sim_model_root(SIM_MODEL_ROOT / SIM_SCENE_RELATIVE_PATH)
    files = []
    for file_path in iter_sim_asset_files():
        relative_path = relative_to_sim_model_root(file_path)
        files.append({
            "path": relative_path,
            "url": sim_asset_url(relative_path),
            "size_bytes": file_path.stat().st_size,
        })

    scene_references = read_xml_asset_references(scene_path)
    include_references = []
    for include_file in scene_references["includes"]:
        include_path = assert_under_sim_model_root(scene_path.parent / include_file)
        include_references.append({
            "path": relative_to_sim_model_root(include_path),
            "url": sim_asset_url(relative_to_sim_model_root(include_path)),
            "references": read_xml_asset_references(include_path),
        })

    return {
        "model_root": str(SIM_MODEL_ROOT),
        "scene_path": relative_to_sim_model_root(scene_path),
        "scene_url": sim_asset_url(relative_to_sim_model_root(scene_path)),
        "wbc_config_path": str(SIM_WBC_CONFIG_PATH),
        "scene_references": scene_references,
        "include_references": include_references,
        "files": files,
    }


def load_wbc_policy_config() -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="PyYAML is required for WBC config loading") from exc

    if not SIM_WBC_CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="WBC config not found")
    raw_config = yaml.safe_load(SIM_WBC_CONFIG_PATH.read_text())
    if not isinstance(raw_config, dict):
        raise HTTPException(status_code=500, detail="WBC config is invalid")

    def float_list(key: str) -> list[float]:
        value = raw_config.get(key)
        if not isinstance(value, list):
            raise HTTPException(status_code=500, detail=f"WBC config missing list: {key}")
        return [float(item) for item in value]

    def float_value(key: str) -> float:
        value = raw_config.get(key)
        if not isinstance(value, (int, float)):
            raise HTTPException(status_code=500, detail=f"WBC config missing number: {key}")
        return float(value)

    def int_value(key: str) -> int:
        value = raw_config.get(key)
        if not isinstance(value, int):
            raise HTTPException(status_code=500, detail=f"WBC config missing integer: {key}")
        return value

    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
        "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    default_angles = float_list("default_angles")
    arm_default_angles = [0.0] * (len(joint_names) - len(default_angles))
    arm_kps = [100.0] * len(arm_default_angles)
    arm_kds = [0.5] * len(arm_default_angles)

    return {
        "config_path": str(SIM_WBC_CONFIG_PATH),
        "resource_root": str(SIM_WBC_RESOURCE_ROOT),
        "simulation_dt": float_value("simulation_dt"),
        "control_decimation": int_value("control_decimation"),
        "num_actions": int_value("num_actions"),
        "num_obs": int_value("num_obs"),
        "obs_history_len": int_value("obs_history_len"),
        "kps": float_list("kps") + arm_kps,
        "kds": float_list("kds") + arm_kds,
        "default_angles": default_angles + arm_default_angles,
        "action_scale": float_value("action_scale"),
        "ang_vel_scale": float_value("ang_vel_scale"),
        "dof_pos_scale": float_value("dof_pos_scale"),
        "dof_vel_scale": float_value("dof_vel_scale"),
        "cmd_scale": float_list("cmd_scale"),
        "cmd_init": float_list("cmd_init"),
        "height_cmd": float_value("height_cmd"),
        "rpy_cmd": float_list("rpy_cmd"),
        "freq_cmd": float_value("freq_cmd"),
        "joint_names": joint_names,
    }


def build_wbc_policy_manifest() -> dict[str, Any]:
    policies = {}
    for policy_name, relative_path in SIM_WBC_POLICY_FILES.items():
        policy_path = assert_under_wbc_resource_root(SIM_WBC_RESOURCE_ROOT / relative_path)
        if not policy_path.exists() or not policy_path.is_file():
            raise HTTPException(status_code=404, detail=f"WBC policy not found: {policy_name}")
        policies[policy_name] = {
            "path": relative_to_wbc_resource_root(policy_path),
            "url": wbc_asset_url(relative_to_wbc_resource_root(policy_path)),
            "size_bytes": policy_path.stat().st_size,
        }
    return {
        "config": load_wbc_policy_config(),
        "policies": policies,
    }


def find_motion_root(path: Path) -> Path:
    if REQUIRED_FILES <= {p.name for p in path.iterdir() if p.is_file()}:
        return path
    candidates = [
        p for p in path.iterdir()
        if p.is_dir() and REQUIRED_FILES <= {f.name for f in p.iterdir() if f.is_file()}
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise HTTPException(status_code=400, detail="Upload must contain one complete motion directory")


def motion_name_from_upload(filename: str) -> str:
    name = Path(filename).name
    if name.lower().endswith(".zip"):
        name = Path(name).stem
    for suffix in ("_sonic_reference", "-sonic-reference", "_reference", "-reference"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return safe_motion_name(name)


def touch_reload_flag() -> None:
    ensure_motion_dir()
    (MOTION_DIR / ".motion_reload_request").write_text(str(time.time()))


def planner_action_group(action_name: str | None) -> str | None:
    if not action_name:
        return None
    for group in PLANNER_MODE_GROUPS:
        if action_name in group["actions"]:
            return group["name"]
    return None


def normalize_sonic_status(raw: dict[str, Any] | None, source: str, stamp: float | None = None) -> dict[str, Any]:
    now = time.time()
    raw = raw or {}
    updated_at = float(stamp or raw.get("updated_at") or now)
    planner_enabled = bool(raw.get("planner_enabled", False))
    planner_initialized = bool(raw.get("planner_initialized", False))
    locomotion_mode = raw.get("locomotion_mode")
    action_name = raw.get("locomotion_mode_name")
    if action_name is None and isinstance(locomotion_mode, int):
        action_name = LOCOMOTION_MODE_NAMES.get(locomotion_mode, f"UNKNOWN_{locomotion_mode}")

    sonic_mode = raw.get("sonic_mode")
    if not sonic_mode:
        sonic_mode = "planner" if planner_enabled else "motion"
    current_motion = raw.get("current_motion") or raw.get("motion_name")
    current_action = action_name if sonic_mode == "planner" else current_motion

    return {
        "connected": now - updated_at <= SONIC_STATUS_STALE_SECONDS,
        "source": source,
        "updated_at": updated_at,
        "updated_at_iso": datetime.fromtimestamp(updated_at).isoformat(timespec="seconds"),
        "stale_seconds": round(max(0.0, now - updated_at), 3),
        "program_state": raw.get("program_state"),
        "sonic_mode": sonic_mode,
        "planner_enabled": planner_enabled,
        "planner_initialized": planner_initialized,
        "motion": {
            "current": current_motion,
            "frame": raw.get("current_frame"),
            "actions": [path.name for path in iter_motion_dirs()],
        },
        "planner": {
            "current_action": action_name,
            "current_group": planner_action_group(action_name),
            "locomotion_mode": locomotion_mode,
            "speed": raw.get("movement_speed"),
            "height": raw.get("height"),
            "groups": PLANNER_MODE_GROUPS,
        },
        "playback_group": raw.get("playback_group") if isinstance(raw.get("playback_group"), dict) else None,
        "current_action": current_action,
    }


def read_status_file() -> dict[str, Any] | None:
    try:
        stat = SONIC_STATUS_FILE.stat()
        data = json.loads(SONIC_STATUS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return normalize_sonic_status(data, "file", stat.st_mtime)


def update_realtime_status(data: dict[str, Any]) -> None:
    with _status_lock:
        _realtime_status.clear()
        _realtime_status.update(data)


def decode_zmq_status(payload: bytes) -> dict[str, Any] | None:
    try:
        import msgpack  # type: ignore
    except ImportError:
        return None
    try:
        data = msgpack.unpackb(payload, raw=False)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    keys = {
        "program_state", "sonic_mode", "planner_enabled", "planner_initialized",
        "locomotion_mode", "locomotion_mode_name", "movement_speed", "height",
        "current_motion", "current_frame",
    }
    if not keys.intersection(data):
        return None
    return normalize_sonic_status(data, "zmq")


def status_subscriber_loop() -> None:
    try:
        import zmq  # type: ignore
    except ImportError:
        return
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, SONIC_ZMQ_TOPIC)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(SONIC_ZMQ_URL)
        prefix_len = len(SONIC_ZMQ_TOPIC.encode())
        while True:
            try:
                message = socket.recv()
            except zmq.Again:
                continue
            status = decode_zmq_status(message[prefix_len:])
            if status:
                update_realtime_status(status)
    finally:
        socket.close(0)


@app.on_event("startup")
def start_status_subscriber() -> None:
    global _status_thread_started
    if _status_thread_started:
        return
    _status_thread_started = True
    thread = threading.Thread(target=status_subscriber_loop, name="sonic-status-zmq", daemon=True)
    thread.start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/sim")
@app.get("/sim/")
def sim_index() -> FileResponse:
    sim_index_path = FRONTEND_DIST_DIR / "index.html"
    if not sim_index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Web MuJoCo frontend has not been built yet. Run `npm install` "
                "and `npm run build` in web_manager/frontend, or use the Vite dev server."
            ),
        )
    return FileResponse(sim_index_path)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "motion_dir": str(MOTION_DIR)}


@app.get("/api/sim/config")
def sim_config() -> dict:
    manifest = build_sim_asset_manifest()
    return {
        "ok": True,
        "simulator": "mujoco-wasm",
        "model_root": manifest["model_root"],
        "scene_path": manifest["scene_path"],
        "scene_url": manifest["scene_url"],
        "manifest_url": "/api/sim/assets/manifest",
        "wbc_config_path": manifest["wbc_config_path"],
        "default_timestep": 0.005,
        "notes": [
            "The browser simulator loads MuJoCo WASM and runs the G1 model directly in the page.",
            "WBC policy assets are exposed via /api/sim/wbc/manifest for in-browser ONNX inference.",
        ],
    }


@app.get("/api/sim/assets/manifest")
def sim_assets_manifest() -> dict:
    return {"ok": True, "manifest": build_sim_asset_manifest()}


@app.get("/api/sim/wbc/manifest")
def sim_wbc_manifest() -> dict:
    return {"ok": True, "manifest": build_wbc_policy_manifest()}


@app.get("/api/sim/wbc/assets/{asset_path:path}")
def sim_wbc_asset(asset_path: str) -> FileResponse:
    safe_asset_path = asset_path.strip().lstrip("/\\")
    if not safe_asset_path:
        raise HTTPException(status_code=400, detail="WBC asset path is required")
    resolved_asset_path = assert_under_wbc_resource_root(SIM_WBC_RESOURCE_ROOT / safe_asset_path)
    if not resolved_asset_path.exists() or not resolved_asset_path.is_file():
        raise HTTPException(status_code=404, detail="WBC asset not found")
    return FileResponse(resolved_asset_path)


@app.get("/api/sim/assets/{asset_path:path}")
def sim_asset(asset_path: str) -> FileResponse:
    safe_asset_path = asset_path.strip().lstrip("/\\")
    if not safe_asset_path:
        raise HTTPException(status_code=400, detail="Simulation asset path is required")
    resolved_asset_path = assert_under_sim_model_root(SIM_MODEL_ROOT / safe_asset_path)
    if not resolved_asset_path.exists() or not resolved_asset_path.is_file():
        raise HTTPException(status_code=404, detail="Simulation asset not found")
    return FileResponse(resolved_asset_path)


@app.get("/api/sonic/status")
def sonic_status() -> dict:
    with _status_lock:
        realtime_status = dict(_realtime_status)
    if realtime_status:
        now = time.time()
        updated_at = float(realtime_status.get("updated_at") or now)
        realtime_status["connected"] = now - updated_at <= SONIC_STATUS_STALE_SECONDS
        realtime_status["stale_seconds"] = round(max(0.0, now - updated_at), 3)
        return realtime_status

    file_status = read_status_file()
    if file_status:
        return file_status

    return normalize_sonic_status(None, "default")


@app.get("/api/motions")
def list_motions() -> dict:
    motions = []
    aliases = read_motion_aliases()
    for path in iter_motion_dirs():
        validation = validate_motion_dir(path, strict=False)
        stat = path.stat()
        motions.append({
            "name": path.name,
            "alias": aliases.get(path.name),
            "display_name": aliases.get(path.name) or path.name,
            "alias_editable": is_user_uploaded_motion(path.name),
            "valid": validation["valid"],
            "errors": validation["errors"],
            "timesteps": validation["metadata_timesteps"] or next(iter(validation["rows"].values()), None),
            "size_bytes": directory_size(path),
            "mtime": stat.st_mtime,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    return {"motion_dir": str(MOTION_DIR), "motions": motions}


@app.get("/api/motion-groups")
def list_motion_groups() -> dict:
    aliases = read_motion_aliases()
    groups = [group_with_motion_details(group, aliases) for group in read_motion_groups()]
    return {"groups": groups}


@app.post("/api/motion-groups")
def create_motion_group(payload: dict[str, Any] = Body(default_factory=dict)) -> dict:
    groups = read_motion_groups()
    timestamp = datetime.now().isoformat(timespec="seconds")
    group = {
        "id": uuid.uuid4().hex,
        "name": clean_motion_group_name(payload.get("name")),
        "motion_names": validate_motion_names(payload.get("motion_names", [])),
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    groups.append(group)
    write_motion_groups(groups)
    return {"ok": True, "group": group_with_motion_details(group)}


@app.patch("/api/motion-groups/{group_id}")
def update_motion_group(group_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict:
    groups, group = find_motion_group(group_id)
    if "name" in payload:
        group["name"] = clean_motion_group_name(payload.get("name"))
    if "motion_names" in payload:
        group["motion_names"] = validate_motion_names(payload.get("motion_names", []))
    group["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_motion_groups(groups)
    return {"ok": True, "group": group_with_motion_details(group)}


@app.delete("/api/motion-groups/{group_id}")
def delete_motion_group(group_id: str) -> dict:
    groups = read_motion_groups()
    next_groups = [group for group in groups if group["id"] != group_id]
    if len(next_groups) == len(groups):
        raise HTTPException(status_code=404, detail="Motion group not found")
    write_motion_groups(next_groups)
    return {"ok": True, "group": group_id}


@app.post("/api/motion-groups/{group_id}/play")
def play_motion_group(group_id: str) -> dict:
    _, group = find_motion_group(group_id)
    request = write_motion_playback_request(group)
    return {"ok": True, "request_id": request["request_id"], "request": request}


@app.post("/api/reload")
def request_reload() -> dict:
    touch_reload_flag()
    return {"ok": True, "flag": str(MOTION_DIR / ".motion_reload_request")}


@app.patch("/api/motions/{name}/alias")
def update_motion_alias(name: str, payload: dict[str, str | None] = Body(default_factory=dict)) -> dict:
    motion_name = safe_motion_name(name)
    motion_path = assert_under_motion_dir(MOTION_DIR / motion_name)
    if not motion_path.exists() or not motion_path.is_dir() or motion_path.is_symlink():
        raise HTTPException(status_code=404, detail="Motion not found")
    if not is_user_uploaded_motion(motion_name):
        raise HTTPException(status_code=403, detail="Preset motions cannot be renamed")
    alias = clean_motion_alias(payload.get("alias"))
    set_motion_alias(motion_name, alias)
    return {"ok": True, "motion": motion_name, "alias": alias}


@app.post("/api/motions/upload")
async def upload_motion(
    files: list[UploadFile] = File(...),
    motion_name: str | None = Form(default=None),
    motion_alias: str | None = Form(default=None),
) -> dict:
    ensure_motion_dir()
    tmp_root = MOTION_DIR / ".upload_tmp" / uuid.uuid4().hex
    tmp_root.mkdir(parents=True)
    upload_filename = files[0].filename if files else ""
    try:
        if len(files) == 1 and upload_filename.lower().endswith(".zip"):
            archive = tmp_root / "upload.zip"
            archive.write_bytes(await files[0].read())
            extract_root = tmp_root / "extracted"
            extract_root.mkdir()
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    target = (extract_root / member.filename).resolve()
                    if extract_root.resolve() not in target.parents and target != extract_root.resolve():
                        raise HTTPException(status_code=400, detail="Zip contains unsafe paths")
                zf.extractall(extract_root)
            source_root = find_motion_root(extract_root)
        else:
            extract_root = tmp_root / "folder"
            extract_root.mkdir()
            for upload in files:
                rel = Path(upload.filename.replace("\\", "/"))
                if rel.is_absolute() or ".." in rel.parts:
                    raise HTTPException(status_code=400, detail=f"Unsafe filename: {upload.filename}")
                target = extract_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(await upload.read())
            source_root = find_motion_root(extract_root)
        if motion_name:
            final_name = safe_motion_name(motion_name)
        elif source_root.name in {"extracted", "folder"}:
            final_name = motion_name_from_upload(upload_filename)
        else:
            final_name = safe_motion_name(source_root.name)
        target_dir = assert_under_motion_dir(MOTION_DIR / final_name)
        if target_dir.exists():
            raise HTTPException(status_code=409, detail=f"Motion already exists: {final_name}")
        validation = validate_motion_dir(source_root, strict=True)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail={"errors": validation["errors"]})
        shutil.move(str(source_root), str(target_dir))
        clean_alias = clean_motion_alias(motion_alias)
        if clean_alias:
            set_motion_alias(final_name, clean_alias)
        touch_reload_flag()
        return {"ok": True, "motion": final_name, "alias": clean_alias, "validation": validation}
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@app.delete("/api/motions/{name}")
def delete_motion(name: str) -> dict:
    ensure_motion_dir()
    motion_name = safe_motion_name(name)
    source = assert_under_motion_dir(MOTION_DIR / motion_name)
    if not source.exists() or not source.is_dir() or source.is_symlink():
        raise HTTPException(status_code=404, detail="Motion not found")
    trash_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{motion_name}"
    target = assert_under_motion_dir(MOTION_DIR / ".trash" / trash_name)
    shutil.move(str(source), str(target))
    set_motion_alias(motion_name, None)
    remove_motion_from_groups(motion_name)
    touch_reload_flag()
    return {"ok": True, "motion": motion_name, "trash": str(target)}
