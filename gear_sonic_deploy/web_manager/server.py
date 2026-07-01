from __future__ import annotations

import csv
import math
import os
import re
import shutil
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_MOTION_DIR = PROJECT_ROOT / "reference" / "self"
MOTION_DIR = Path(os.environ.get("SONIC_MOTION_DIR", DEFAULT_MOTION_DIR)).resolve()

REQUIRED_FILES = {
    "metadata.txt",
    "info.txt",
    "joint_pos.csv",
    "joint_vel.csv",
    "body_pos.csv",
    "body_quat.csv",
    "body_lin_vel.csv",
    "body_ang_vel.csv",
}
OPTIONAL_FILES = {"smpl_joint.csv", "smpl_pose.csv"}
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

app = FastAPI(title="Sonic Motion Web Manager")
static_dir = APP_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def ensure_motion_dir() -> None:
    MOTION_DIR.mkdir(parents=True, exist_ok=True)
    (MOTION_DIR / ".upload_tmp").mkdir(exist_ok=True)
    (MOTION_DIR / ".trash").mkdir(exist_ok=True)


def safe_motion_name(name: str) -> str:
    cleaned = name.strip().strip("/\\")
    if not MOTION_NAME_RE.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail=f"Invalid motion name: {name!r}")
    return cleaned


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
    if not body_indexes:
        errors.append("metadata.txt missing Body part indexes")
    elif len(body_indexes) != 14:
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "motion_dir": str(MOTION_DIR)}


@app.get("/api/motions")
def list_motions() -> dict:
    motions = []
    for path in iter_motion_dirs():
        validation = validate_motion_dir(path, strict=False)
        stat = path.stat()
        motions.append({
            "name": path.name,
            "valid": validation["valid"],
            "errors": validation["errors"],
            "timesteps": validation["metadata_timesteps"] or next(iter(validation["rows"].values()), None),
            "size_bytes": directory_size(path),
            "mtime": stat.st_mtime,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    return {"motion_dir": str(MOTION_DIR), "motions": motions}


@app.post("/api/reload")
def request_reload() -> dict:
    touch_reload_flag()
    return {"ok": True, "flag": str(MOTION_DIR / ".motion_reload_request")}


@app.post("/api/motions/upload")
async def upload_motion(files: list[UploadFile] = File(...), motion_name: str | None = Form(default=None)) -> dict:
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
        touch_reload_flag()
        return {"ok": True, "motion": final_name, "validation": validation}
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
    touch_reload_flag()
    return {"ok": True, "motion": motion_name, "trash": str(target)}
