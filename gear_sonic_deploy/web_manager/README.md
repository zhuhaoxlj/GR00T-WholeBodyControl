# Sonic Motion Web Manager

Local web UI for adding and deleting Sonic reference motions while `g1_deploy_onnx_ref` is running.

## Start

```bash
cd /home/mark/Documents/Dance/gr00t-wholebodycontrol/gear_sonic_deploy
uv run --project web_manager uvicorn web_manager.server:app --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

By default the manager uses:

```text
reference/self
```

Override it with:

```bash
SONIC_MOTION_DIR=/absolute/path/to/reference/self \
uv run --project web_manager uvicorn web_manager.server:app --host 127.0.0.1 --port 8080
```

## Motion package requirements

A motion directory or zip must contain exactly one motion folder with:

```text
metadata.txt
info.txt
joint_pos.csv
joint_vel.csv
body_pos.csv
body_quat.csv
body_lin_vel.csv
body_ang_vel.csv
```

Optional:

```text
smpl_joint.csv
smpl_pose.csv
```

Uploads are validated in `.upload_tmp` and then moved into the motion directory. Deletes are moved to `.trash` and are not permanently removed.

After upload/delete, the manager updates `.motion_reload_request`; the Sonic process detects this flag and hot-reloads the motion list. You can also press `U` in keyboard input mode to reload manually.
