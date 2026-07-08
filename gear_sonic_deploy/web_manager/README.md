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

## MuJoCo Web simulator smoke test

The browser-side MuJoCo migration starts as a separate Vite + TypeScript app in
`web_manager/frontend`. The FastAPI server exposes the G1 MJCF and mesh assets
through `/api/sim/*`, while the Vite app loads `@mujoco/mujoco`, creates a real
MuJoCo `MjModel`/`MjData`, runs `mj_step`, and renders MuJoCo scene geoms with
Three.js on a WebGL canvas.

Install and build the frontend:

```bash
cd /home/mark/Documents/Dance/GR00T-WholeBodyControl/gear_sonic_deploy/web_manager/frontend
npm install
npm run build
```

Then start the FastAPI server from `gear_sonic_deploy`:

```bash
uv run --project web_manager uvicorn web_manager.server:app \
  --host 127.0.0.1 \
  --port 8080
```

Open the built simulator page:

```text
http://127.0.0.1:8080/sim
```

For frontend development, run Vite in another terminal:

```bash
cd /home/mark/Documents/Dance/GR00T-WholeBodyControl/gear_sonic_deploy/web_manager/frontend
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8080`.

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

## Motion groups

The web UI stores motion groups in `.motion_groups.json` under `SONIC_MOTION_DIR`.
Playing a group writes `.motion_playback_request.json`; the running Sonic process
detects that request and plays each referenced motion in order. During group
playback, intermediate motion endings advance directly to the next clip. The
final clip stops on its last frame.
