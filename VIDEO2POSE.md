# 搭建环境

1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. 仿真

```bash
install_scripts/install_mujoco_sim.sh
```

1. 下载模型

```bash
uv run \
  --python 3.10 \
  --with huggingface-hub \
  --with socksio \
  python download_from_hf.py
```

3.1 制作离线模型包给无法联网设备部署使用

```bash
bash install_scripts/package_gear_sonic_models.sh ./sonic_model.tar.gz
```

3.2 使用离线模型包

```bash
bash install_scripts/install_gear_sonic_models_offline.sh ./sonic_model.tar.gz
```

# 测试舞蹈

## 仿真

终端1 开启仿真

```bash
env -u PYTHONPATH \
  .venv_sim/bin/python \
  gear_sonic/scripts/run_sim_loop.py
```

终端2 启动 sonic 控制

```bash
./run_sonic_sim.sh
```

终端3 启动原始舞蹈动作和 sonic 动作对比 

左：标动作参考姿态  中：红色为实机动作，绿色为目标动作  右：温度可视化机器人

```bash
source .venv_sim/bin/activate

python gear_sonic_deploy/visualize_motion.py \
  --realtime_debug_url tcp://127.0.0.1:5557 \
  --realtime_debug_topic g1_debug \
  --fps 50 \
  --align-target-height \
  --ground-clearance 0.005 \
  --overlay-target-y-offset 0.0 \
  --temperature-y-offset -1.0 \
  --hide-vr-markers
```

终端4 管理 SONIC 舞蹈页面

```bash
cd gear_sonic_deploy
uv run --project web_manager uvicorn web_manager.server:app \
  --host 127.0.0.1 \
  --port 8888
```



## 真机

跳舞

```bash
tmux
cd ~/GR00T-WholeBodyControl
yes | bash ~/GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh real --input-type gamepad --motion-data ~/GR00T-WholeBodyControl/gear_sonic_deploy/reference/self
```

群控

```bash
cd ~/GR00T-WholeBodyControl
yes | bash ~/GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh real --input-type gamepad --motion-catalog ./gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/motion_catalog.example.yaml
```

触发

```bash
ros2 topic pub --once /WBCPolicy/select_motion std_msgs/msg/String "{data: 'squat'}"
```



# 转换青瞳舞蹈数据

默认版本不带平滑

```bash
python3 gear_sonic_deploy/reference/convert_g1_retargeting_csv_to_sonic.py
```

生成平滑版本

```bash
python3 gear_sonic_deploy/reference/convert_g1_retargeting_csv_to_sonic.py \
--output-dir gear_sonic_deploy/reference/self/qt_take_017_smooth \
--motion-name qt_take_017_smooth \
--smooth-sigma 1.5
```

离线动作查看器（不包含电机和sonic控制器）

```bash
cd /home/mark/Documents/Dance/GR00T-WholeBodyControl
source .venv_sim/bin/activate

python gear_sonic_deploy/visualize_motion.py \
--motion_dir gear_sonic_deploy/reference/self/qt_take_017_smooth \
--fps 50
```



# 机器人状态管理

开机自启动，sonic启动 Init Done 之后机器人状态为 WAIT_FOR_CONTROL，此时电机使能 + 默认站姿 PD hold + 等待 Start 命令

按下 Start 机器人直接从 Init Done （要人扶）的状态变成站起来了（不用人扶）,可以直接下发群控命令下蹲

把机器人扶起来站立，然后按下 Start 启动 policy 控制系统，默认进入 motion_catalog.example.yaml配置的第一个动作的第 0 帧

select 进入阻尼

# 重启 SONIC 服务

```bash
sudo systemctl stop groot-deploy.service
tmux kill-session -t sonic 2>/dev/null || true
sudo systemctl start groot-deploy.service
```

