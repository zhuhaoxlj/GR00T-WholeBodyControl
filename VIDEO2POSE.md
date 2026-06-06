# 测试舞蹈

## 仿真
终端1 开启仿真
```bash
cd /home/mark/Project/01-RL/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

终端2 启动 sonic 控制
```bash
cd /home/mark/Project/01-RL/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh --input-type keyboard --motion-data reference/all sim
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