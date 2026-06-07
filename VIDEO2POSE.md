# 测试舞蹈

## 仿真
终端1 开启仿真
```bash
cd /home/mark/Documents/Dance/gr00t-wholebodycontrol
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

终端2 启动 sonic 控制
```bash
cd /home/mark/Documents/Dance/gr00t-wholebodycontrol/gear_sonic_deploy
bash deploy.sh --input-type keyboard --motion-data reference/self sim
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



# 转换青铜舞蹈数据
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
cd /home/mark/Documents/Dance/gr00t-wholebodycontrol
source .venv_sim/bin/activate

python gear_sonic_deploy/visualize_motion.py \
--motion_dir gear_sonic_deploy/reference/self/qt_take_017_smooth \
--fps 50
```