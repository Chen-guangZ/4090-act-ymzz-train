# 目标 01：可重置纸箱抓取任务

## 与 ymbot_display 的对照

2026-11-24 查阅了 4090 的 `/home/nvidia4090/ymbot_display/HANDOFF.md`、`README.md`、网格检查脚本和 GLB。实际导出的 GLB 为 730,028 字节，包含 36 个网格和 40,606 个三角形，无动画。右腕有 `L_Right_Arm_Link8` 层级节点。交接记录中底盘仍有孔洞、右掌有破面，15 个简化胶囊已经封口。

| 可复用项 | 放到本路线的位置 |
|---|---|
| 轻量 GLB 和 URDF 对应的零件名 | 后续可用于浏览器展示。接动画前需对齐连杆坐标系、关节轴、零位和相机；以实际 K3 状态驱动。 |
| 近景检查与程序化报告 | 验收同时留数值日志和录像。检查底盘、固定关节、抓空与夹爪外观时分开看。 |
| 网格减面、胶囊替换 | 可在目标 03 单独对比渲染速度和头腕图像差异。当前抓取使用 K3 校准 USD 的碰撞、关节、惯量和相机。 |

显示工程只提供外观模型，缺少仿真关节控制与碰撞物理。两个工程的资产各自保存。

## 接口和运行方法

`config/k3_box_task_01.json` 指向已通过五例回归的 #21 分块 USD。固定观察位来自 #21 现场反馈的 2026-11-20 19:13:21（北京时间）。求解为 Isaac Lab CUDA MJWarp、360 Hz、椭圆摩擦锥、阻抗比 800。动作频率为 30 Hz。

- `K3BoxTask.reset(seed=..., randomize=...)` 重建物理上下文，重置求解器、动作和夹爪时序。启用随机化时，种子决定箱体的平面位置和偏航。
- `step(action)` 接收七个绝对关节角（rad）和一个夹爪比例（0–1），返回七关节反馈和夹爪指令代理值。日志另含实际应用的关节目标、夹爪间距、接触力、箱底高度和结束原因。
- 底盘固定，躯干、颈部和左臂保持观察位目标。右臂目标按模型限位和配置中的模拟斜率处理。
- 双指带载、箱底离桌至少 10 cm 且持续 1 秒时报告 `lifted_and_held`；另外报告 `box_fell`、`grasp_empty` 或 `timeout`。

```bash
cd /home/nvidia4090/ymbot_grasp
export OMNI_KIT_ACCEPT_EULA=YES
D="$HOME/real_data/k3_20261120"
P="$HOME/IsaacLab/.venv/bin/python"
OUT=$(mktemp -d "$D/validation/task01_XXXXXX")
"$P" -B tools/run_k3_box_task.py "$D" "$OUT/episodes"
```

每次运行生成三条运动日志、张爪对照日志、两段动画 USD 和 `task_score.json`。张爪对照通过同一动作接口执行 #21 的手臂接近和抬升指令，夹爪比例始终为 1。可用下面的命令导出录像：

```bash
for trial in motion_0 open_jaw; do
  "$P" -B tools/show_k3_animation.py "$OUT/episodes/$trial.usda" --gif "$OUT/episodes/$trial.gif"
done
```

## 4090 验收结果

2026-11-24 在 `cuda:0` 跑完三次重置、一次张爪负例和旧的五例基线。任务和回归评分均通过。

| 检查 | 结果 |
|---|---|
| 同种子 | 种子 7 两次重置得到相同箱位，整段动作的八维观测最大差值为 0。 |
| 不同种子 | 种子 8 更换了箱体初始平面位置。 |
| 运动与夹爪 | 右臂第一关节跟踪 0.05 rad 指令后误差约 0.00000080 rad。夹爪合至 26.0 mm，重新张开至 88.0 mm。 |
| 底座与固定关节 | 底座位置和姿态在记录中保持不变，固定关节最大偏移约 0.000364 rad。 |
| 张爪负例 | 最小开口 87.94 mm，纸箱始终未离桌，结束原因是 `grasp_empty`。 |
| 原五例基线 | #17 复现抓空。#18–#21 保持到开爪并稳定放回，本次放回距初始平面位置分别为 7.14、6.47、8.49、6.83 mm。 |

[任务评分](results/task01_score.json) · [重跑五例评分](results/task01_replay_regression.json) · [右臂和夹爪动作录像](results/task01_motion.gif)

## 验收产物

4090 的目录是 `/home/nvidia4090/real_data/k3_20261120/validation/task_01/`，Windows 副本位于工作区的 `real_data/k3_20261120/validation/task_01/`。

| 相对路径 | 内容 |
|---|---|
| `task_run_final/motion_0.json` 至 `motion_2.json` | 三次重置的箱位、实际动作和每步状态。 |
| `task_run_final/open_jaw.json` | 张爪负例的关节、间距、接触和结束原因。 |
| `task_run_final/task_score.json`、`task_final.log` | 新任务评分和运行日志。 |
| `task_run_final/motion_0.usda`、`open_jaw.usda` 及同名 GIF | 运动和张爪负例的 USD 动画与离屏录像。 |
| `replay_17.json` 至 `replay_21.json` 及同名日志 | 按旧命令生成的五例回放时序和运行日志。 |
| `replay_score_all.json`、`run_replay_regression.sh` | 五例回归评分和运行命令。 |

旧的 #17–#21 回放继续按 [4090 验收手册](../4090_ACCEPTANCE_GUIDE.md) 复现。目标 01 验证了可重置的执行接口。目标 02 通过该接口接几何专家，先做一例抓起保持，再测试不同箱位。
