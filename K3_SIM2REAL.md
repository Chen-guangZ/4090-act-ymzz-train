# K3 现场试验回放

这一套流程使用 2026-11-20 K3 的真机关节反馈与夹爪命令，验证 Isaac Lab 工作区里的抓取资产。动力学在 Newton MJWarp 中运行，Isaac Sim 用于标定相机的 RGB 渲染。两者使用同一份 USD。

## 资料与产物

- 本地与 4090 各有一份 `real_data/k3_20261120/`，其中 `raw/` 为原始资料，`derived/` 为可再生的试验索引与轨迹，`validation/` 为验证结果。归档来源、标签和字段含义见数据目录的 README。
- `k3_scene.py` 从单次试验生成校准 USD，包含关节姿态、头部 Gemini 2L 和腕部 D405、静态桌、箱体及右手夹爪。
- `fit_k3_pad_offsets.py` 从归档的开口标定 RGB-D 重建指垫中点，并保留像素标注、点云分位数与拟合法向。
- `replay_k3_grasp.py` 回放真实关节角反馈与夹爪比例，箱体只受动力学接触力驱动。输出盒底净空、抬升持续时间、位置和开口时间序列。
- `probe_k3_pad_contact.py` 用直线抬升试验隔离居中夹持和偏离箱体质心的接触行为。
- `render_k3_scene.py` 渲染同一 USD 的头部和腕部视角。
- `prepare_k3_replay.py` 可从原始备份重建试验索引和轨迹，使用 Python 的 `rosbags`、`numpy`。导出前先校验 `SHA256SUMS`。

## 4090 复现

```bash
cd /home/nvidia4090/ymbot_grasp
PY=/home/nvidia4090/IsaacLab/.venv/bin/python
D=/home/nvidia4090/real_data/k3_20261120
V="$D/validation/reproduced"
mkdir -p "$V"
export OMNI_KIT_ACCEPT_EULA=YES
for trial in 17 18 19 20 21; do
  "$PY" -B tools/k3_scene.py "$D/derived" "$V/trial_$trial.usda" --trial "$trial"
  "$PY" -B tools/replay_k3_grasp.py "$V/trial_$trial.usda" "$D/derived/trial_$trial" "$V/dynamics_$trial.json" \
    --sample-contacts --arm-bias-compensation --through-release
done
"$PY" -B tools/render_k3_scene.py "$V/trial_21.usda" "$V/render_21"
"$PY" -B tools/probe_k3_pad_contact.py "$V/pad_contact.json"
```

## 物理与观测说明

- 厂家图纸给出 CTAG2F90 的 90 mm 行程、23 N 最大夹持力和 0.7 kg 整体重量。STEP 指面包围盒约 23 × 39.5 × 6 mm。每指由横向及轴向滑块构成，回放根据实际横向位置联动轴向滑块，近似平动连杆在开合时的 17.5 mm 轴向位移。保留了现有腕部惯量作为回放初值，指垫质量与摩擦均为名义参数。
- 真机开口标定在比例 1.0 / 0.5 / 0.3 / 0.2 时分别约为 88 / 50 / 31 / 21 mm。0.25 约 26 mm，取插值。夹持力上限按厂家最大夹持力均分为每指 11.5 N，用于初步接触验证。
- 台面高度和法向使用各试验的 RGB-D 平面拟合。从 #21 的侧面和顶面恢复纸箱共同的约 150 × 49 mm 水平外形，并按各试验侧面点云定位和估高。纸箱质量暂用 60 g。腕部渲染使用针孔投影，D405 畸变参数作为 USD 元数据保留。
- 回放默认覆盖闭合前 0.75 s 至开始放置前。启用 `--through-release` 后沿录制轨迹回摆、开爪并撤离，输出撤离后的箱底净空、箱体相对起点的平面位移及静止情况。
- 回放用实测关节角作位置参考。`--arm-bias-compensation` 补偿仿真中的重力与科氏偏置，用于减少因跟踪偏差导致的接触错位。真实臂控制器、电流、力矩和胶垫变形需要单独标定。
- RGB-D 指垫中点在 `Right_Arm_Link8` 下约为 `[-11.0, 15.7, -158.7]` mm。两指横向行程通过 Newton mimic 约束同步。法向拟合得到的轴角保留为可选对照，默认仍采用工具 X 轴。

## 验收口径

盒底净空按箱体当前姿态计算，包含转动导致的最低角点变化。离桌至少 10 cm 且连续保持至少 1 秒为通过。单体夹持通过后，还要用 #17 抓空及 #18–#21 的人工确认记录回归整机。闭合位置、钳口材料、摩擦、装载后的质量和质心需要进一步实测，现有整机数据不足以唯一识别这些参数。

## 本轮验证

- 原始 31 个 bag 均有 `/joint_states`。五个对照试验的姿态、夹爪指令和人工结果已归档。头部 1280×720 与腕部 640×480 均通过 RGB 渲染检查。
- 五个试验的 Newton FK 抓取点与 K3 在 `Body_Link5` 的 FK 记录相差约 0.11–0.22 mm。这是坐标系与模型一致性检查。
- 夹爪单体测试中，0.50 抓空；0.25 居中夹持的箱底抬高约 14.8 cm 并保持超过 1 秒；夹点偏离箱体中心 60 mm 时未稳定保持。
- 首轮报告保存在 `validation/baseline/summary.json`。第二轮结果在 `validation/contact_replay_confirmed/summary.json`，同目录保留两指接触载荷、持续离桌时间和撤离后的箱体状态。
- #17 复现抓空。#20 复现抓起和搬运，峰值净空约 37.0 cm，两指保持接触且箱底离桌至少 10 cm 连续约 37.0 秒。返回桌面的途中滑落，最后一次双指接触早于录制的开爪指令约 4.9 秒。
- #18、#19、#21 峰值净空约 4.4、4.1、4.9 cm，均未达到整机抬升门槛。默认参数采用 60 g 名义质量和 K3 RGB-D 指垫位置。继续优化需要实测纸箱装载重量和夹持区域尺寸，以及可同步的抬升、回摆侧视记录。
