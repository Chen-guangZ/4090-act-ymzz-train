# 工作进度、数据与验收结果

## 提交里程碑

| 提交 | 交付 |
|---|---|
| `7cc19d8`、`d6514ce` | 指垫接触和箱体相对腕部滑移测量，新增放回失夹和载荷敏感性诊断 |
| `e6fb52c`、`42043ea` | 完整放回评分、夹爪开合时间对照、从腕部图像估计真实放回位移 |
| `c6beeee`、`62c0c4c` | 摩擦约束阻抗调参、CPU/GPU 与确定性模式对照 |
| `9ed0306` | 接通 Isaac Lab CUDA MJWarp 管理器，阻抗比 800 下五例通过 |
| `884f426` | 重新验收、#21 USD 动画、只读预览和操作手册 |
| `1aade2d` | 4090 ACT 路线图和目标 01 验收任务单 |

## 数据存放位置

| 位置 | 内容 |
|---|---|
| `/home/nvidia4090/ymbot_grasp/` | 4090 项目工作区 |
| `/home/nvidia4090/IsaacLab/` | IsaacLab 源码，验证版本 `3939445`，运行 Python 为 `.venv/bin/python` |
| `/home/nvidia4090/real_data/k3_20261120/` | 4090 的数据根目录 |
| Windows 工作区 `real_data/k3_20261120/` | 原始数据、派生输入和完整验证输出，已对照校验 1,069 个稳定文件 |
| Windows 工作区 `4090_sync/runtime/` | 同步前 Git bundle、IsaacLab 源码归档、源码工作区补丁与软件包版本 |
| Windows 工作区 `4090_sync/references/` | URDF、MoveIt、驱动资料及旧演示引用素材 |
| Windows 工作区 `4090_WORKSPACE_STATUS.md` | 远端路径、对应本地位置和原始同步记录 |

## 数据根目录索引

| 相对路径 | 用途 |
|---|---|
| `README.md`、`SHA256SUMS` | 数据索引及原始归档校验 |
| `raw/run_logs_and_bags.tar.gz` | 31 个 ROS bag 和运行日志，用于提取关节反馈、动作事件和夹爪命令 |
| `raw/calibration_and_labels.tar.gz` | 真机标定、现场图像、人工标签和配置，用于重建环境与核对物体行为 |
| `raw/deployed_drivers/`、`raw/gripper_reference/` | 当时的驱动实现、夹爪图纸与 STEP 参考 |
| `raw/k3_snapshot_20261121*/` | K3 现场补充快照及其来源 |
| `derived/bag_catalog.json`、`derived/trial_index.json` | bag 话题与现场案例索引 |
| `derived/trial_17/` 到 `trial_21/` | 五例控制回放输入，包括关节轨迹、夹爪命令、感知快照和事件 |
| `validation/pad_patch_grid/torsional/` | 五例验收所用的分块指垫 USD 与场景参数 |
| `validation/placement_regression_360hz_gpu/` | 优化前五例轨迹，供对照失夹及掉落 |
| `validation/contact_impedance/` | 阻抗敏感性、不同步长和 CPU/GPU 对照 |
| `validation/isaaclab_profile/` | Isaac Lab 管理器回放、评分和独立 GPU 对照 |
| `validation/acceptance_demo/` | 最后五例评分、#21 动画 USD、截图和运行日志 |

公开仓库保留代码、USD 资产和下面四份小型仿真验收证据。两个原始真机归档各超过 100 MB，包含现场记录，保存在既有 4090 与 Windows 数据备份。部分 USD 使用 Ubuntu 绝对引用，更换路径时需重建校准场景或重定位引用。

原始 bag 适合复现和时序对照。目前缺连续双相机流和逐帧实际下发的右臂命令。双相机 ACT 的训练样本来自后续仿真生成或同步实采。

## 仓库中的验收证据

- [五例评分](results/score_all.json) 来自 `validation/acceptance_demo`，保留原报告中的阈值、逐例检查和数据根目录下的轨迹引用。
- [运行入口对照](results/runtime_comparison.json) 来自 `validation/isaaclab_profile`，比较独立 MJWarp GPU 与 Isaac Lab 管理器。
- [抬升截图](results/preview_lift.jpg) 和 [放回截图](results/preview_place.jpg) 来自 #21 的 Isaac Sim 渲染。
