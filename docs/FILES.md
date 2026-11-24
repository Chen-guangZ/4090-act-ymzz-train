# 逐文件说明

首次验收从 [README](../README.md) 和 [4090 验收手册](../4090_ACCEPTANCE_GUIDE.md) 开始。现场归档与实验目录的对应关系见 [数据和工作进度](DATA_AND_RESULTS.md)。

## 根目录与文档

| 文件 | 作用 |
|---|---|
| `README.md` | 项目现状、验收结果、#21 快速复现和文档入口。 |
| `LICENSE` | GitHub 仓库初始化时的 Apache 2.0 许可证。 |
| `.gitignore` | 忽略 Python 缓存、日志、运行环境、本地备份和完整数据。 |
| `4090_ACCEPTANCE_GUIDE.md` | 4090 登录、Ubuntu/Windows 查看、单例和五例回放、评分条件及预览管理。 |
| `4090_ACT_ROADMAP.md` | 4090 后续十个目标及对应的交付物、验收和现场配合边界。 |
| `K3_SIM2REAL.md` | K3 仿真技术记录，含校准、接触试验、求解参数、对照结果和复现方法。 |
| `docs/FILES.md` | 当前这份文件索引。 |
| `docs/DATA_AND_RESULTS.md` | 外置数据、4090 路径、里程碑及验收证据来源。 |
| `docs/results/score_all.json` | 五例验收结果、阈值和每项检查。 |
| `docs/results/runtime_comparison.json` | Isaac Lab 管理器和独立 MJWarp GPU 的五例对照。 |
| `docs/results/preview_lift.jpg` | #21 抬起纸箱时的仿真截图。 |
| `docs/results/preview_place.jpg` | #21 将纸箱放回台面时的仿真截图。 |

## USD 与纹理

| 文件 | 作用 |
|---|---|
| `assets/scene_table_box.usda` | 早期 Isaac Sim 桌面演示入口，引用机器人、白桌和纸箱资产。 |
| `assets/table_white.usda` | 白色桌面演示资产及材质引用。 |
| `assets/box_cardboard.usda` | 牛皮纸箱外观资产及材质引用。 |
| `assets/textures/table_white.png` | 演示白桌使用的纹理图。 |
| `assets/textures/box_cardboard.png` | 演示纸箱使用的纹理图。 |
| `assets/ymbot_d_rl/ymbot_d_rl.usda` | YMBot-D 机器人根层，组装模型并选择物理变体。 |
| `assets/ymbot_d_rl/payloads/base.usda` | 机器人基础层，组织连杆层级、变换和相机挂载，并引用网格与材质。 |
| `assets/ymbot_d_rl/payloads/geometries.usd` | 二进制 USD 几何资源，存储网格。 |
| `assets/ymbot_d_rl/payloads/instances.usda` | 为各连杆组织网格的实例定义，供基础层引用。 |
| `assets/ymbot_d_rl/payloads/materials.usda` | 视觉材质和颜色定义。 |
| `assets/ymbot_d_rl/payloads/robot.usda` | 机器人描述层，保留部件和关节信息。 |
| `assets/ymbot_d_rl/payloads/Physics/mujoco.usda` | MuJoCo 物理变体及其求解相关属性。 |
| `assets/ymbot_d_rl/payloads/Physics/physics.usda` | USD 通用物理变体，定义刚体、关节与碰撞属性。 |
| `assets/ymbot_d_rl/payloads/Physics/physx.usda` | PhysX 专用物理变体和属性。 |

## 数据、标定和诊断工具

| 文件 | 作用 |
|---|---|
| `tools/prepare_k3_replay.py` | 从 K3 归档构建 bag 索引，导出关节状态、夹爪指令、现场事件和逐例回放输入。 |
| `tools/fit_k3_pad_offsets.py` | 用深度与手工区域标注拟合夹爪指垫的位置、方向和间距。 |
| `tools/k3_camera.py` | D405 图像坐标去畸变和归一化成像射线。 |
| `tools/k3_gripper.py` | 夹爪常数、指令比例到开口映射、指垫轴向补偿和连杆近似。 |
| `tools/k3_scene.py` | 从派生输入生成校准 USD 和场景参数 JSON，包含机器人、两相机、夹爪、桌面、纸箱。 |
| `tools/measure_k3_placement.py` | 从现场腕部视图、观测标注与机器人姿态估计纸箱放回位移。 |
| `tools/track_k3_box_crease.py` | 跟踪归档腕相机帧中的纸箱折痕，测量保持时的相对滑移。 |
| `tools/probe_k3_pad_contact.py` | 指垫接触和参数敏感性的小型独立探针。 |
| `tools/analyze_k3_retention.py` | 汇总夹持、失夹、开爪、放回阶段和桌面支撑的诊断信息。 |

## 回放、评分、渲染与预览

| 文件 | 作用 |
|---|---|
| `tools/k3_isaaclab_profile.py` | 配置 Isaac Lab CUDA MJWarp 管理器，支持导入、步进和参数核验。 |
| `tools/replay_k3_grasp.py` | 回放真机关节与夹爪指令，通过独立 Newton 或 Isaac Lab 求解，输出姿态和接触数据 JSON。 |
| `tools/check_k3_placement.py` | 对抓空、带载抬升、保持到开爪、撤离后支撑和位置误差评分。 |
| `tools/export_k3_animation.py` | 将求解后的机器人和纸箱位姿写成可播放的 USD 动画。 |
| `tools/render_k3_scene.py` | 使用 Isaac Sim 离屏渲染校准后的头部或腕部相机静态图。 |
| `tools/show_k3_animation.py` | 在 Isaac Sim 循环播放动画，可离屏更新浏览器预览帧。 |
| `tools/serve_k3_view.py` | 通过本机回环 HTTP 端口提供只读预览页面和当前帧。 |
| `tools/run_k3_preview.sh` | 在 Ubuntu 启动渲染器和预览服务，统一管理这两个进程。 |
| `tools/add_wrist_camera.py` | 早期演示资产的腕部相机注入工具。 |
| `tools/make_scene_props.py` | 从既有推盒素材生成早期白桌、纸箱和纹理。 |
| `tools/render_check.py` | 早期桌面演示的离屏渲染与桌面高度、包围盒检查。 |
