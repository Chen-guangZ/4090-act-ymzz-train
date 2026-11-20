# YMBot-D 视觉抓取项目（Isaac Lab 3.0）

目标：轮式人形机器人（上身）用**头部双目 + 手腕相机**的视觉反馈，RL 学会用 RS485 二指夹爪抓取桌上的快递盒，并支持仿真数据采集。

## 当前进度（v0.3）

- ✅ 场景 `assets/scene_table_box.usda`：机器人 + 白色桌面（顶面 z=1.0m）+ **6.5cm 牛皮纸箱**（按真机夹爪最大开口 ~8cm 留抓取余量）+ 地面 + 灯光
- ✅ 相机：头部双目 HeadCamL/R（基线 8cm，来自推盒项目 3a 验证版）+ 腕部 WristCam（挂 Right_Arm_Link8，沿指方向）
- ✅ 渲染验证 `tools/render_check.py`（无头出图 + 包围盒检查）
- ❌ 未做：**夹爪手指关节**（现手为整块刚体——真机二指夹爪未建模，最大开口 8cm）、盒子物理资产（现场景盒是视觉摆设）、RL 环境与训练、视觉观测

## 目录

```
assets/
├── scene_table_box.usda      # 主场景（GUI 直接打开这个文件）
├── ymbot_d_rl/               # 机器人资产副本（含双目+WristCam）
├── table_white.usda          # 白色桌面（复用推盒桌板网格）
├── box_cardboard.usda        # 纸箱外观资产（视觉用）
└── textures/                 # 白桌面/纸箱纹理
tools/
├── add_wrist_camera.py       # 腕部相机注入（一次性，已执行）
├── make_scene_props.py       # 生成白桌/纸箱纹理与资产副本
└── render_check.py           # 无头渲染验证
```

## 常用命令

```bash
# GUI 里打开场景：File → Open → /home/nvidia4090/ymbot_grasp/assets/scene_table_box.usda
# （不要在 GUI 里 File→Save 覆盖源文件！改动走脚本+git）

# 渲染验证（无头）
OMNI_KIT_ACCEPT_EULA=YES /home/nvidia4090/IsaacLab/.venv/bin/python /home/nvidia4090/ymbot_grasp/tools/render_check.py
```

## 规划（详见 /home/nvidia4090/HANDOFF.md 总交接文档）

① 夹爪 2 指 prismatic（0-8cm 开口）+ 盒子物理资产（~60g）+ 物理可行性验证
② RL 环境（8 维动作：7 臂关节+1 夹爪）+ 奖励（距离引导→抬升成功）
③ One-shot 初始化课程 W0→W3（手预摆盒上方 → 位置渐宽随机）
④ 视觉观测版（头双目+腕相机替换真值，复用 ymbot_push perception 模块）
⑤ 真机数据对齐（姿态快照/桌高/盒尺寸，见总交接文档 §6）
