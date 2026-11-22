# K3 抓放仿真工作总结与 4090 验收手册

更新日期 2026-11-22。环境为 RTX 4090、Ubuntu、Isaac Lab 3.0 和 Isaac Sim 6.2。项目在 `/home/nvidia4090/ymbot_grasp`，数据在 `/home/nvidia4090/real_data/k3_20261120`。

## 先看画面，再跑验收

| 想做什么 | 从哪里开始 |
|---|---|
| 在 Windows 看当前画面 | 浏览器打开 `http://127.0.0.1:18765/`，看 #21 的循环回放 |
| 在 Ubuntu 图形桌面操作相机和时间轴 | 按第 2 节打开专用回放 USD |
| 确认五个实机案例能不能复现 | 按第 3 节执行完整回放和评分 |
| 判断是否可以验收 | 对照第 4 节的通过条件和历史结果 |

## 1. 在 4090 上完成的工作

| 工作 | 结果和位置 |
|---|---|
| 真机数据归档 | 31 个 ROS bag、抓放阶段事件、人工标签、RGB-D 快照、夹爪图纸和驱动配置。原始文件位于 `raw/`，处理后的逐例输入位于 `derived/trial_17/` 至 `trial_21/`。本机与 4090 副本已核对 SHA-256。 |
| 场景还原 | 从真机轨迹和深度重建右臂工作姿态、桌面、纸箱碰撞体、二指夹爪、Gemini 2L 头部相机和 D405 腕部相机。分块接触场景在 `validation/pad_patch_grid/torsional/`。 |
| 掉落诊断和物理优化 | 原配置提前失夹。对照了指面结构、质心、推力、闭合速度和碰撞求解；使用 2×2 指垫、360 Hz、椭圆摩擦锥和摩擦约束阻抗比 800 后，Isaac Lab 的 #18–#21 四例保持到开爪并完成放回。 |
| 仿真验收 | `tools/check_k3_placement.py` 给出逐例抬升、保持、桌面支撑、放回位置和静止评分。已核验的 Isaac Lab CUDA 报告在 `validation/isaaclab_profile/score_800_cuda.json`。 |
| 可视化交付 | `validation/acceptance_demo/K3_Trial21_Replay.usda` 将实际 Isaac Lab 输出的机器人、夹爪和箱体位姿按 30 Hz 写入时间轴，供 Isaac Sim 播放。物理回放按 360 Hz 求解。 |

需要追查数据字段、模型假设及历史对照时，看 [K3 现场试验回放](K3_SIM2REAL.md)。

## 2. 看 Isaac Sim 画面

### Windows 浏览器

当前浏览器入口为 `http://127.0.0.1:18765/`。这是由 4090 上的 Isaac Sim 实时渲染、经 SSH 转发的回放视口。浏览器约每三分之一秒取一次画面。页面为只读，切换视角和拖动时间轴可在 Ubuntu 图形桌面操作。

如果 Windows 重启或转发已关闭，在 PowerShell 运行以下命令。按提示输入 Ubuntu 账户密码，保持这一窗口打开，再访问浏览器入口。退出转发用 `Ctrl+C`。

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18765:127.0.0.1:18765 nvidia4090@192.168.11.136
```

如果页面打不开，先在 4090 检查预览服务。已有服务时直接继续使用。没有服务时，运行下列启动命令。

```bash
cd /home/nvidia4090/ymbot_grasp
D=/home/nvidia4090/real_data/k3_20261120
O="$D/validation/acceptance_demo"
curl -fsS -o /dev/null http://127.0.0.1:18765/
# 仅在 curl 确认没有运行实例时，执行以下三行
nohup bash tools/run_k3_preview.sh "$O/K3_Trial21_Replay.usda" > "$O/preview.log" 2>&1 </dev/null &
echo "$!" > "$O/preview.pid"
tail -f "$O/preview.log"
```

日志出现 `Viewing Isaac Sim renderer` 后即可打开页面，首帧需要等待 Isaac Sim 加载。此预览可以在 4090 显示器关闭时运行。Ubuntu 服务和 Windows 转发都绑定本机回环地址，不需要更改路由器端口或防火墙。

如果页面已开但没有图像，在 4090 执行 `curl -f -o /tmp/k3_view.jpg http://127.0.0.1:18765/frame.jpg`。首次加载时可以稍后再试；持续失败时查看 `$O/preview.log`。离线示例图在 `$O/preview_lift.jpg` 和 `$O/preview_place.jpg`。

要停止占用 GPU 的预览，先运行 `ps -fp "$(cat "$O/preview.pid")"`，确认显示 `run_k3_preview.sh`，再运行 `kill -TERM "$(cat "$O/preview.pid")"`。

### Ubuntu 图形桌面

1. 在 Ubuntu 图形桌面打开 Isaac Sim。若要直接打开并循环播放，可在图形桌面的终端运行下方命令。
2. 若要独立检查，在 Isaac Sim 的 `File → Open` 打开：
   `/home/nvidia4090/real_data/k3_20261120/validation/acceptance_demo/K3_Trial21_Replay.usda`。
3. 用 Viewport 的相机菜单选 `K3ReviewCamera`，再用时间轴拖动查看。也可切到 Perspective 改视角。需要保存修改时请另存副本。
4. 看三个关键时段。约 7–9 秒抬起，约 19–28 秒回到桌面上方并下放，约 28.9 秒发出开爪命令，约 36–37 秒夹爪撤离。确认纸箱随手上升、回摆时没有掉落、撤离后留在台面。

时间轴是已求解后的画面。重新运行物理、判断通过或失败，请使用下面的命令。

```bash
cd /home/nvidia4090/ymbot_grasp
D=/home/nvidia4090/real_data/k3_20261120
OMNI_KIT_ACCEPT_EULA=YES /home/nvidia4090/IsaacLab/.venv/bin/python -B tools/show_k3_animation.py \
  "$D/validation/acceptance_demo/K3_Trial21_Replay.usda"
```

## 3. 在 4090 上重新运行

在 Windows PowerShell 登录，或直接打开 Ubuntu 的 Terminal：

```powershell
ssh nvidia4090@192.168.11.136
```

下面的命令都在 **Ubuntu** 终端执行。先确认分支、显卡和数据：

```bash
cd /home/nvidia4090/ymbot_grasp
git status --short --branch
git log -1 --oneline
nvidia-smi
D=/home/nvidia4090/real_data/k3_20261120
(cd "$D" && sha256sum -c SHA256SUMS)
```

期望分支为 `sim2real/k3-field-replay`。每次验收新建输出目录，已归档的轨迹与报告可留作对照。

### 先跑 #21

```bash
cd /home/nvidia4090/ymbot_grasp
export OMNI_KIT_ACCEPT_EULA=YES
PY=/home/nvidia4090/IsaacLab/.venv/bin/python
D=/home/nvidia4090/real_data/k3_20261120
S="$D/validation/pad_patch_grid/torsional"
OUT=$(mktemp -d "$D/validation/acceptance_XXXXXX")
printf '本次结果目录 %s\n' "$OUT"

"$PY" -B tools/replay_k3_grasp.py "$S/trial_21.usda" "$D/derived/trial_21" "$OUT/dynamics_21.json" \
  --isaaclab --device cuda:0 --physics-hz 360 --cone elliptic --contact-impratio 800 \
  --arm-bias-compensation --sample-contacts --through-release \
  --usd-animation "$OUT/K3_Trial21_Replay.usda"

"$PY" -B tools/check_k3_placement.py "$D/derived" "$OUT/score_21.json" "$OUT/dynamics_21.json" \
  --max-placement-error-mm 20 --max-rest-motion-mm 2 --require-pass
"$PY" -m json.tool "$OUT/score_21.json"
```

看报告顶层的 `"passed": true`。最后三行会写评分、执行通过检查并打印 JSON。如果失败，先查看 `trials → 21 → checks` 中哪一项是 `false`。

### 再跑剩余四例

在同一个终端中继续。若重新登录过，需要重设上一段的 `PY`、`D`、`S` 和 `OUT`，其中 `OUT` 要指向刚才的结果目录。

```bash
for trial in 17 18 19 20; do
  "$PY" -B tools/replay_k3_grasp.py "$S/trial_$trial.usda" "$D/derived/trial_$trial" "$OUT/dynamics_$trial.json" \
    --isaaclab --device cuda:0 --physics-hz 360 --cone elliptic --contact-impratio 800 \
    --arm-bias-compensation --sample-contacts --through-release || break
done

"$PY" -B tools/check_k3_placement.py "$D/derived" "$OUT/score_all.json" \
  "$OUT"/dynamics_{17,18,19,20,21}.json --max-placement-error-mm 20 --max-rest-motion-mm 2 --require-pass
"$PY" -m json.tool "$OUT/score_all.json"
```

## 4. 验收看什么

| 检查项 | 通过条件 |
|---|---|
| #17 抓空 | 复现抓空。未持续离桌 10 cm 达 1 秒。 |
| #18–#21 抬升和保持 | 双指带载，箱底离桌至少 10 cm 连续 1 秒。保持到录制的开爪指令。 |
| #18–#21 放回支撑 | 撤离后最后 0.5 秒，桌面法向载荷中位数至少 0.1 N，箱底距台面不超过 2 mm，地面和机器人接触均低于阈值。 |
| #18–#21 放回精度 | 箱体相对原取件位置的平面偏差不超过 20 mm。撤离后最后一秒，箱体中心最大单轴位移范围不超过 2 mm。 |

4090 按本手册重新跑出的报告保存在 `validation/acceptance_demo/score_all.json`。浮点位置会随 GPU 运行略有变化，验收以通过条件为准。

| 试验 | 预期 | 归档实测回放 |
|---|---|---|
| #17 | 抓空 | 通过 |
| #18 | 夹起并放回 | 通过，偏差 7.57 mm |
| #19 | 夹起并放回 | 通过，偏差 6.23 mm |
| #20 | 夹起并放回 | 通过，偏差 8.19 mm |
| #21 | 夹起并放回 | 通过，偏差 7.17 mm |

**交付验收以 `score_all.json` 的五例结果为准。** 当前工作覆盖归档真机轨迹的 Isaac Lab 物理回放，以及 Isaac Sim 可视检查。后续实物标定要补同一个装载纸箱的重量、夹点宽度、沿长边的平衡点和带尺的固定侧视视频。现用 60 g 为名义箱重。现有约 50 Hz 的反馈是关节位置，归档中的 `velocity` 和 `effort` 记录告警和使能状态。视觉策略训练和上机部署属于后续工作。
