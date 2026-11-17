# figurobot-mimic

Figurobot 桌面人形机器人的动作模仿demo：两台摄像头（正面 + 侧面）→
RTMPose 关键点 → 人体动作角度 → 映射成舵机目标 → 经 HTTP 桥接下发。

> 这是一个独立仓库，从 `rtm_action_demo` 的 mimic 分支迁移出来。
> 动作识别那条线留在 `rtm_action_demo`；这里只做机器人模仿。

## 数据流

```
cam1 正面 (DSHOW+MJPG) ─┐
                        ├─→ RTMPose ─→ human_adapter ─→ mapper ─→ safety ─→ bridge ─→ 机器人
cam2 侧面 (MSMF) ───────┘        （侧视进程单独跑，发布 JSON 目录）
```

- **正面相机**：负责正面可见的肩臂角度（具体关节由 `mimic_config.json` 的 `source` 决定）。
- **侧面相机**：负责侧面可见的肩臂角度（`upper_arm_flexion_signed`）。
- **侧视进程**（`mimic_side.py`）单独开相机 2，把测量写进 `data/side_stream/`；
  主进程按最新一帧读取。两路摄像头必须跑在**两个独立进程**里（同进程双开会抢源）。
- **安全层**（`safety.py`）负责白名单、软限位、限速、看门狗、错误熔断。
- **桥接**（`bridge/`）负责把 tick 写给机器人；支持批量写和插值 ramp。

## 舵机映射

具体映射写在 `mimic_config.json`：每个关节的 `servo_id`、`source`、`sign`、
`robot_neutral_tick`。换一台机器人必须重新跑 `calibrate_robot.py` 标定。

控制台 HTML 里的 `JOINT_NAMES`和实测不一定一致。

## 目录

```
main_mimic_demo.py   主入口（默认 dry-run；--live 才下发）
mimic_side.py        侧视进程（相机 2 → 测量 → data/side_stream/）
mimic_overlay.py     画面叠加层
mimic_calib.py       按 n 采集人体中立位
mimic_config.py      配置加载/校验
mimic_config.json    映射配置
mimic_io.py          视频回放 / CSV 记录
mimic_geometry.py    几何门控常量（与动作线 features.py 同口径）
mapper.py            人体测量 → 舵机 tick
safety.py            白名单 / 限位 / 限速 / 看门狗 / 熔断
robot_link.py        桥接 HTTP 客户端（批量写 / ramp / 回退）
calibrate_robot.py   机器人侧标定（reset / neutral / probe / apply）
one_euro.py          One Euro 平滑
capture.py           摄像头输入
pose_estimation.py   RTMPose 推理
pose_schema.py       COCO17 / 机器人 27 DOF 规范 schema
mimic_output.py      骨架绘制（从动作线 output.py 精简）
bridge/              Figurobot HTTP 桥接（带批量写 + ramp 的补丁版）
data/                本地输出（gitignore；只留 .gitkeep）
```

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

需要 Python 3.11（与 `rtm_action_demo` 相同）。桥接还需要 **adb** /
Android platform-tools。

### 模型

`models/` **不在仓库里**（约 21MB）。模型是 **RTMPose-s**（body7, 256×192）
的 ONNX 导出，原始权重：

```
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth
```

用 **mmdeploy** 导出成 SDK 包（`deploy.json / pipeline.json / end2end.onnx`），
对应 mmpose 配置
`projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py`。
把整个包解压到 `models/` 下任意位置即可——代码用 `rglob('end2end.onnx')`
自动查找。

## 运行

```bash
# 终端 1：桥接
cd bridge
python robot_bridge.py --port 8888

# 终端 2：侧视进程（相机 2）
python mimic_side.py --backend msmf

# 终端 3：dry-run（不动机器人）
python main_mimic_demo.py --side-view data/side_stream
```

窗口里按 **`n`** 采集 5 秒人体中立位（同时采正面/侧面），再测试动作。

确认方向无误后 live：

```bash
python main_mimic_demo.py --live --side-view data/side_stream \
    --live-hz 8 --ramp-ms 0
```

- 桥接现在用设备端**常驻串口 worker**，单次写入约 **30ms**；`--live-hz` 控制提交频率。
- `--ramp-ms`：可选，把一次目标铺成连续轨迹；0 = 不插值。舵机有 3~4° 死区，
  步长小于死区时可能不动，所以先试 8Hz、每步约 5.6°；需要更平滑再试 `--ramp-ms 100`。
- `--ramp-steps`：插值步数；越大帧数越多。
- **退出回位**：默认 `--park-on-exit`，按 `q`/关窗时把机器人缓慢送回配置里的中立 tick
  （`--park-ramp-ms 4000`，约 45°/s）；`--no-park-on-exit` 关闭。
- 如果动作太慢：调 `mimic_config.json` 的 `global.max_step_deg` / `max_speed_deg_s`。

## 相机

- 编号认不准先跑 `python main_mimic_demo.py --list-cameras`。
- 正面相机建议 **DSHOW + MJPG**；侧视相机建议 **MSMF**。两路同时开时
  如果正面打不开，把其中一台换到另一个 USB 控制器/端口。
- 机位要求：正面能看到头/双肩/肘/手；侧面能看到整个左臂。背光会让关键点不稳。

## 标定

```bash
python calibrate_robot.py status
python calibrate_robot.py reset --yes          # 会让机器人动
python calibrate_robot.py neutral              # 保存机器人中立位快照
python calibrate_robot.py probe --joint 3 --delta 60 --yes
python calibrate_robot.py probe --joint 8 --delta 60 --yes
python calibrate_robot.py probe --joint 6 --delta 60 --yes
python calibrate_robot.py probe --joint 13 --delta 60 --yes
python calibrate_robot.py apply --sign 3=+1 --sign 8=-1 --sign 6=+1 --sign 13=+1
```

人体中立位（`human_neutral`）在运行时按 `n` 采集，不在标定工具里。

## 已知限制

- 单相机 2D 投影有歧义：前后抬和侧向抬在某些视角下会互相混淆；遮挡也会让读数不可靠。
- 都是 2D 像面代理量，不是真实三维角。
- 桥接写已改为设备端常驻 worker，单次约 30ms；剩余卡顿主要来自关键点抖动、
  **完全折叠时腕点丢失**（`no_point`/`out_of_frame`），以及舵机 3~4° 死区。
- 肘部角度限幅在 `mimic_config.json` 的 `min_deg` / `max_deg`；当前左 140°、右 -180°，
  是按全局 tick 放的，机械极限必须现场确认后按需收紧。
- `models/` 和 `data/` 都不提交；`data/` 里可能有实拍画面，默认被 gitignore。
