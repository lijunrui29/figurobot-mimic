# bridge/ — Figurobot HTTP 桥接（mimic 补丁版）

这两个文件来自
[figurobot-console](https://github.com/donghuixin/figurobot-console)（MIT），
为实时模仿加了**批量写 + 插值 ramp**：

- `robot_bridge.py`
  - `POST /api/joint/write_batch`：一次 adb 调用写多个关节。
  - 可选 `ramp_ms` / `steps`：把"上次目标 → 本次目标"分成多个中间 tick，
    设备端按总时长均匀发完，解决"一秒跳一次"的卡顿。
- `serial_exec.py`
  - 新增 `SERIAL_FRAME_DELAY_MS`：设备端每帧之间的延时（毫秒）。

## 运行

```bash
# 先装 adb（Android platform-tools），或 --adb 指定 adb 路径
python robot_bridge.py --port 8888
# 桥接默认连 127.0.0.1:8888；机器人通过 adb 连接
```

`serial_exec.py` 会在第一次写关节时自动推到设备 `/tmp/`。

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 连接状态 |
| POST | `/api/pose` | 读全部关节姿态（含 safety/zero_values） |
| POST | `/api/joint/write` | 写单个关节 `{"id":3,"position":2000}` |
| POST | `/api/joint/write_batch` | 批量写 `{"targets":{"3":2000,"6":2100},"ramp_ms":800,"steps":8}` |
| POST | `/api/joint/read` | 读单个关节 |
| POST | `/api/joint/enable` | 扭矩使能/失能 |
| POST | `/api/reset` | Motion_Reset（会让机器人动） |

## 自测

```bash
python robot_bridge.py --selftest
```

不碰硬件，只验证批量写/ramp 的帧数与间隔。
