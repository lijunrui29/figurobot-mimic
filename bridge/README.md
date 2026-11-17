# bridge/ — Figurobot HTTP 桥接（mimic 补丁版）

这些文件改编自
[figurobot-console](https://github.com/donghuixin/figurobot-console)（MIT），
为实时模仿加了**常驻串口 worker + 批量写 + 插值 ramp**：

- `robot_bridge.py`
  - `POST /api/joint/write_batch`：一次请求写多个关节。
  - 首次写入时启动设备端常驻 worker；之后每次写只走 stdin 管道，
    实测单次约 **0.03s**（不再每次跑 `adb + python3` 的 ~0.8s）。
  - 可选 `ramp_ms` / `steps`：把"上次目标 → 本次目标"分成多个中间 tick，
    设备端按总时长均匀发完，解决"一秒跳一次"的卡顿。
- `serial_worker.py`
  - 设备端常驻串口进程：打开串口后一直不退出，逐行读 JSON 请求并写帧。
  - 写路径把串口读超时设为 0，ack 不再多等 0.1s。
- `serial_exec.py`
  - 一次性的串口执行脚本，用于读姿态/单帧写。
  - 新增 `SERIAL_FRAME_DELAY_MS`：设备端每帧之间的延时（毫秒）。

> adb shell 会分配 pty；桥接启动 worker 时用 `stty -echo` 关掉回显，
> 否则发过去的 JSON 会被回显成一行，被误当成 ack。

## 运行

```bash
# 先装 adb（Android platform-tools），或 --adb 指定 adb 路径
python robot_bridge.py --port 8888
# 桥接默认连 127.0.0.1:8888；机器人通过 adb 连接
```

`serial_exec.py` / `serial_worker.py` 会在第一次写关节时自动推到设备 `/tmp/`。

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
