"""RTMPose → 机器人手臂模仿：摄像头入口（默认 dry-run，`--live` 才下发）。

数据流（本入口只做装配）：
    capture → pose_estimation.detect_pose → pose_schema.human_adapter
            → mapper.MimicMapper → safety.SafetyGovernor → 画面/CSV
                                                       ↘ robot_link（仅 --live）

**默认行为是 dry-run**：`SafetyGovernor` 照常限位/限速/看门狗，但结果只画在
画面上、写进 CSV，**没有任何网络出口**。只有显式 `--live` 才会通过
`robot_link.RobotLink` 把 safety 允许的 tick 写给机器人；且要求每个启用关节
`calibrated=true`、`robot_neutral_tick` 非空、桥接在线、当前姿态不是 danger。

用法：
    python main_mimic_demo.py                    # 正对机位 cam#1，dry-run
    python mimic_side.py --backend msmf          # 另一个终端：侧视进程
    python main_mimic_demo.py --side-view data/side_stream   # 正面 + 侧视
    python main_mimic_demo.py --list-cameras     # 先确认编号
    python main_mimic_demo.py --camera 2         # 换成 45° 那台
    python main_mimic_demo.py --video data/clip.mp4          # 离线回放
    python main_mimic_demo.py --record data/mimic_log.csv    # 逐帧落盘
    python main_mimic_demo.py --smooth-beta 0.02 --smooth-min-cutoff 0.6
    python main_mimic_demo.py --no-display --video clip.mp4 --record log.csv
    python main_mimic_demo.py --robot-status     # 只读：桥接/关节回读（不下发）
    python main_mimic_demo.py --live             # 真机下发（未标定会直接拒绝）
    python main_mimic_demo.py --live --ramp-ms 800 --ramp-steps 8  # 插值平滑

    python main_mimic_demo.py --selftest         # 合成数据自测（不碰摄像头/模型）

按键（有窗口时）：
    n   采集 5 秒人体中立位（中位数）→ 写回 --config 的 human_neutral
    c   取消本次采集
    r   重置平滑与安全状态（换人 / 手抖得厉害时用）
    s   存快照（需 --snapshot PATH）
    q / ESC  退出

机位说明：
    默认正对机位。正面能测**肩的侧抬**；肘屈是矢状面动作，前臂朝镜头弯时
    会被透视缩短（线段太短 → `geom` → 该关节 N/A），属预期，45° 机位会好些。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# 本目录内的模块直接 import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cv2                                                    # noqa: E402

from capture import CameraCapture, probe_cameras              # noqa: E402
from mapper import MimicMapper                                # noqa: E402
from mimic_calib import NeutralCollector                      # noqa: E402
from mimic_config import (MimicConfigError, get_joint,        # noqa: E402
                          load_config, save_config)
from mimic_io import CsvRecorder, VideoFileSource             # noqa: E402
from mimic_overlay import MimicOverlay, joint_label           # noqa: E402
from mimic_output import FrameRenderer                        # noqa: E402
from pose_estimation import detect_pose                       # noqa: E402
from pose_schema import human_adapter                         # noqa: E402
from robot_link import (DEFAULT_BASE_URL, DEFAULT_TIMEOUT,    # noqa: E402
                        RobotLink, RobotLinkError)
from safety import (ALLOWED_SERVO_IDS, SafetyGovernor,        # noqa: E402
                    SafetyResult)

WINDOW_DRY = "Mimic Demo - dry-run (no robot)"
WINDOW_LIVE = "Mimic Demo - LIVE (robot)"


class LivePreflightError(RuntimeError):
    """`--live` 的前置条件不满足（未标定 / 机器人离线 / 当前姿态危险）。"""


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RTMPose → 机器人手臂模仿（阶段 2：dry-run，不连机器人）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=1,
                     help="摄像头编号（默认 1 = 正对机位；认不准先 --list-cameras）")
    src.add_argument("--video", metavar="PATH",
                     help="离线视频回放（与 --camera 二选一）")
    p.add_argument("--list-cameras", nargs="?", const=6, type=int, metavar="N",
                   help="只枚举 0..N-1 号摄像头（不跑主循环）")

    p.add_argument("--config", default=str(SCRIPT_DIR / "mimic_config.json"),
                   help="映射配置（按 n 采集中立位会写回这个文件）")
    p.add_argument("--record", metavar="CSV",
                   help="把每帧每个关节的读数/目标/限位限速标记写成 CSV")
    p.add_argument("--snapshot", metavar="PATH",
                   help="按 s 保存当前帧到这个路径")
    p.add_argument("--no-display", action="store_true",
                   help="不开窗口（离线批处理/远程跑时用，仍可 --record）")
    p.add_argument("--max-frames", type=int, default=None,
                   help="最多处理多少帧（离线回放调参时用）")
    p.add_argument("--width", type=int, default=None, help="摄像头期望宽度")
    p.add_argument("--height", type=int, default=None, help="摄像头期望高度")
    p.add_argument("--side-view", metavar="DIR", default=None,
                   help="侧视数据目录（mimic_side.py --out 的目录，读最新 "
                        "side_*.json）；启用后正面摄像头强制 DSHOW+MJPG，"
                        "避免两路同时开时抢源")
    p.add_argument("--side-timeout", type=float, default=0.6,
                   help="侧视 JSON 超过多少秒视为过期（秒）")

    p.add_argument("--neutral-seconds", type=float, default=5.0,
                   help="按 n 后采集中立位的时长（秒）")
    p.add_argument("--neutral-min-samples", type=int, default=30,
                   help="每个关节至少采到多少个有效样本才写回")

    p.add_argument("--smooth-off", action="store_true",
                   help="本次运行关掉 One Euro（不改配置文件）")
    p.add_argument("--smooth-beta", type=float, default=None,
                   help="覆盖配置里的 One Euro beta（越大越跟手）")
    p.add_argument("--smooth-min-cutoff", type=float, default=None,
                   help="覆盖配置里的 One Euro min_cutoff（越小静止越稳）")

    p.add_argument("--log-interval", type=float, default=1.0,
                   help="控制台读数打印间隔（秒；0 = 关闭）")

    # ---- 阶段 4：机器人桥接 ----
    p.add_argument("--live", action="store_true",
                   help="真正向机器人下发（要求所有启用关节 calibrated=true；"
                        "桥接需先启动）")
    p.add_argument("--bridge", default=DEFAULT_BASE_URL,
                   help="figurobot 桥接地址（robot_bridge.py）")
    p.add_argument("--bridge-timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="单次桥接请求超时（秒）")
    p.add_argument("--live-hz", type=float, default=4.0,
                   help="下发频率上限（Hz）。桥接优先批量写，实际受 ~0.8s/次限制")
    p.add_argument("--ramp-ms", type=int, default=0,
                   help="批量写插值总时长（毫秒；0 = 不插值，一步到位）。"
                        "建议设成两次下发的间隔，如 800~1200")
    p.add_argument("--ramp-steps", type=int, default=8,
                   help="插值步数（越大越平滑，帧数 = steps × 关节数）")
    p.add_argument("--robot-status", action="store_true",
                   help="只读：打印桥接状态与关节回读后退出（不跑摄像头、不下发）")

    p.add_argument("--selftest", action="store_true",
                   help="合成数据自测（不碰摄像头/模型/机器人）")
    return p


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """把 --smooth-* 覆盖到配置副本上（**不写回文件**）。"""
    out = copy.deepcopy(cfg)
    sm = out["smoothing"]
    if args.smooth_off:
        sm["enabled"] = False
    if args.smooth_beta is not None:
        sm["beta"] = float(args.smooth_beta)
    if args.smooth_min_cutoff is not None:
        sm["min_cutoff"] = float(args.smooth_min_cutoff)
    return out


def build_safety(cfg: dict) -> SafetyGovernor:
    """按配置装配安全层（白名单只能收紧；每关节步进/速度可单独收紧）。"""
    g = cfg["global"]
    gov = SafetyGovernor(
        whitelist=g["whitelist"],
        min_tick=g["tick_min"],
        max_tick=g["tick_max"],
        default_max_step_deg=g["max_step_deg"],
        default_max_speed_deg_s=g["max_speed_deg_s"],
        watchdog_s=g["watchdog_s"],
        max_consecutive_errors=g["max_consecutive_errors"],
    )
    for joint in cfg["joints"]:
        gov.configure_joint(joint["servo_id"],
                            joint.get("max_step_deg"),
                            joint.get("max_speed_deg_s"))
    return gov


def _seed_targets(cfg: dict) -> dict:
    """启动时用 robot_neutral_tick 作为限速起点（还不知道真实位置）。"""
    return {int(j["servo_id"]): int(j["robot_neutral_tick"])
            for j in cfg["joints"]
            if j["enabled"] and j.get("robot_neutral_tick") is not None}


def _read_side_view(path, timeout: float) -> Optional[dict]:
    """读侧视进程发布的最新一帧；缺失/损坏/过期一律返回 None。

    `path` 是**目录**时读 `side_*.json` 里文件名最大的一个（侧视进程每帧
    写一个唯一文件，避免 Windows 单文件覆盖锁）；也兼容传入单个文件。
    None 在 mapper 里会变成 `no_side`（侧视来源的关节 N/A），不会阻塞
    正面来源的关节。过期判断用 `time.monotonic()`，同一台机器跨进程可比。
    """
    if not path:
        return None
    p = Path(path)
    try:
        if p.is_dir():
            files = sorted(p.glob("side_*.json"))
            if not files:
                return None
            p = files[-1]
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = float(data.get("ts"))
    except (OSError, ValueError, TypeError):
        return None
    if time.monotonic() - ts > float(timeout):
        return None
    return data


def _enabled_joints(cfg: dict) -> list:
    return [j for j in cfg["joints"] if j["enabled"]]


def preflight_live(cfg: dict, link) -> dict:
    """`--live` 的硬闸：任何一条不满足就抛 LivePreflightError，绝不"带病下发"。

    检查顺序（先便宜后昂贵）：
        1. 配置：启用关节必须 calibrated=true，且 robot_neutral_tick 非空；
        2. 桥接：/api/status 的 connected 必须为 true；
        3. 机器人：/api/pose 的 safety.level 不能是 danger；
        4. 回读：每个启用关节必须 online 且有 raw tick（限速起点要真值）。
    返回 {'positions': {servo_id: raw tick}, 'safety': dict, 'status': dict}。
    `positions` 直接喂 safety.seed_current()，其余只用于启动日志。
    """
    enabled = _enabled_joints(cfg)
    if not enabled:
        raise LivePreflightError("配置里没有启用的关节，--live 无意义")

    uncal = [f"{j['name']}({j['servo_id']})" for j in enabled
             if not j.get("calibrated")]
    if uncal:
        raise LivePreflightError(
            "以下关节尚未标定（calibrated=false），拒绝 --live："
            + "、".join(uncal)
            + "。先做阶段 3 标定：回读 robot_neutral_tick、验证 sign/scale，"
              "再把对应关节的 calibrated 改成 true")

    no_neutral = [f"{j['name']}({j['servo_id']})" for j in enabled
                  if j.get("robot_neutral_tick") is None]
    if no_neutral:
        raise LivePreflightError(
            "以下关节缺 robot_neutral_tick，拒绝 --live："
            + "、".join(no_neutral)
            + "。阶段 3 Motion_Reset 后从 /api/pose 回读再写入配置")

    status = link.status()
    if not status.get("connected"):
        raise LivePreflightError(
            f"桥接在线但机器人未连接（/api/status: {status}）。"
            "检查 USB / adb devices，再重试")

    pose = link.read_pose(fast=True)
    level = (pose.get("safety") or {}).get("level")
    if level == "danger":
        raise LivePreflightError(
            "机器人当前姿态评估为 danger（有超限/极限位关节），拒绝 --live。"
            f"先人工处理：{(pose.get('safety') or {}).get('issues')}")

    positions = {}
    for key, joint in (pose.get("joints") or {}).items():
        if not isinstance(joint, dict) or not joint.get("online"):
            continue
        if joint.get("raw") is None:
            continue
        try:
            positions[int(key)] = int(joint["raw"])
        except (TypeError, ValueError):
            continue

    missing = [f"{j['name']}({j['servo_id']})" for j in enabled
               if int(j["servo_id"]) not in positions]
    if missing:
        raise LivePreflightError(
            "以下启用关节回读不到位置，拒绝 --live：" + "、".join(missing)
            + "。确认关节在线、串口没被 motion_main 抢占")

    return {
        "positions": {sid: tick for sid, tick in positions.items()
                      if sid in {int(j["servo_id"]) for j in enabled}},
        "safety": pose.get("safety") or {},
        "status": status,
    }


class LiveSender:
    """按频率上限把 safety 允许的目标写给机器人，并把成败回报给 safety。

    只做"什么时候发 + 发完回报"，不做任何过滤/限幅（那是 safety 的事）。
    `send_targets()` 的失败不抛异常，统一走 `safety.notify_result(False)`，
    连续失败由 safety 熔断。

    **`safety.update()` 只在 `due()` 为真的帧调用**：这样限速的 dt 是真实
    的"两次下发间隔"（而不是摄像头帧间隔），单帧 6° 的步进上限才拦得住
    4Hz 下发时一次 11° 的跳变。

    `threaded=True` 时网络写放到后台线程：桥接一次批量写约 0.9s，同步发
    会把画面卡成 ~1 FPS。线程只做 `link.send_targets()`；
    `safety.update()` / `notify_result()` 仍全部在主线程，safety 无并发。
    """

    def __init__(self, link, safety: SafetyGovernor, hz: float,
                 threaded: bool = False, ramp_ms: int = 0,
                 ramp_steps: int = 8) -> None:
        self.link = link
        self.safety = safety
        self.hz = max(float(hz), 1e-6)
        self.period = 1.0 / self.hz
        self.ramp_ms = max(0, int(ramp_ms))
        self.ramp_steps = max(1, int(ramp_steps))
        self.last_send = -1e9
        self.last_result = None
        self.sent_total = 0
        self.fail_total = 0
        self.last_error = None
        # 后台发送：桥接一次批量写约 0.9s，同步发会把画面卡成 1 FPS。
        # 线程只负责网络写；safety.update / notify_result 仍全部在主线程。
        self.threaded = bool(threaded)
        self._lock = threading.Lock()
        self._pending = None
        self._busy = False
        self._result = None
        self._stop = threading.Event()
        self._thread = None

    # ---- 后台线程 ----
    def start(self) -> None:
        if not self.threaded or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="mimic-live-sender")
        self._thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _worker(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                targets = self._pending
                self._pending = None
            if targets is None:
                self._stop.wait(0.005)
                continue
            res = self.link.send_targets(targets, ramp_ms=self.ramp_ms,
                                         steps=self.ramp_steps)
            with self._lock:
                self._result = res
                self._busy = False

    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def poll(self):
        """主线程每帧调用：取走后台发送结果并回报给 safety。"""
        with self._lock:
            res = self._result
            self._result = None
        if res is None:
            return None
        self.last_result = res
        self.sent_total += len(res.sent)
        self.fail_total += len(res.failed)
        self.last_error = res.error
        self.safety.notify_result(res.ok)
        return res

    def submit(self, sres, now: float):
        """主线程调用：把 safety 允许的目标交给后台线程。忙则跳过。"""
        if sres.tripped or sres.holding or not sres.targets:
            self.last_send = float(now)
            return False
        with self._lock:
            if self._busy:
                return False
            self._pending = dict(sres.targets)
            self._busy = True
            self.last_send = float(now)
        return True

    def due(self, now: float) -> bool:
        """这一帧是否到了下发时刻。"""
        return float(now) - self.last_send >= self.period

    def send(self, sres, now: float):
        """同步下发（自测 / 不启用后台线程时用）；返回 LinkResult。"""
        self.last_send = float(now)
        if sres.tripped or sres.holding or not sres.targets:
            return None
        res = self.link.send_targets(sres.targets, ramp_ms=self.ramp_ms,
                                     steps=self.ramp_steps)
        self.last_result = res
        self.sent_total += len(res.sent)
        self.fail_total += len(res.failed)
        self.last_error = res.error
        self.safety.notify_result(res.ok)
        return res

    def status_line(self) -> str:
        if self.last_result is None:
            return f"LIVE {self.hz:g}Hz waiting"
        res = self.last_result
        if res.ok:
            return (f"LIVE sent={len(res.sent)} ok  total={self.sent_total}"
                    f"  {res.elapsed_s * 1000:.0f}ms")
        return (f"LIVE FAIL sent={len(res.sent)} fail={len(res.failed)}"
                f"  {self.last_error or ''}")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    base_cfg = load_config(args.config)
    cfg = apply_overrides(base_cfg, args)
    mapper = MimicMapper(cfg)
    safety = build_safety(cfg)
    overlay = MimicOverlay()
    renderer = FrameRenderer(draw_skeleton=True)
    recorder = CsvRecorder(args.record) if args.record else None
    collector = None
    seeded = False
    frame_no = 0
    last_log = -1e9
    fps_times = []
    sres = SafetyResult()
    side_state = None
    last_side_read = -1e9

    # ---- 阶段 4：只有显式 --live 才建链路；默认 dry-run 一个字节都不发 ----
    link = None
    live_sender = None
    live_positions = {}
    live_status = None
    if args.live:
        if args.video:
            raise LivePreflightError(
                "--live 不能和 --video 一起用：离线回放会按录像驱动真机，"
                "容易做出意料之外的动作。先单独回放看数据")
        link = RobotLink(args.bridge, args.bridge_timeout)
        info = preflight_live(cfg, link)
        live_positions = info["positions"]
        live_status = info["status"]
        safety.seed_current(live_positions, now=time.monotonic())
        seeded = True
        live_sender = LiveSender(link, safety, args.live_hz, threaded=True,
                                 ramp_ms=args.ramp_ms,
                                 ramp_steps=args.ramp_steps)
        live_sender.start()

    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)

    if args.video:
        src = VideoFileSource(args.video)
        source_label = f"video {Path(args.video).name}"
    else:
        if args.side_view:
            # 双摄时必须和侧视进程分开后端：正面 DSHOW+MJPG，侧视默认 MSMF
            src = CameraCapture(index=args.camera, width=args.width,
                                height=args.height,
                                backend=cv2.CAP_DSHOW, fourcc="MJPG")
        else:
            src = CameraCapture(index=args.camera, width=args.width,
                                height=args.height)
        source_label = f"cam#{args.camera}"

    window = WINDOW_LIVE if args.live else WINDOW_DRY
    mode_label = "LIVE (robot)" if args.live else "DRY-RUN (no robot)"
    _print_startup(args, cfg, source_label,
                   live_positions=live_positions, live_status=live_status)

    try:
        try:
            with src:
                while True:
                    frame = src.read()
                    if frame is None:
                        break
                    if args.video:
                        ts = frame_no / max(float(getattr(src, "fps", 30.0)), 1e-3)
                    else:
                        ts = time.monotonic()

                    pose = detect_pose(frame, cache_key="mimic")
                    schema = human_adapter(pose)
                    size = (frame.shape[1], frame.shape[0])
                    # 侧视 JSON 限流到 10Hz，减少目录/文件竞争
                    if args.side_view:
                        if ts - last_side_read >= 0.1:
                            side_state = _read_side_view(args.side_view,
                                                         args.side_timeout)
                            last_side_read = ts
                    else:
                        side_state = None
                    mapping = mapper.update(schema, size, now=ts,
                                            side_measures=side_state)

                    if not seeded:
                        seed = _seed_targets(cfg)
                        if seed:
                            safety.seed_current(seed, now=ts)
                        seeded = True
                    # ---- 阶段 4：仅 --live 时下发（失败走 safety 熔断） ----
                    # 发送在后台线程：主循环不被 ~0.9s 的桥接写卡住，画面保持流畅。
                    # safety.update 仍只在"到下发时刻且后台空闲"时调用，让限速的
                    # dt 等于真实下发间隔；其余帧沿用上一帧的 sres 显示。
                    if live_sender is not None:
                        live_sender.poll()
                        if live_sender.due(ts) and not live_sender.busy():
                            sres = safety.update(mapping["targets"], ts,
                                                 data_valid=pose is not None)
                            live_sender.submit(sres, ts)
                    else:
                        sres = safety.update(mapping["targets"], ts,
                                             data_valid=pose is not None)

                    # ---- 中立位采集 ----
                    if collector is not None:
                        collector.add(mapping["details"], ts)
                        if collector.done(ts):
                            base_cfg, mapper = _finish_neutral(
                                collector, args, base_cfg, cfg)
                            collector = None

                    # ---- 记录 ----
                    if recorder is not None:
                        recorder.write(frame_no, ts, pose is not None,
                                       mapping, sres,
                                       neutral_capture=collector is not None)

                    # ---- 画面 ----
                    if not args.no_display:
                        renderer.draw_skeleton_frame(frame, pose)
                        extra = _extra_lines(collector, ts, live_sender)
                        overlay.draw(frame, mapping, sres,
                                     source_label=source_label,
                                     pose_ok=pose is not None,
                                     fps=_fps(fps_times, ts),
                                     extra_lines=extra,
                                     mode_label=mode_label)

                    # ---- 控制台 ----
                    if args.log_interval > 0 and ts - last_log >= args.log_interval:
                        last_log = ts
                        _log_line(ts, pose, mapping, sres, live_sender)

                    # ---- 窗口/按键 ----
                    if not args.no_display:
                        cv2.imshow(window, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord("q"):
                            break
                        if key == ord("n"):
                            if collector is None:
                                collector = NeutralCollector(
                                    args.neutral_seconds,
                                    args.neutral_min_samples)
                                collector.start(ts)
                                print(f"[neutral] 开始采集 {args.neutral_seconds:g}s："
                                      "站正、双臂自然下垂、肘微屈", flush=True)
                            else:
                                print("[neutral] 已经在采集，按 c 可取消", flush=True)
                        elif key == ord("c") and collector is not None:
                            collector = None
                            print("[neutral] 已取消，未写回配置", flush=True)
                        elif key == ord("r"):
                            mapper.reset()
                            safety.reset()
                            seeded = False
                            print("[mimic] 平滑/安全状态已重置", flush=True)
                        elif key == ord("s") and args.snapshot:
                            cv2.imwrite(args.snapshot, frame)
                            print(f"[snapshot] 已存 {args.snapshot}", flush=True)

                    if args.max_frames is not None and frame_no + 1 >= args.max_frames:
                        break
                    frame_no += 1
        except RuntimeError as exc:
            # FrameSource.open() 失败时抛的；补一句怎么查编号
            print(f"[mimic] 数据源打开失败：{exc}")
            if not args.video:
                print("[mimic] 先跑 `python main_mimic_demo.py --list-cameras` "
                      "看有哪些编号，再用 --camera 指定")
            return
    finally:
        if live_sender is not None:
            live_sender.stop()
        if recorder is not None:
            recorder.close()
            print(f"[record] 已写 {recorder.rows} 行 → {recorder.path}")
        if not args.no_display:
            cv2.destroyAllWindows()


def _finish_neutral(collector: NeutralCollector, args, base_cfg: dict,
                    cfg: dict):
    """收尾：中位数写回配置 → 重建 mapper。返回 (新 base_cfg, 新 mapper)。"""
    names = {int(j["servo_id"]): joint_label(j["servo_id"])
             for j in cfg["joints"]}
    print("[neutral] 采集结果：")
    print(collector.report(names))

    new_cfg, applied, skipped = collector.apply(base_cfg)
    if not applied:
        print("[neutral] 没有关节达到最少样本数，**未写回**配置文件")
        return base_cfg, MimicMapper(cfg)

    save_config(args.config, new_cfg)
    # 把中立位同步到本次运行的 cfg（保留 CLI 覆盖的平滑参数）
    for joint in cfg["joints"]:
        updated = get_joint(new_cfg, int(joint["servo_id"]))
        if updated is not None:
            joint["human_neutral"] = updated["human_neutral"]

    for sid, value in sorted(applied.items()):
        print(f"  {names.get(sid, sid):<7} human_neutral = {value:+.2f}deg")
    for sid, why in sorted(skipped.items()):
        print(f"  {names.get(sid, sid):<7} 跳过（{why}）")
    print(f"[neutral] 已写回 {args.config}（只改 human_neutral）")
    return new_cfg, MimicMapper(cfg)


def _fps(times: list, ts: float, window: int = 30) -> float:
    """最近 N 帧的平均帧率（视频回放时就是文件 fps）。"""
    times.append(ts)
    if len(times) > window:
        del times[:-window]
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 1e-6 else 0.0


def _extra_lines(collector, ts: float, live_sender=None) -> list:
    lines = []
    if collector is not None:
        elapsed = collector.progress(ts) * collector.duration_s
        counts = collector.counts()
        parts = [f"{joint_label(sid)}:{info['n']}"
                 for sid, info in sorted(counts.items())]
        lines.append(f"NEUTRAL {elapsed:.1f}/{collector.duration_s:.1f}s  "
                     + (" ".join(parts) if parts else "no samples yet"))
    if live_sender is not None:
        lines.append(live_sender.status_line())
    return lines


def _log_line(ts: float, pose, mapping: dict, sres, live_sender=None) -> None:
    parts = []
    for sid, detail in mapping["details"].items():
        if not detail.get("enabled"):
            continue
        human = "--" if detail.get("human_angle") is None else \
            f"{detail['human_angle']:+.1f}"
        tick = "--" if detail.get("tick") is None else str(detail["tick"])
        flags = []
        if detail.get("reason"):
            flags.append(detail["reason"])
        if sres.rate_limited.get(int(sid)):
            flags.append("RATE")
        flag = ("/" + ",".join(flags)) if flags else ""
        parts.append(f"{joint_label(sid)}={human}deg->{tick}{flag}")
    state = "OK" if pose is not None else "NONE"
    if sres.tripped:
        state += " TRIPPED"
    elif sres.holding:
        state += " HOLD"
    live = ""
    if live_sender is not None:
        live = f"  {live_sender.status_line()}"
    print(f"[mimic] t={ts:7.1f}s pose={state} " + " ".join(parts) + live,
          flush=True)


def _print_startup(args, cfg: dict, source_label: str,
                   live_positions: dict = None, live_status: dict = None) -> None:
    g = cfg["global"]
    if args.live:
        print("[mimic] 阶段 4：**LIVE** —— safety 允许的 tick 会经桥接下发到机器人")
        extra = ""
        if live_status:
            extra = (f"（serial={live_status.get('serial')}）"
                     if live_status.get("serial") else "")
        print(f"[mimic] 桥接：{args.bridge}{extra}；"
              f"下发上限 {args.live_hz:g}Hz（优先批量写，桥接约 0.8s/次）")
        if args.ramp_ms > 0:
            print(f"[mimic] 插值：ramp {args.ramp_ms}ms / {args.ramp_steps} 步"
                  "（目标铺成连续轨迹）")
        else:
            print("[mimic] 插值：关闭（--ramp-ms 0；机器人会一步到位）")
    else:
        print("[mimic] 阶段 2/3：**dry-run** —— 只算目标和限幅，不发任何指令")
    print(f"[mimic] 配置：{args.config}")
    print(f"[mimic] 数据源：{source_label}；机位：{cfg.get('mount', 'unknown')}")
    if getattr(args, "side_view", None):
        print(f"[mimic] 侧视：{args.side_view}（读最新 side_*.json，"
              f"超时 {args.side_timeout:g}s；先启动 `python mimic_side.py`）")
    names = ", ".join(f"{j['name']}({j['servo_id']})" for j in cfg["joints"]
                      if j["enabled"])
    print(f"[mimic] 启用关节：{names}")
    print(f"[mimic] 有效白名单：{sorted(g['whitelist'])}"
          f"（硬编码 {sorted(ALLOWED_SERVO_IDS)} 的子集）")
    uncal = [j["name"] for j in cfg["joints"]
             if j["enabled"] and not j.get("calibrated")]
    if uncal:
        print(f"[mimic] 尚未标定：{'、'.join(uncal)}"
              "（sign/neutral/scale 是占位值，tick 只用于看趋势）")
    if args.live and live_positions:
        print("[mimic] 限速起点（回读真值）：")
        for joint in cfg["joints"]:
            sid = int(joint["servo_id"])
            if not joint["enabled"]:
                continue
            raw = live_positions.get(sid)
            neutral = joint.get("robot_neutral_tick")
            delta = ("--" if raw is None or neutral is None
                     else f"{raw - int(neutral):+d}tick")
            print(f"    {joint['name']:<7} raw={raw}  neutral={neutral}  "
                  f"差 {delta}")
    print(f"[mimic] 安全：tick {g['tick_min']}-{g['tick_max']}，"
          f"单帧 {g['max_step_deg']:g}°，{g['max_speed_deg_s']:g}°/s，"
          f"看门狗 {g['watchdog_s']:g}s")
    print(f"[mimic] 按键：n=采集中立位 {args.neutral_seconds:g}s  "
          f"r=重置  s=快照  q/ESC=退出"
          + ("  （--no-display：无按键）" if args.no_display else ""))


# ---------------------------------------------------------------------------
# 自测：合成 schema 把 mapper → safety → overlay/CSV/中立位 走一遍
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _synthetic_schema(px: dict, size: tuple = (640, 480)) -> dict:
    w, h = size
    landmarks = [(0.5, 0.5, 0.0, 0.0) for _ in range(17)]
    for pid, (x, y) in px.items():
        landmarks[pid] = (x / w, y / h, 0.0, 1.0)
    return human_adapter({"landmarks": landmarks, "image_size": size})


def selftest() -> None:
    import tempfile
    import numpy as np

    cfg = copy.deepcopy(load_config(str(SCRIPT_DIR / "mimic_config.json")))
    # 自测不依赖用户当前采集的 human_neutral / 标定 sign（它们会被 n 键和
    # calibrate_robot 改写）；固定成一组已知值，断言才稳定。
    test_signs = {3: 1, 8: -1, 6: 1, 13: 1}
    for joint in cfg["joints"]:
        joint["human_neutral"] = 0.0
        sid = int(joint["servo_id"])
        if sid in test_signs:
            joint["sign"] = test_signs[sid]
    mapper = MimicMapper(cfg)
    safety = build_safety(cfg)
    seed = _seed_targets(cfg)
    _assert(len(seed) == 4, f"expected 4 seeded joints, got {seed}")
    neutral = {3: 1981, 6: 2138, 8: 2133, 13: 1178}

    # 正对镜头：人的左肩在画面右；双臂外展到水平、肘伸直
    px = {5: (400, 100), 6: (240, 100), 7: (520, 100), 8: (120, 100),
          9: (620, 100), 10: (20, 100)}
    # 侧视：左臂向前抬 90°（侧视里向前为负角），供 13 号 flexion 源
    side = {"online": True, "arms": {
        "L": {"valid_arm": True, "abduction_signed": -90.0,
              "reason_arm": None, "valid_elbow": True, "elbow_flex": 0.0,
              "reason_elbow": None},
        "R": {"valid_arm": True, "abduction_signed": 0.0,
              "reason_arm": None, "valid_elbow": True, "elbow_flex": 0.0,
              "reason_elbow": None},
    }}
    mapping = mapper.update(_synthetic_schema(px), (640, 480), now=0.0,
                            side_measures=side)

    # 左外展 13：+90° × sign=-1，clamp 到 -40° → 1178 - 455 = 723
    # 右外展 6 ：-90° × sign=+1，clamp 到 -40° → 2138 - 455 = 1683
    _assert(mapping["targets"][13] == 723, f"L abd tick={mapping['targets'][13]}")
    _assert(mapping["targets"][6] == 1683, f"R abd tick={mapping['targets'][6]}")
    _assert(mapping["targets"][3] == 1981, f"L elbow tick={mapping['targets'][3]}")
    _assert(mapping["targets"][8] == 2133, f"R elbow tick={mapping['targets'][8]}")
    _assert(mapping["details"][13]["clamped"], "shoulder should be angle-clamped")

    # safety：seed 在 -1s，第一次下发被 6°/帧 限住
    safety.seed_current(seed, now=-1.0)
    res = safety.update(mapping["targets"], now=0.0, data_valid=True)
    _assert(res.targets[13] == neutral[13] - int(round(6.0 * 11.375)),
            f"safety step-limited tick={res.targets[13]}")
    _assert(res.rate_limited[13], "rate_limited flag")

    # 看门狗：0.6s 无有效数据 → holding
    res_hold = safety.update(mapping["targets"], now=0.7, data_valid=False)
    _assert(res_hold.holding and not res_hold.targets, "watchdog should hold")

    # 没人 → 全 None
    empty = mapper.update(human_adapter(None), (640, 480), now=1.0)
    _assert(all(v is None for v in empty["targets"].values()),
            "no pose must map to None")
    _assert(all(d["reason"] == "no_pose" for d in empty["details"].values()
                if d["enabled"]), "no pose reason")

    # overlay 能画（不校验像素内容，只要求不抛异常）
    from mimic_overlay import MimicOverlay
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    MimicOverlay().draw(frame, mapping, res, source_label="selftest",
                        pose_ok=True, fps=30.0,
                        extra_lines=["NEUTRAL 1.0/5.0s  L-shAbd:30"])
    _assert(int(frame.sum()) > 0, "overlay drew nothing")

    with tempfile.TemporaryDirectory() as td:
        # CSV：一行 = 一帧 × 一个关节（模板 4 个关节）
        cpath = str(Path(td) / "log.csv")
        with CsvRecorder(cpath) as rec:
            rec.write(0, 0.0, True, mapping, res)
        import csv
        with open(cpath, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        _assert(len(rows) == len(cfg["joints"]), f"csv rows={len(rows)}")
        _assert(rows[0]["safety_tick"] != "", "safety tick should be recorded")

        # 中立位采集 → 写回 human_neutral → 保存/加载
        collector = NeutralCollector(duration_s=1.0, min_samples=10)
        collector.start(0.0)
        for i in range(20):
            collector.add(mapping["details"], now=i * 0.05)
        _assert(collector.done(1.0), "collector should finish")
        new_cfg, applied, skipped = collector.apply(cfg)
        _assert(set(applied) == {3, 6, 8, 13}, f"applied={applied}")
        _assert(abs(applied[13] + 90.0) < 0.1, f"L flex neutral={applied[13]}")
        _assert(abs(applied[6] + 90.0) < 0.1, f"R abd neutral={applied[6]}")
        _assert(abs(applied[3]) < 0.1, f"L elbow neutral={applied[3]}")
        _assert(abs(applied[8]) < 0.1, f"R elbow neutral={applied[8]}")
        out_path = str(Path(td) / "cfg.json")
        save_config(out_path, new_cfg)
        back = load_config(out_path)
        _assert(abs(get_joint(back, 13)["human_neutral"] + 90.0) < 0.1,
                "human_neutral should survive save/load")

    # ---- 阶段 4 软件侧：--live 的硬闸与发送节流 ----
    from robot_link import LinkResult

    class _FakeLink:
        def __init__(self, positions=None, connected=True, level="safe"):
            self.positions = (positions if positions is not None
                              else dict(neutral))
            self.connected = connected
            self.level = level
            self.batches = []

        def status(self):
            return {"connected": self.connected, "serial": "FAKE"}

        def read_pose(self, fast=True):
            return {
                "ok": True,
                "joints": {str(sid): {"online": True, "raw": tick}
                           for sid, tick in self.positions.items()},
                "safety": {"level": self.level},
            }

        def send_targets(self, targets, ramp_ms=0, steps=1):
            self.batches.append(dict(targets))
            self.last_ramp = (ramp_ms, steps)
            return LinkResult(ok=True, sent=dict(targets), elapsed_s=0.001)

    # 1) 未标定配置必须拒绝 --live（模板本身已标定，这里构造一份未标定的）
    uncal = copy.deepcopy(cfg)
    for joint in uncal["joints"]:
        joint["calibrated"] = False
    try:
        preflight_live(uncal, _FakeLink())
    except LivePreflightError as exc:
        _assert("未标定" in str(exc), f"unexpected message: {exc}")
    else:
        raise AssertionError("uncalibrated config must be rejected by --live")

    # 2) 已标定配置通过，并把回读位置作为限速起点
    cal = copy.deepcopy(cfg)
    info = preflight_live(cal, _FakeLink())
    _assert(info["positions"] == neutral,
            f"positions={info['positions']}")
    _assert(info["status"]["serial"] == "FAKE", "status should be passed through")

    # 3) 当前姿态 danger / 关键关节回读不到 → 拒绝
    for bad, needle in ((_FakeLink(level="danger"), "danger"),
                        (_FakeLink(positions={3: 1981, 6: 2138, 8: 2133}),
                         "回读不到")):
        try:
            preflight_live(cal, bad)
        except LivePreflightError as exc:
            _assert(needle in str(exc), f"expected {needle!r} in {exc}")
        else:
            raise AssertionError(f"preflight should reject: {needle}")

    # 4) LiveSender：按频率上限节流，并把成功回报给 safety
    link = _FakeLink()
    gov = build_safety(cal)
    gov.seed_current(neutral, now=0.0)
    sender = LiveSender(link, gov, hz=10.0)
    _assert(sender.due(0.0), "first send should be due")
    small = {3: 2050, 6: 2138, 8: 2133, 13: 1178}
    sres = gov.update(small, now=0.0)
    _assert(sender.send(sres, 0.0) is not None, "first send should pass")
    _assert(not sender.due(0.05), "rate limit should skip")
    _assert(len(link.batches) == 1, f"batches={link.batches}")
    _assert(sender.due(0.11), "send should resume after period")
    sres = gov.update(small, now=0.11)
    _assert(sender.send(sres, 0.11) is not None, "second send should pass")
    _assert(gov.consecutive_errors == 0, "successful send should not count errors")

    # 5) 4Hz 下发时 safety 的 dt 是 0.25s：限速取 min(6°, 45°/s×0.25s) = 6°，
    #    不能因为是"一帧"就只走 45°/s×1/30 的 1.5°
    gov4 = build_safety(cal)
    gov4.seed_current(neutral, now=0.0)
    sres4 = gov4.update({3: 2500, 6: 2138, 8: 2133, 13: 1178}, now=0.25)
    _assert(sres4.targets[3] == neutral[3] + int(round(6.0 * 11.375)),
            f"4Hz step should be 6deg, got {sres4.targets[3]}")

    # 6) 后台发送线程：submit 后不阻塞主线程，poll 能取回结果
    gov_t = build_safety(cal)
    gov_t.seed_current(neutral, now=0.0)
    link_t = _FakeLink()
    sender_t = LiveSender(link_t, gov_t, hz=10.0, threaded=True)
    sender_t.start()
    sres_t = gov_t.update(small, now=0.0)
    _assert(sender_t.submit(sres_t, 0.0), "submit should be accepted")
    deadline = time.monotonic() + 1.0
    while sender_t.busy() and time.monotonic() < deadline:
        time.sleep(0.005)
    res_t = sender_t.poll()
    _assert(res_t is not None and res_t.ok, f"threaded result={res_t}")
    _assert(len(link_t.batches) == 1, f"threaded batches={link_t.batches}")
    sender_t.stop()

    # 7) ramp 参数透传到 send_targets
    link_r = _FakeLink()
    gov_r = build_safety(cal)
    gov_r.seed_current(neutral, now=0.0)
    sender_r = LiveSender(link_r, gov_r, hz=10.0, ramp_ms=800, ramp_steps=8)
    sres_r = gov_r.update(small, now=0.0)
    _assert(sender_r.send(sres_r, 0.0) is not None, "ramp send should pass")
    _assert(link_r.last_ramp == (800, 8), f"ramp={link_r.last_ramp}")

    print("  selftest_main_mimic: OK")


def print_robot_status(args: argparse.Namespace) -> None:
    """`--robot-status`：只读打印桥接 / 姿态 / zero / 配置中立位，不下发。

    这是"机器人到底连上没有"和阶段 3 回读中立位的第一站；不需要摄像头，
    也不会让任何关节动。
    """
    cfg = load_config(args.config)
    # 完整回读（fast=False）实测约 5~6s，状态检查给足超时；不沿用 live 的 3s
    link = RobotLink(args.bridge, max(float(args.bridge_timeout), 10.0))
    status = link.status()
    print(f"[robot] 桥接 {args.bridge}：connected={status.get('connected')} "
          f"serial={status.get('serial')} "
          f"protocol={status.get('protocol')}")
    if not status.get("connected"):
        print("[robot] 机器人未连接：检查 USB、`adb devices`、控制盒供电")
        return

    pose = link.read_pose(fast=False)
    safety = pose.get("safety") or {}
    print(f"[robot] safety：{safety.get('level')} "
          f"online={safety.get('online_count')}/{safety.get('total')}"
          + (f" issues={safety.get('issues')}" if safety.get("issues") else ""))
    zeros = {int(k): int(v) for k, v in (pose.get("zero_values") or {}).items()}
    joints = pose.get("joints") or {}

    print(f"[robot] {'id':>2} {'name':<7}{'on':>3}{'raw':>6}{'angle':>8}"
          f"{'zero':>6}{'neutral':>8}{'raw-zero':>9}  cfg")
    for key in sorted(joints, key=lambda k: int(k)):
        sid = int(key)
        joint = joints[key]
        cfg_joint = get_joint(cfg, sid)
        name = (cfg_joint or {}).get("name", "")
        raw = joint.get("raw")
        zero = zeros.get(sid)
        delta = ("--" if raw is None or zero is None else f"{raw - zero:+d}")
        neutral = (cfg_joint or {}).get("robot_neutral_tick")
        flags = []
        if cfg_joint is not None:
            if cfg_joint.get("enabled"):
                flags.append("enabled")
            if not cfg_joint.get("calibrated"):
                flags.append("UNCAL")
        raw_s = "--" if raw is None else str(raw)
        zero_s = "--" if zero is None else str(zero)
        neutral_s = "--" if neutral is None else str(neutral)
        print(f"[robot] {sid:>2} {name:<7}{'Y' if joint.get('online') else 'N':>3}"
              f"{raw_s:>6}{_fmt_angle(joint.get('angle')):>8}"
              f"{zero_s:>6}{neutral_s:>8}{delta:>9}  "
              + (",".join(flags) if flags else ""))

    print("[robot] 提示：neutral 是 mimic_config.json 里的 robot_neutral_tick"
          "（阶段 3 由 Motion_Reset 后回读写入）；--robot-status 只读，不动机器人")


def _fmt_angle(value) -> str:
    return "--" if value is None else f"{float(value):.1f}"


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        print("main_mimic_demo selftest: ALL PASSED")
        return

    if args.robot_status and args.live:
        print("[mimic] --robot-status 是只读检查，不能和 --live 一起用")
        sys.exit(2)
    if args.live_hz <= 0:
        print("[mimic] --live-hz 必须大于 0")
        sys.exit(2)
    if args.ramp_ms < 0:
        print("[mimic] --ramp-ms 不能为负")
        sys.exit(2)
    if args.ramp_steps < 1:
        print("[mimic] --ramp-steps 必须 >= 1")
        sys.exit(2)
    if args.bridge_timeout <= 0:
        print("[mimic] --bridge-timeout 必须大于 0")
        sys.exit(2)
    if args.side_timeout <= 0:
        print("[mimic] --side-timeout 必须大于 0")
        sys.exit(2)

    if args.list_cameras is not None:
        print(f"[cameras] 枚举 0..{args.list_cameras - 1}：")
        for row in probe_cameras(args.list_cameras):
            shape = row["shape"]
            size = f"{shape[1]}x{shape[0]}" if shape else "--"
            print(f"  #{row['index']}: {'可用' if row['opened'] else '不可用':<4} {size}")
        print("[cameras] #1 传统上是正对机位、#2 是扩展坞上的 45° 前侧"
              "（换插口会变，看画面确认）")
        return

    if args.robot_status:
        try:
            print_robot_status(args)
        except RobotLinkError as exc:
            print(f"[robot] {exc}")
            sys.exit(2)
        return

    try:
        run_pipeline(args)
    except LivePreflightError as exc:
        print(f"[mimic] --live 前置检查未通过：{exc}")
        sys.exit(2)
    except RobotLinkError as exc:
        print(f"[mimic] 机器人桥接错误：{exc}")
        sys.exit(2)
    except MimicConfigError as exc:
        print(f"[mimic] 配置错误：{exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
