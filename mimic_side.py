"""侧视辅助进程：一台摄像头 → RTMPose → 上臂前/后抬测量 → JSON 发布。

为什么单独一个进程：这台机器上两路 USB 摄像头**不能在同一个进程里**
同时开（MSMF 会抢源，第二路抓不到帧）。正面主入口用 DSHOW+MJPG 开 1 号，
本进程用默认 MSMF 开 2 号，两个进程各自持有一台摄像头。

本进程只做**测量**，不碰机器人、不做映射：
    相机 → pose_estimation.detect_pose → pose_schema.human_adapter
         → mapper.HumanMeasurement → data/side_stream/side_*.json（每帧唯一文件）

JSON 格式（主入口按 `ts` 判断是否过期）：
    {"ts": <monotonic>, "online": bool, "reason": str|null,
     "frame_size": [w, h],
     "arms": {"L": {...}, "R": {...}}}

侧视里 `abduction_signed`（肩→肘 相对向下轴的有符号角）看到的是**前/后抬**，
主入口把它当作 `upper_arm_flexion_signed` 使用。符号由现场标定决定。

用法：
    python mimic_side.py                       # 2 号摄像头，写 data/side_stream/
    python mimic_side.py --camera 2 --out data/side_stream
    python mimic_side.py --display             # 同时开预览窗口（调试用）
    python mimic_side.py --selftest            # 合成数据自测，不碰摄像头/模型
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cv2                                                # noqa: E402
from capture import CameraCapture                         # noqa: E402
from mapper import HumanMeasurement                       # noqa: E402
from mimic_config import MimicConfigError, load_config    # noqa: E402
from pose_estimation import detect_pose                   # noqa: E402
from pose_schema import human_adapter                     # noqa: E402

DEFAULT_OUT = SCRIPT_DIR / "data" / "side_stream"
DEFAULT_CAMERA = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="侧视辅助进程（测量上臂前/后抬，发布 JSON）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--camera", type=int, default=DEFAULT_CAMERA,
                   help="侧视摄像头编号")
    p.add_argument("--backend", choices=("msmf", "dshow", "auto"),
                   default="msmf",
                   help="摄像头后端；本机的侧视 2 号是 MSMF-only，默认 msmf")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="发布目录（每帧写一个唯一 JSON；主入口读最新一个）")
    p.add_argument("--config", default=str(SCRIPT_DIR / "mimic_config.json"),
                   help="映射配置（只取门控参数）")
    p.add_argument("--width", type=int, default=None,
                   help="期望宽度（默认不设置；MSMF 设分辨率可能卡几十秒）")
    p.add_argument("--height", type=int, default=None,
                   help="期望高度（默认不设置）")
    p.add_argument("--log-interval", type=float, default=1.0,
                   help="控制台读数间隔（秒；0 = 关闭）")
    p.add_argument("--display", action="store_true",
                   help="开预览窗口（调试；默认无窗口）")
    p.add_argument("--selftest", action="store_true",
                   help="合成数据自测（不碰摄像头/模型）")
    return p


def _trim_arms(measures: dict) -> dict:
    """只保留主入口需要的字段，避免 JSON 里塞像素坐标。"""
    out = {}
    for side in ("L", "R"):
        arm = measures["arms"][side]
        out[side] = {
            "abduction_signed": arm.get("abduction_signed"),
            "elbow_flex": arm.get("elbow_flex"),
            "valid_arm": bool(arm.get("valid_arm")),
            "valid_elbow": bool(arm.get("valid_elbow")),
            "reason_arm": arm.get("reason_arm"),
            "reason_elbow": arm.get("reason_elbow"),
        }
    return out


def _write_frame(out_dir: Path, payload: dict, seq: int,
                 frame_no: int) -> None:
    """每帧写一个唯一文件（纳秒时间戳命名），主入口读最新一个。

    单文件 `os.replace` 在 Windows 上会被读者短暂锁住并抛
    `PermissionError`；唯一文件名不覆盖任何现有文件，因此没有竞争。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir / f"side_{seq:020d}.json"
    tmp = out_dir / f".tmp_{seq:020d}"
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, name)
    # 保留最近 60 个，避免目录无限膨胀
    if frame_no % 50 == 0:
        files = sorted(out_dir.glob("side_*.json"))
        for old in files[:-60]:
            try:
                old.unlink()
            except OSError:
                pass


def _fmt(value) -> str:
    return "--" if value is None else f"{float(value):+6.1f}"


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    g = cfg["global"]
    meas = HumanMeasurement(visibility_min=g["visibility_min"],
                            edge_margin=g["edge_margin"],
                            min_segment_ratio=g["min_segment_ratio"])
    backend = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW,
               "auto": None}[args.backend]
    cap = CameraCapture(index=args.camera, width=args.width,
                        height=args.height, backend=backend)
    if not cap.open():
        print(f"[side] 打不开摄像头 #{args.camera}；先跑 "
              f"`python main_mimic_demo.py --list-cameras` 看编号")
        sys.exit(2)

    out_path = Path(args.out)
    print(f"[side] 侧视进程：cam#{args.camera} → {out_path}/side_*.json")
    print("[side] 只测量、不控制机器人；主入口用 --side-view 读取")

    window = "mimic side view"
    if args.display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_log = -1e9
    frame_no = 0
    try:
        with cap:
            while True:
                frame = cap.read()
                if frame is None:
                    break
                pose = detect_pose(frame, cache_key="mimic-side")
                schema = human_adapter(pose)
                size = (frame.shape[1], frame.shape[0])
                measures = meas.update(schema, size)
                now = time.monotonic()
                _write_frame(out_path, {
                    "ts": now,
                    "online": bool(measures["online"]),
                    "reason": measures["reason"],
                    "frame_size": list(measures["frame_size"] or size),
                    "arms": _trim_arms(measures),
                }, time.time_ns(), frame_no)
                frame_no += 1
                if args.log_interval > 0 and now - last_log >= args.log_interval:
                    last_log = now
                    print(f"[side] t={now:8.1f}s online={measures['online']} "
                          f"L={_fmt(measures['arms']['L']['abduction_signed'])}"
                          f"({measures['arms']['L']['reason_arm'] or 'ok'}) "
                          f"R={_fmt(measures['arms']['R']['abduction_signed'])}"
                          f"({measures['arms']['R']['reason_arm'] or 'ok'})",
                          flush=True)
                if args.display:
                    from mimic_output import FrameRenderer
                    FrameRenderer.draw_skeleton_frame(frame, pose)
                    cv2.putText(frame, "SIDE VIEW (flexion source)",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow(window, frame)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
    finally:
        if args.display:
            cv2.destroyAllWindows()


def _synthetic_schema(px: dict, size=(640, 480)) -> dict:
    w, h = size
    landmarks = [(0.5, 0.5, 0.0, 0.0) for _ in range(17)]
    for pid, (x, y) in px.items():
        landmarks[pid] = (x / w, y / h, 0.0, 1.0)
    return human_adapter({"landmarks": landmarks, "image_size": size})


def selftest() -> None:
    """合成侧视姿态：左臂向前抬 45°，验证测量与 JSON 序列化。"""
    import math
    # 侧视：肩 5 在 (200,100)，肘 7 向前抬到 45°（图像左 = 前方）
    length = 120
    el = (200 - length * math.sin(math.radians(45)),
          100 + length * math.cos(math.radians(45)))
    schema = _synthetic_schema({5: (200, 100), 7: (int(el[0]), int(el[1])),
                                9: (int(el[0]), int(el[1]) + 120),
                                6: (220, 100), 8: (220, 220), 10: (220, 340)})
    meas = HumanMeasurement()
    out = meas.update(schema, (640, 480))
    assert out["online"], out
    l = out["arms"]["L"]
    assert l["valid_arm"], l
    assert abs(l["abduction_signed"] + 45.0) < 1.0, l["abduction_signed"]
    trimmed = _trim_arms(out)
    json.dumps({"ts": 0.0, "online": True, "arms": trimmed})
    print("  selftest_mimic_side: OK")


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        print("mimic_side selftest: ALL PASSED")
        return
    try:
        run(args)
    except (MimicConfigError, KeyboardInterrupt) as exc:
        if isinstance(exc, MimicConfigError):
            print(f"[side] 配置错误：{exc}")
            sys.exit(2)


if __name__ == "__main__":
    main()
