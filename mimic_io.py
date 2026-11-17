"""mimic_io 层：离线视频回放源 + 逐帧 CSV 记录（不碰摄像头/机器人）。

为什么单独一层：
    * `capture.py` 的 `FrameSource` 只管"打开/读帧/释放"，本仓库原本只有
      摄像头实现。离线回放要喂同一套上层链路，所以在这里补一个
      `VideoFileSource` 子类 —— **不修改 capture.py**（它的接口约定稳定，
      加能力走加子类）。
    * CSV 记录是调参的主要依据：画面上的读数是瞬时的，只有落盘才能事后
      对齐"人体角 / 机器人角 / 是否被限位或限速 / 丢帧原因"。
      默认**不写盘**，只有显式 `--record PATH` 才写。

接口约定：
    VideoFileSource(path, loop=False)       FrameSource 子类；额外暴露 fps / frame_count
    CsvRecorder(path)                       with / close()；write() 每帧一次
    list_video_info(path)                   -> {'fps','frame_count','size'} 或 None

可独立测试（需要 cv2）：python mimic_io.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2

from capture import FrameSource


class VideoFileSource(FrameSource):
    """视频文件图像源（离线回放）。

    与 `CameraCapture` 同一套生命周期（open/read/release/with），上层不用分支。
    `loop=True` 时读到末尾回到第一帧（调参时反复看同一段动作）。
    """

    def __init__(self, path: str, loop: bool = False) -> None:
        super().__init__()
        self.path = Path(path)
        self.loop = bool(loop)
        self.fps: float = 0.0
        self.frame_count: int = 0

    def _open(self):
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            return None
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 1e-3:          # 有些容器不写 fps；按 30 处理
            self.fps = 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return cap

    def _read(self):
        ret, frame = self._cap.read()
        if ret:
            return frame
        if not self.loop:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        return frame if ret else None

    def _release(self) -> None:
        self._cap.release()


def list_video_info(path: str) -> Optional[dict]:
    """读视频元信息（不遍历帧）。打不开返回 None。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


class CsvRecorder:
    """把每帧每个关节的映射结果写成 CSV（一行 = 一帧 × 一个舵机）。

    列的含义：
        frame / time_s        帧号与时间戳（视频回放用 帧号/fps；摄像头用单调时钟）
        pose_ok               本帧是否判到人
        human_angle_deg       人体侧原始读数（**平滑前**，调参看它）
        robot_deg             sign/scale/中立位/平滑之后的机器人相对角
        robot_tick            mapper 的期望目标
        reason                无效原因（no_point / geom / ...）；有效为空
        safety_tick           safety 允许下发的目标（dry-run 里只记录、不发）
        rate_limited          是否被限速
        safety_holding        是否因看门狗停止下发
    """

    COLUMNS = [
        "frame", "time_s", "pose_ok", "servo_id", "name", "source", "enabled",
        "human_angle_deg", "delta_deg", "robot_deg", "robot_tick", "reason",
        "clamped", "smoothed", "calibrated", "safety_tick", "rate_limited",
        "safety_holding", "neutral_capture",
    ]

    def __init__(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self._fh = open(p, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self.rows = 0

    def write(self, frame_no: int, ts: float, pose_ok: bool,
              mapping: dict, safety_result, neutral_capture: bool = False) -> None:
        """记录一帧。`mapping` 是 MimicMapper.update() 的返回，`safety_result`
        是 SafetyResult（dry-run 也照样记录"允许下发"的值）。"""
        for sid, detail in mapping["details"].items():
            sid = int(sid)
            self._writer.writerow({
                "frame": int(frame_no),
                "time_s": f"{float(ts):.4f}",
                "pose_ok": int(bool(pose_ok)),
                "servo_id": sid,
                "name": detail.get("name", ""),
                "source": detail.get("source") or "",
                "enabled": int(bool(detail.get("enabled"))),
                "human_angle_deg": _num(detail.get("human_angle")),
                "delta_deg": _num(detail.get("delta")),
                "robot_deg": _num(detail.get("robot_deg")),
                "robot_tick": _int(detail.get("tick")),
                "reason": detail.get("reason") or "",
                "clamped": int(bool(detail.get("clamped"))),
                "smoothed": int(bool(detail.get("smoothed"))),
                "calibrated": int(bool(detail.get("calibrated"))),
                "safety_tick": _int(safety_result.targets.get(sid)),
                "rate_limited": int(bool(safety_result.rate_limited.get(sid))),
                "safety_holding": int(bool(safety_result.holding)),
                "neutral_capture": int(bool(neutral_capture)),
            })
            self.rows += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CsvRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def _num(value, digits: int = 3) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _int(value) -> str:
    return "" if value is None else str(int(value))


# ---------------------------------------------------------------------------
# 自测：临时视频 + 临时 CSV，不碰摄像头/机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest_mimic_io() -> None:
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        # ---- 1) 视频回放源：写 6 帧 → 读 6 帧 → loop 再读一帧 ----
        vpath = str(Path(td) / "tiny.avi")
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"MJPG"),
                                 10.0, (64, 48))
        _assert(writer.isOpened(), "VideoWriter MJPG/avi should open")
        for i in range(6):
            frame = np.full((48, 64, 3), i * 30, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        src = VideoFileSource(vpath)
        _assert(src.open(), "VideoFileSource should open the test video")
        frames = []
        while True:
            f = src.read()
            if f is None:
                break
            frames.append(f)
        src.release()
        _assert(len(frames) == 6, f"expected 6 frames, got {len(frames)}")
        _assert(frames[0].shape == (48, 64, 3), f"shape {frames[0].shape}")
        _assert(abs(src.fps - 10.0) < 0.5, f"fps={src.fps}")

        src = VideoFileSource(vpath, loop=True)
        src.open()
        for _ in range(6):
            _assert(src.read() is not None, "loop source should keep reading")
        _assert(src.read() is not None, "loop source should wrap to frame 0")
        src.release()

        info = list_video_info(vpath)
        _assert(info and info["frame_count"] == 6, f"video info={info}")

        # ---- 2) CSV：内容与列名 ----
        cpath = str(Path(td) / "log.csv")
        mapping = {
            "details": {
                4: {"name": "L-shR", "source": "upper_arm_abduction_signed",
                    "enabled": True, "human_angle": -12.345, "delta": -12.345,
                    "robot_deg": 12.0, "tick": 2184, "reason": None,
                    "clamped": True, "smoothed": True, "calibrated": False},
                6: {"name": "L-elb", "source": "elbow_flex", "enabled": True,
                    "human_angle": None, "delta": None, "robot_deg": None,
                    "tick": None, "reason": "no_point", "clamped": False,
                    "smoothed": False, "calibrated": False},
            },
            "targets": {4: 2184, 6: None},
        }

        class _FakeSafety:
            targets = {4: 2200}
            rate_limited = {4: True}
            holding = False

        with CsvRecorder(cpath) as rec:
            rec.write(0, 0.5, True, mapping, _FakeSafety(), neutral_capture=True)
            _assert(rec.rows == 2, f"recorder rows={rec.rows}")

        with open(cpath, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        _assert(len(rows) == 2, f"csv rows={len(rows)}")
        _assert(rows[0]["human_angle_deg"] == "-12.345", rows[0]["human_angle_deg"])
        _assert(rows[0]["rate_limited"] == "1", "rate_limited should be recorded")
        _assert(rows[0]["neutral_capture"] == "1", "neutral flag should be recorded")
        _assert(rows[1]["reason"] == "no_point" and rows[1]["robot_tick"] == "",
                "invalid joint should record reason + empty tick")

    print("  selftest_mimic_io: OK")


if __name__ == "__main__":
    selftest_mimic_io()
    print("mimic_io selftest: ALL PASSED")
