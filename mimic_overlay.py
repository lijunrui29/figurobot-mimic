"""mimic_overlay 层：把映射结果画到画面上（**只呈现，不重算**）。

画面结构（左侧半透明面板）：

    MIMIC  DRY-RUN (no robot)
    src: cam#1  640x480  29.8fps  pose=OK
    --------------------------------------------
    joint     human      deg     want   send  flags
    L-shR     +12.3     +12.3     2188   2188  CLAMP
    L-elb       --        --       --     --   no_point
    --------------------------------------------
    n: neutral 5s   r: reset   s: snapshot   q: quit

"want" = mapper 的期望目标，"send" = safety 允许下发的目标；两者不同就说明
被限速（flags 里同时有 RATE）。**dry-run 下 send 也不会发给机器人**。

分层的边界：本层不读 schema、不算角度、不做限位/限速判断 —— 所有数值都由
mapper / safety 算好传进来，这里只排版。中文关节名无法用 cv2.putText 渲染，
所以用 `L-shR` / `L-elb` 这类 ASCII 短标（由 pose_schema 的规范名推出来）。

接口约定：
    MimicOverlay().draw(frame, mapping, safety_result, *, source_label="",
                        pose_ok=True, fps=None, extra_lines=())
    joint_label(servo_id) -> str

可独立测试（需要 cv2）：python mimic_overlay.py
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from pose_schema import SERVO_TO_CANONICAL

# 颜色（BGR）
_C_WHITE = (235, 235, 235)
_C_DIM = (160, 160, 160)
_C_GREEN = (120, 220, 120)
_C_AMBER = (80, 190, 250)
_C_RED = (90, 90, 240)

_SHORT_PART = {
    "shoulder": "sh", "elbow": "elb", "wrist": "wr", "hip": "hip",
    "knee": "knee", "ankle": "ank", "waist": "waist", "chest": "chest",
    "neck": "neck", "head": "head",
}


# 2026-11-17 真机实测映射（与控制台 HTML 的猜测不同）：mimic 只用这四个关节。
# pose_schema 的 SERVO_TO_CANONICAL 仍是"按对称推测"的旧表，这里覆盖掉，
# 免得画面把 6 号标成"左肘"、13 号标成"右腕"之类。
_MIMIC_LABELS = {
    3: "L-elb",
    8: "R-elb",
    6: "R-shAbd",
    13: "L-shP",
}


def joint_label(servo_id: int) -> str:
    """舵机号 → 画面用 ASCII 短标（如 3 → `L-elb`，6 → `R-shAbd`）。"""
    sid = int(servo_id)
    if sid in _MIMIC_LABELS:
        return _MIMIC_LABELS[sid]
    canonical = SERVO_TO_CANONICAL.get(sid)
    if not canonical:
        return f"s{sid}"
    parts = canonical.split("_")
    if len(parts) == 3:
        part, side, axis = parts
        return f"{side[0].upper()}-{_SHORT_PART.get(part, part)}{axis[0].upper()}"
    if len(parts) == 2:
        return f"{parts[0]}{parts[1][0].upper()}"
    return canonical


def _put(frame: np.ndarray, text: str, x: int, y: int,
         color=_C_WHITE, scale: float = 0.48, thickness: int = 1) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _fmt(value, digits: int = 1, width: int = 0, signed: bool = True) -> str:
    if value is None:
        return f"{'--':>{width}}" if width else "--"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return f"{fmt.format(float(value)):>{width}}" if width else fmt.format(float(value))


class MimicOverlay:
    """dry-run 画面叠加器。无状态，可复用。"""

    def __init__(self, panel_alpha: float = 0.62, line_h: int = 20,
                 text_scale: float = 0.48, margin: int = 10) -> None:
        self.panel_alpha = float(panel_alpha)
        self.line_h = int(line_h)
        self.text_scale = float(text_scale)
        self.margin = int(margin)

    def draw(self, frame: np.ndarray, mapping: dict, safety_result, *,
             source_label: str = "", pose_ok: bool = True,
             fps: Optional[float] = None,
             extra_lines: Iterable[str] = (),
             mode_label: str = "DRY-RUN (no robot)") -> None:
        """在 frame 上原地绘制。mapping = MimicMapper.update() 的返回。

        mode_label 只影响标题文字（默认保持 dry-run 文案）；是否真的下发由
        入口的 --live 决定，本层不参与。
        """
        h, w = frame.shape[:2]
        details = mapping.get("details") or {}
        targets = mapping.get("targets") or {}
        safety_targets = getattr(safety_result, "targets", {}) or {}
        rate_limited = getattr(safety_result, "rate_limited", {}) or {}
        holding = bool(getattr(safety_result, "holding", False))
        tripped = bool(getattr(safety_result, "tripped", False))

        # ---- 组行 ----
        lines = []      # [(text, color)]
        header = f"MIMIC  {mode_label}"
        lines.append((header, _C_AMBER))
        info = f"src: {source_label or '?'}"
        if fps:
            info += f"  {fps:.1f}fps"
        info += f"  pose={'OK' if pose_ok else 'NO PERSON'}"
        lines.append((info, _C_GREEN if pose_ok else _C_RED))

        if tripped:
            lines.append(("SAFETY TRIPPED - sending disabled", _C_RED))
        elif holding:
            lines.append(("WATCHDOG HOLD - no valid data", _C_AMBER))

        for extra in extra_lines:
            lines.append((str(extra), _C_AMBER))

        lines.append(("-" * 48, _C_DIM))
        lines.append((f"{'joint':<7}{'human':>9}{'deg':>8}{'want':>7}{'send':>7}"
                      f"  flags", _C_DIM))

        disabled = 0
        for sid, detail in details.items():
            if not detail.get("enabled"):
                disabled += 1
                continue
            sid = int(sid)
            label = joint_label(sid)
            human = _fmt(detail.get("human_angle"), 1, 9)
            deg = _fmt(detail.get("robot_deg"), 1, 8)
            want = detail.get("tick")
            send = safety_targets.get(sid)
            want_s = f"{want:>7}" if want is not None else f"{'--':>7}"
            send_s = f"{send:>7}" if send is not None else f"{'--':>7}"

            flags = []
            reason = detail.get("reason")
            if reason:
                flags.append(str(reason))
            if detail.get("clamped"):
                flags.append("CLAMP")
            if rate_limited.get(sid):
                flags.append("RATE")
            if holding:
                flags.append("HOLD")
            if not detail.get("calibrated"):
                flags.append("UNCAL")
            flag_s = " ".join(flags)

            if reason:
                color = _C_DIM
            elif detail.get("clamped") or rate_limited.get(sid) or holding:
                color = _C_AMBER
            elif not detail.get("calibrated"):
                color = _C_AMBER
            else:
                color = _C_GREEN
            lines.append((f"{label:<7}{human}{deg}{want_s}{send_s}  {flag_s}",
                          color))

        lines.append(("-" * 48, _C_DIM))
        if disabled:
            lines.append((f"{disabled} joint(s) disabled in config", _C_DIM))
        lines.append(("n: neutral 5s   r: reset   s: snapshot   q: quit", _C_DIM))

        # ---- 面板尺寸 ----
        pad = self.margin
        max_w = 0
        for text, _ in lines:
            (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           self.text_scale, 1)
            max_w = max(max_w, tw)
        panel_w = min(w, max_w + pad * 2)
        panel_h = min(h, self.line_h * len(lines) + pad * 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1.0 - self.panel_alpha,
                        0.0, frame)

        y = pad + int(self.line_h * 0.75)
        for text, color in lines:
            if y > panel_h - pad:
                break
            _put(frame, text, pad, y, color, self.text_scale)
            y += self.line_h


# ---------------------------------------------------------------------------
# 自测：空白画布上画，确认不抛异常且真的写了像素
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest_mimic_overlay() -> None:
    class _FakeSafety:
        def __init__(self, targets, limited=None, holding=False, tripped=False):
            self.targets = targets
            self.rate_limited = limited or {}
            self.holding = holding
            self.tripped = tripped

    details = {
        4: {"name": "左肩R", "source": "upper_arm_abduction_signed",
            "enabled": True, "human_angle": 12.34, "robot_deg": 12.34,
            "tick": 2188, "reason": None, "clamped": True, "smoothed": True,
            "calibrated": False},
        6: {"name": "左肘P", "source": "elbow_flex", "enabled": True,
            "human_angle": None, "robot_deg": None, "tick": None,
            "reason": "no_point", "clamped": False, "smoothed": False,
            "calibrated": False},
        7: {"name": "右肩P", "source": None, "enabled": False, "human_angle": None,
            "robot_deg": None, "tick": None, "reason": "disabled",
            "clamped": False, "smoothed": False, "calibrated": False},
    }
    mapping = {"details": details, "targets": {4: 2188, 6: None, 7: None}}

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = MimicOverlay()
    overlay.draw(frame, mapping,
                 _FakeSafety({4: 2200}, limited={4: True}),
                 source_label="cam#1", pose_ok=True, fps=29.7,
                 extra_lines=["NEUTRAL 3.2/5.0 s  samples=96"])
    _assert(int(frame.sum()) > 0, "overlay should have drawn something")

    # 各种状态都不许抛异常
    for safety in (_FakeSafety({}, holding=True),
                   _FakeSafety({}, tripped=True),
                   _FakeSafety({4: 2188})):
        frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
        overlay.draw(frame2, mapping, safety, source_label="video",
                     pose_ok=False, fps=None)
        _assert(int(frame2.sum()) > 0, "overlay should draw on small frames")

    # 短标（mimic 实测映射）
    _assert(joint_label(3) == "L-elb", joint_label(3))
    _assert(joint_label(8) == "R-elb", joint_label(8))
    _assert(joint_label(6) == "R-shAbd", joint_label(6))
    _assert(joint_label(13) == "L-shP", joint_label(13))
    _assert(joint_label(1) == "headY", joint_label(1))
    _assert(joint_label(99) == "s99", joint_label(99))

    print("  selftest_mimic_overlay: OK")


if __name__ == "__main__":
    selftest_mimic_overlay()
    print("mimic_overlay selftest: ALL PASSED")
