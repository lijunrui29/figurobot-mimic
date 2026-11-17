"""mapper 层：人体姿态（pose_schema 规范 schema）→ 机器人舵机目标 tick。

本层是"人体 → 机器人"的翻译官，只做数值映射，**不读摄像头、不发网络**。
阶段 2 的 dry-run 入口和阶段 4 的实时驱动都只依赖它。

=== 为什么不用 pose_schema.human_adapter 算好的 angle_deg ===
1. `shoulder_*_pitch.angle_deg` 是 `atan2(|dx|, |dy|)`，值域 0–90°，
   **"手臂垂下"和"举过头顶"同为 0°**，分不出朝向（技术坑 1）。
2. 它是在**归一化坐标**里算的，640×480 的画面会把真实 45° 算成 36.9°
   （技术坑 2）。
所以本层只从 schema 的 `position`（归一化坐标）取点，配合调用方给的
`frame_size` **还原到像素空间**再重算。schema 的签名不用改。

=== 唯一例外：head_yaw ===
`human_adapter` 的 head_yaw 只用了归一化 x 坐标（鼻相对耳中点的水平偏移
除以耳距），分子分母同时被 W 除，**纵横比天然约掉**，所以直接复用它的
`angle_deg` 是对的。这是唯一一个用 angle_deg 的 source。

=== 每个关节的映射链（与任务书一致）===
    人体角 − human_neutral                     （相对中立位）
    × sign × scale                             （方向 / 比例）
    → One Euro 平滑                             （可选；平滑的是机器人角）
    → clamp 到 [min_deg, max_deg]               （相对中立位的偏移范围）
    → robot_neutral_tick + 角度 × TICKS_PER_DEG
    → clamp 到全局 [tick_min, tick_max]
    → 整数 tick
"限速"不在这里做：它需要"上一次真正下发的值"和真实 dt，属 safety 层
（每个关节的 max_step_deg / max_speed_deg_s 由 safety 读取配置）。mapper 的
输出是**期望目标**，safety 的输出才是**允许下发**。

=== 丢帧与平滑 ===
关键点低置信度 / 越界 / 线段过短 / 没人 → 该关节输出 `None`（= 保持），
带原因短码，并且**清掉该关节的 One Euro 状态**：下一有效帧从原始值重新起步，
不跨丢帧维持旧值（与 features/smoothing 的纪律一致）。

=== 左右与镜像 ===
COCO 5/7/9 是**解剖学左侧**，所以 servo 13（左肩外展）/6（右肩外展）直接对应
人的左/右臂，画面里的镜像自动成立（2026-11-17 真机实测映射）。需要标定的是
`sign`：正面机位下左臂外展时上臂指向画面右（有符号角为正）、右臂外展时指向
画面左（为负）。实测两肩外展都是 **tick 减小 = 向外抬**，所以左臂 source 正角
要配 sign=-1、右臂 source 负角要配 sign=+1（见 mimic_config.json）。

=== 角度约定 ===
`upper_arm_abduction_signed`：肩→肘 向量相对**向下竖直轴**的有符号角，
    atan2(dx, dy)（图像 y 向下）：
        0° = 竖直向下；+90° = 指向画面右；−90° = 指向画面左；±180° = 竖直向上。
`upper_arm_elevation`：同一根轴的无符号幅度 0–180°（0=下垂，180=上举）。
`elbow_flex`：肘弯曲量 = 180° − 内角（0=伸直，180=完全折叠）。
`elbow_inner`：内角（180=伸直），与 features.ElbowFlexFeatures 同口径。
注意 pose_schema 的 `elbow_*_pitch` 是"相对伸直的偏折"（0=伸直），**别直接混用**。

接口约定（保持稳定）：
    HumanMeasurement(visibility_min=0.3, edge_margin=..., min_segment_ratio=...)
        update(schema, frame_size) -> dict    逐侧的人体测量（不含任何机器人量）
    MimicMapper(config, measurement=None)
        update(schema, frame_size, now=None) -> dict
            {'targets': {servo_id: int|None},
             'details': {servo_id: {...每关节诊断...}},
             'measures': HumanMeasurement 的输出,
             'timestamp': float}
        reset()
纯标准库（math / time），可独立测试：python mapper.py
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

from mimic_geometry import FRAME_EDGE_MARGIN, MIN_SEGMENT_RATIO
from mimic_config import validate_config
from one_euro import OneEuroFilter
from pose_schema import SERVO_TO_CANONICAL
from safety import TICKS_PER_DEG, clamp_tick

# 无效原因短码（与 features 的命名保持一致；本层独有的两个另加）
REASON_NO_POSE = "no_pose"
REASON_NO_SIZE = "no_size"
REASON_NO_POINT = "no_point"
REASON_OUT_OF_FRAME = "out_of_frame"
REASON_GEOM = "geom"
REASON_DISABLED = "disabled"
REASON_NO_SOURCE = "no_source"
REASON_NO_NEUTRAL = "no_robot_neutral"
REASON_NO_SIDE = "no_side"        # upper_arm_flexion_signed 需要侧视测量

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# 人体侧测量：schema.position + 帧宽高 → 像素空间角度
# ---------------------------------------------------------------------------

def _empty_arm(reason: Optional[str] = None) -> dict:
    return {
        "shoulder_px": None, "elbow_px": None, "wrist_px": None,
        "abduction_signed": None,      # 相对向下竖直轴，有符号（图像平面）
        "elevation": None,             # 相对向下竖直轴，无符号 0-180
        "elbow_inner": None,           # 内角（180 = 伸直）
        "elbow_flex": None,            # 弯曲量（0 = 伸直）
        "valid_arm": False, "reason_arm": reason,
        "valid_elbow": False, "reason_elbow": reason,
        "conf_arm": None, "conf_elbow": None,
    }


def _joint(schema: dict, canonical: str) -> dict:
    return ((schema or {}).get("joints") or {}).get(canonical) or {}


def _position(joint: dict) -> Optional[Point]:
    """schema 里的归一化 position → (x, y)；缺失返回 None。"""
    p = joint.get("position")
    if not isinstance(p, dict):
        return None
    x, y = p.get("x"), p.get("y")
    if x is None or y is None:
        return None
    return (float(x), float(y))


def _outside(point: Point, margin: float) -> bool:
    """归一化坐标是否越出画面（含 RTMPose 的外推点）。与 features 同口径。"""
    x, y = point
    return (x < -margin or x > 1.0 + margin
            or y < -margin or y > 1.0 + margin)


def _signed_from_down(v: Point) -> float:
    """向量相对向下竖直轴的有符号角（度）。0=下，+90=画面右，±180=上。"""
    return math.degrees(math.atan2(v[0], v[1]))


def _elevation_from_down(v: Point) -> float:
    """向量相对向下竖直轴的无符号幅度（度）。0=下，90=水平，180=上。"""
    return math.degrees(math.atan2(abs(v[0]), v[1]))


def _inner_angle(joint: Point, p1: Point, p2: Point) -> float:
    """以 joint 为顶点的内角（度，0-180；180 = 三点共线且 p1/p2 在两侧）。"""
    u = (p1[0] - joint[0], p1[1] - joint[1])
    v = (p2[0] - joint[0], p2[1] - joint[1])
    nu = math.hypot(u[0], u[1])
    nv = math.hypot(v[0], v[1])
    cos = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _valid_size(size) -> Optional[Tuple[int, int]]:
    """帧宽高校验：必须是两个正数。"""
    if not size or len(size) < 2:
        return None
    w, h = float(size[0]), float(size[1])
    if w <= 0 or h <= 0:
        return None
    return int(w), int(h)


# 2026-11-17 真机实测覆盖：pose_schema 的 SERVO_TO_CANONICAL 是"按对称推测"的
# 旧表（13=右腕、6=左肘），与这台机器人不符。mimic 使用的四个关节以实测为准。
_MIMIC_SIDES = {3: "L", 8: "R", 6: "R", 13: "L"}


def joint_side(servo_id: int) -> Optional[str]:
    """舵机 → 人体侧别（'L'/'R'）。mimic 的四个关节用实测覆盖。"""
    sid = int(servo_id)
    if sid in _MIMIC_SIDES:
        return _MIMIC_SIDES[sid]
    canonical = SERVO_TO_CANONICAL.get(sid)
    if not canonical:
        return None
    if "_left" in canonical:
        return "L"
    if "_right" in canonical:
        return "R"
    return None


class HumanMeasurement:
    """从 human schema 的 position 重算像素空间角度（**只管人体侧**）。

    为什么从 position 重算：见模块头（技术坑 1、2）。
    点可用性判据与 features 一致：DOF 置信度 >= visibility_min、
    归一化坐标不越出 edge_margin、线段不短于画面对角线的 min_segment_ratio。
    """

    def __init__(self, visibility_min: float = 0.3,
                 edge_margin: float = FRAME_EDGE_MARGIN,
                 min_segment_ratio: float = MIN_SEGMENT_RATIO) -> None:
        self.visibility_min = float(visibility_min)
        self.edge_margin = float(edge_margin)
        self.min_segment_ratio = float(min_segment_ratio)

    def update(self, schema: Optional[dict], frame_size) -> dict:
        """返回 {'online', 'frame_size', 'reason', 'arms': {'L','R'}, 'head'}。"""
        out = {
            "online": False,
            "frame_size": None,
            "reason": None,
            "arms": {"L": _empty_arm(), "R": _empty_arm()},
            "head": {"yaw": None, "valid": False, "reason": None, "conf": None},
        }
        if schema is None:
            return self._fail(out, REASON_NO_POSE)
        if not (schema.get("source") or {}).get("online"):
            return self._fail(out, REASON_NO_POSE)
        size = _valid_size(frame_size)
        if size is None:
            return self._fail(out, REASON_NO_SIZE)

        out["online"] = True
        out["frame_size"] = size
        w, h = size
        min_len = max(2.0, self.min_segment_ratio * math.hypot(w, h))
        for side in ("L", "R"):
            out["arms"][side] = self._arm(schema, side, w, h, min_len)
        out["head"] = self._head(schema)
        return out

    # ---------- 内部实现 ----------

    @staticmethod
    def _fail(out: dict, reason: str) -> dict:
        """整体不可用（没人 / 没尺寸）：每个测点都标同一个原因。"""
        out["reason"] = reason
        for side in ("L", "R"):
            out["arms"][side] = _empty_arm(reason)
        out["head"] = {"yaw": None, "valid": False, "reason": reason, "conf": None}
        return out

    def _arm(self, schema: dict, side: str, w: int, h: int,
             min_len: float) -> dict:
        suffix = "left" if side == "L" else "right"
        sh_j = _joint(schema, f"shoulder_{suffix}_pitch")
        el_j = _joint(schema, f"elbow_{suffix}_pitch")
        wr_j = _joint(schema, f"wrist_{suffix}_pitch")
        sh_p, el_p, wr_p = _position(sh_j), _position(el_j), _position(wr_j)
        conf_arm = float(sh_j.get("confidence") or 0.0)   # = min(vis 肩, vis 肘)
        conf_elbow = float(el_j.get("confidence") or 0.0)  # = min(肩, 肘, 腕)

        m = _empty_arm()
        m["conf_arm"], m["conf_elbow"] = conf_arm, conf_elbow

        # --- 上臂：只需要肩 + 肘 ---
        reason = None
        if sh_p is None or el_p is None:
            reason = REASON_NO_POINT
        elif conf_arm < self.visibility_min:
            reason = REASON_NO_POINT
        elif _outside(sh_p, self.edge_margin) or _outside(el_p, self.edge_margin):
            reason = REASON_OUT_OF_FRAME
        if reason is None:
            sh_px = (sh_p[0] * w, sh_p[1] * h)
            el_px = (el_p[0] * w, el_p[1] * h)
            if math.hypot(el_px[0] - sh_px[0], el_px[1] - sh_px[1]) < min_len:
                reason = REASON_GEOM
            else:
                v = (el_px[0] - sh_px[0], el_px[1] - sh_px[1])
                m["shoulder_px"], m["elbow_px"] = sh_px, el_px
                m["abduction_signed"] = _signed_from_down(v)
                m["elevation"] = _elevation_from_down(v)
                m["valid_arm"] = True
        m["reason_arm"] = reason

        # --- 肘：需要肩 + 肘 + 腕 ---
        reason_e = None
        if sh_p is None or el_p is None or wr_p is None:
            reason_e = REASON_NO_POINT
        elif conf_elbow < self.visibility_min:
            reason_e = REASON_NO_POINT
        elif any(_outside(p, self.edge_margin) for p in (sh_p, el_p, wr_p)):
            reason_e = REASON_OUT_OF_FRAME
        if reason_e is None:
            sh_px = (sh_p[0] * w, sh_p[1] * h)
            el_px = (el_p[0] * w, el_p[1] * h)
            wr_px = (wr_p[0] * w, wr_p[1] * h)
            if (math.hypot(el_px[0] - sh_px[0], el_px[1] - sh_px[1]) < min_len
                    or math.hypot(wr_px[0] - el_px[0], wr_px[1] - el_px[1]) < min_len):
                reason_e = REASON_GEOM
            else:
                m["wrist_px"] = wr_px
                inner = _inner_angle(el_px, sh_px, wr_px)
                m["elbow_inner"] = inner
                m["elbow_flex"] = 180.0 - inner
                m["valid_elbow"] = True
        m["reason_elbow"] = reason_e
        return m

    def _head(self, schema: dict) -> dict:
        """头偏航：唯一直接用 schema.angle_deg 的量（纵横比天然约掉）。"""
        j = _joint(schema, "head_yaw")
        conf = float(j.get("confidence") or 0.0)
        angle = j.get("angle_deg")
        out = {"yaw": None, "valid": False, "reason": None, "conf": conf}
        if angle is None:
            out["reason"] = REASON_NO_POINT
        elif conf < self.visibility_min:
            out["reason"] = REASON_NO_POINT
        else:
            out["yaw"] = float(angle)
            out["valid"] = True
        return out


# ---------------------------------------------------------------------------
# 映射：人体测量 → 目标 tick
# ---------------------------------------------------------------------------

class MimicMapper:
    """配置驱动的映射器。**一个数据源一个实例**（持有 One Euro 跨帧状态）。

    配置由 `mimic_config.load_config` 校验；本类内部只做结构校验
    （`enforce_whitelist=False`），运行时白名单由 safety 兜底。
    """

    def __init__(self, config: dict,
                 measurement: Optional[HumanMeasurement] = None) -> None:
        self.config = validate_config(config, source_label="<mapper>",
                                      enforce_whitelist=False)
        g = self.config["global"]
        self.measurement = measurement or HumanMeasurement(
            visibility_min=g["visibility_min"],
            edge_margin=g["edge_margin"],
            min_segment_ratio=g["min_segment_ratio"],
        )
        self._smooth_cfg = self.config["smoothing"]
        self._filters: Dict[int, OneEuroFilter] = {}
        self._build_filters()

    def _build_filters(self) -> None:
        self._filters = {}
        if not self._smooth_cfg["enabled"]:
            return
        for joint in self.config["joints"]:
            if joint["enabled"] and joint["smooth"]:
                self._filters[int(joint["servo_id"])] = OneEuroFilter(
                    min_cutoff=self._smooth_cfg["min_cutoff"],
                    beta=self._smooth_cfg["beta"],
                    d_cutoff=self._smooth_cfg["d_cutoff"],
                )

    def reset(self) -> None:
        """清空所有平滑状态（换人 / 重新标定 / 丢帧后重新起步）。"""
        for flt in self._filters.values():
            flt.reset()

    def update(self, human_schema: Optional[dict], frame_size,
               now: Optional[float] = None,
               side_measures: Optional[dict] = None) -> dict:
        """一帧映射。`targets[servo_id] is None` = 本帧保持（不下发）。

        side_measures：可选的侧视测量（`HumanMeasurement.update()` 的返回，
        由侧视进程发布）。`upper_arm_flexion_signed` 用它；缺省时该源报
        `no_side`，不影响正面来源的关节。
        """
        ts = time.monotonic() if now is None else float(now)
        measures = self.measurement.update(human_schema, frame_size)
        targets: Dict[int, Optional[int]] = {}
        details: Dict[int, dict] = {}
        for joint in self.config["joints"]:
            sid = int(joint["servo_id"])
            detail = self._map_joint(joint, measures, ts, side_measures)
            details[sid] = detail
            targets[sid] = detail["tick"]
        return {
            "targets": targets,
            "details": details,
            "measures": measures,
            "side_measures": side_measures,
            "timestamp": ts,
        }

    # ---------- 内部实现 ----------

    def _map_joint(self, joint: dict, measures: dict, now: float,
                   side_measures: Optional[dict] = None) -> dict:
        sid = int(joint["servo_id"])
        detail = {
            "servo_id": sid,
            "name": joint.get("name", ""),
            "source": joint.get("source"),
            "enabled": bool(joint["enabled"]),
            "calibrated": bool(joint.get("calibrated")),
            "human_angle": None,      # 人体侧原始读数
            "delta": None,            # 减中立位后的偏移
            "robot_deg_raw": None,    # 平滑后、clamp 前
            "robot_deg": None,        # clamp 后（相对 robot_neutral）
            "tick": None,
            "reason": None,
            "clamped": False,
            "smoothed": False,
        }

        if not joint["enabled"]:
            detail["reason"] = REASON_DISABLED
            self._reset_filter(sid)
            return detail
        if not measures["online"]:
            detail["reason"] = measures["reason"] or REASON_NO_POSE
            self._reset_filter(sid)
            return detail
        if joint.get("source") is None:
            detail["reason"] = REASON_NO_SOURCE
            self._reset_filter(sid)
            return detail

        value, reason = self._value(joint, measures, side_measures)
        if reason is not None:
            detail["reason"] = reason
            self._reset_filter(sid)       # 丢帧不跨帧维持滤波状态
            return detail
        detail["human_angle"] = value

        delta = value - float(joint["human_neutral"])
        detail["delta"] = delta
        deg = int(joint["sign"]) * float(joint["scale"]) * delta

        flt = self._filters.get(sid)
        if flt is not None:
            deg = flt.update(deg, now)
            detail["smoothed"] = True
        detail["robot_deg_raw"] = deg

        lo, hi = float(joint["min_deg"]), float(joint["max_deg"])
        if deg < lo:
            deg = lo
            detail["clamped"] = True
        elif deg > hi:
            deg = hi
            detail["clamped"] = True
        detail["robot_deg"] = deg

        neutral = joint.get("robot_neutral_tick")
        if neutral is None:
            detail["reason"] = REASON_NO_NEUTRAL
            return detail

        g = self.config["global"]
        tick = float(neutral) + deg * TICKS_PER_DEG
        tick = clamp_tick(tick, g["tick_min"], g["tick_max"])
        detail["tick"] = int(round(tick))
        return detail

    def _value(self, joint: dict, measures: dict,
               side_measures: Optional[dict] = None):
        """取出该 source 的人体读数：返回 (value, None) 或 (None, reason)。"""
        src = joint["source"]

        if src == "head_yaw":
            head = measures["head"]
            if head["valid"]:
                return head["yaw"], None
            return None, head["reason"] or REASON_NO_POINT

        side = joint_side(joint["servo_id"])
        if side is None:
            return None, REASON_NO_SOURCE

        if src == "upper_arm_flexion_signed":
            # 侧视：同一套 abduction_signed 几何，但在侧面看到的是前/后抬。
            if not side_measures or not side_measures.get("online"):
                return None, REASON_NO_SIDE
            arm_s = side_measures["arms"][side]
            if arm_s["valid_arm"]:
                return arm_s["abduction_signed"], None
            return None, arm_s["reason_arm"] or REASON_NO_POINT

        arm = measures["arms"][side]

        if src == "upper_arm_abduction_signed":
            if arm["valid_arm"]:
                return arm["abduction_signed"], None
            return None, arm["reason_arm"] or REASON_NO_POINT
        if src == "upper_arm_elevation":
            if arm["valid_arm"]:
                return arm["elevation"], None
            return None, arm["reason_arm"] or REASON_NO_POINT
        if src == "elbow_flex":
            if arm["valid_elbow"]:
                return arm["elbow_flex"], None
            return None, arm["reason_elbow"] or REASON_NO_POINT
        if src == "elbow_inner":
            if arm["valid_elbow"]:
                return arm["elbow_inner"], None
            return None, arm["reason_elbow"] or REASON_NO_POINT
        return None, REASON_NO_SOURCE

    def _reset_filter(self, servo_id: int) -> None:
        flt = self._filters.get(int(servo_id))
        if flt is not None:
            flt.reset()


# ---------------------------------------------------------------------------
# 自测：合成 schema + 合成配置，不碰摄像头/机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _assert_close(actual, expected, eps, msg) -> None:
    if actual is None or abs(float(actual) - float(expected)) > eps:
        raise AssertionError(f"{msg}: expected {expected} ± {eps}, got {actual}")


def _schema(px: Dict[int, tuple], w: int = 640, h: int = 480,
            vis: Optional[Dict[int, float]] = None) -> dict:
    """按像素坐标造一帧合成 pose，再走真正的 human_adapter。"""
    from pose_schema import human_adapter
    landmarks = [(0.5, 0.5, 0.0, 0.0) for _ in range(17)]
    for pid, (x, y) in px.items():
        v = 1.0 if not vis or pid not in vis else float(vis[pid])
        landmarks[pid] = (x / w, y / h, 0.0, v)
    return human_adapter({"landmarks": landmarks, "image_size": (w, h)})


def _joint_cfg(servo_id: int, source: str, **kw) -> dict:
    j = {
        "servo_id": servo_id, "name": f"s{servo_id}", "enabled": True,
        "source": source, "sign": 1, "scale": 1.0, "human_neutral": 0.0,
        "robot_neutral_tick": 2048, "min_deg": -180.0, "max_deg": 180.0,
        "smooth": False, "calibrated": True,
    }
    j.update(kw)
    return j


def _cfg(joints, smooth: bool = False) -> dict:
    return {
        "global": {"whitelist": [3, 6, 8, 13]},
        "smoothing": {"enabled": smooth, "min_cutoff": 1.0,
                      "beta": 0.05, "d_cutoff": 1.0},
        "joints": joints,
    }


def selftest_mapper() -> None:
    """覆盖：下垂/水平/上举、肘直/弯、非方形画面、丢帧/低置信/越界/过短、
    中立/符号/缩放、限位、平滑、头部、无中立位。"""
    W, H = 640, 480
    sh = {5: (200, 100), 6: (400, 100)}   # 左右肩
    down = {**sh, 7: (200, 200), 8: (400, 200)}
    horiz = {**sh, 7: (100, 100), 8: (500, 100)}     # 左臂向左、右臂向右
    up = {**sh, 7: (200, 0), 8: (400, 0)}

    # 1) 上臂：下垂 / 水平 / 上举（有符号与无符号两套 source）
    m = MimicMapper(_cfg([_joint_cfg(4, "upper_arm_abduction_signed")]))
    d = m.update(_schema(down), (W, H), now=0.0)["details"][4]
    _assert_close(d["human_angle"], 0.0, 0.05, "arm down signed")
    _assert(d["tick"] == 2048, f"arm down tick={d['tick']}")

    d = m.update(_schema(horiz), (W, H), now=0.1)["details"][4]
    _assert_close(d["human_angle"], -90.0, 0.05, "left arm horizontal signed")
    _assert(d["tick"] == int(round(2048 - 90 * TICKS_PER_DEG)),
            f"horizontal tick={d['tick']}")

    d = m.update(_schema(up), (W, H), now=0.2)["details"][4]
    _assert_close(abs(d["human_angle"]), 180.0, 0.05, "left arm overhead")

    m2 = MimicMapper(_cfg([_joint_cfg(9, "upper_arm_elevation")]))
    _assert_close(m2.update(_schema(horiz), (W, H), now=0.0)
                  ["details"][9]["human_angle"], 90.0, 0.05,
                  "elevation magnitude should be 90 at horizontal")

    # 2) 肘：伸直 / 弯曲（两个口径）
    straight = {**sh, 7: (200, 200), 8: (400, 200),
                9: (200, 300), 10: (400, 300)}
    bent = {**straight, 9: (100, 200)}
    m = MimicMapper(_cfg([_joint_cfg(3, "elbow_flex")]))
    d = m.update(_schema(straight), (W, H), now=0.0)["details"][3]
    _assert_close(d["human_angle"], 0.0, 0.05, "elbow straight flex")
    d = m.update(_schema(bent), (W, H), now=0.1)["details"][3]
    _assert_close(d["human_angle"], 90.0, 0.05, "elbow bent flex")
    _assert(d["tick"] is not None, "elbow bent should map to a tick")

    m = MimicMapper(_cfg([_joint_cfg(3, "elbow_inner")]))
    d = m.update(_schema(straight), (W, H), now=0.0)["details"][3]
    _assert_close(d["human_angle"], 180.0, 0.05, "elbow inner straight=180")

    # 2b) 侧视 flexion 源：数值来自 side_measures，缺侧视时 no_side
    side = {"online": True, "arms": {
        "L": {"valid_arm": True, "abduction_signed": -30.0, "reason_arm": None},
        "R": {"valid_arm": False, "abduction_signed": None,
              "reason_arm": "no_point"},
    }}
    j = _joint_cfg(13, "upper_arm_flexion_signed", sign=1)
    m = MimicMapper(_cfg([j]))
    d = m.update(_schema(down), (W, H), now=0.0,
                 side_measures=side)["details"][13]
    _assert_close(d["human_angle"], -30.0, 0.05, "side flexion angle")
    _assert(d["tick"] == int(round(2048 - 30.0 * TICKS_PER_DEG)),
            f"side flexion tick={d['tick']}")
    d = m.update(_schema(down), (W, H), now=0.1,
                 side_measures=None)["details"][13]
    _assert(d["tick"] is None and d["reason"] == "no_side",
            f"missing side reason={d['reason']}")

    # 3) 非方形画面：同样的像素几何必须给同样的角度
    m = MimicMapper(_cfg([_joint_cfg(4, "upper_arm_abduction_signed")]))
    for (w, h) in ((640, 480), (480, 640), (1920, 1080)):
        d = m.update(_schema(horiz, w=w, h=h), (w, h), now=0.0)["details"][4]
        _assert_close(d["human_angle"], -90.0, 0.15,
                      f"aspect {w}x{h} should keep -90deg")
    # 真实 45° 的像素几何（640x480）：归一化空间会算成 36.9°，像素空间必须 45°
    p45 = {5: (200, 100), 6: (400, 100), 7: (300, 200), 8: (500, 200)}
    d = m.update(_schema(p45), (W, H), now=0.0)["details"][4]
    _assert_close(d["human_angle"], 45.0, 0.1, "true 45deg in pixel space")

    # 4) 没人 / 没尺寸
    res = m.update(None, (W, H), now=0.0)
    _assert(res["targets"][4] is None, "no pose must give None")
    _assert(res["details"][4]["reason"] == "no_pose", "no pose reason")
    res = m.update(_schema(down), None, now=0.0)
    _assert(res["details"][4]["reason"] == "no_size", "no size reason")

    # 5) 低置信度 / 画面外 / 线段过短
    d = m.update(_schema(down, vis={7: 0.1}), (W, H), now=0.0)["details"][4]
    _assert(d["tick"] is None and d["reason"] == "no_point",
            f"low conf should be no_point, got {d['reason']}")
    d = m.update(_schema({**down, 7: (int(1.05 * W), 200)}),
                 (W, H), now=0.0)["details"][4]
    _assert(d["reason"] == "out_of_frame", f"oob reason={d['reason']}")
    d = m.update(_schema({**down, 7: (201, 101)}), (W, H), now=0.0)["details"][4]
    _assert(d["reason"] == "geom", f"short segment reason={d['reason']}")

    # 6) 中立位 / 符号 / 缩放
    p40 = {5: (200, 100), 7: (200 + 100 * math.sin(math.radians(40)),
                              100 + 100 * math.cos(math.radians(40)))}
    j = _joint_cfg(4, "upper_arm_abduction_signed",
                   human_neutral=10.0, sign=-1, scale=0.5)
    m = MimicMapper(_cfg([j]))
    d = m.update(_schema(p40), (W, H), now=0.0)["details"][4]
    _assert_close(d["human_angle"], 40.0, 0.1, "40deg pose")
    _assert_close(d["delta"], 30.0, 0.1, "delta after neutral")
    _assert_close(d["robot_deg"], -15.0, 0.1, "sign/scale applied")
    _assert(d["tick"] == int(round(2048 - 15 * TICKS_PER_DEG)),
            f"neutral/sign/scale tick={d['tick']}")

    # 7) 角度限位（相对中立位 ±40）与全局 tick 限位
    j = _joint_cfg(4, "upper_arm_abduction_signed", min_deg=-40.0, max_deg=40.0)
    m = MimicMapper(_cfg([j]))
    d = m.update(_schema(horiz), (W, H), now=0.0)["details"][4]
    _assert(d["clamped"] and abs(d["robot_deg"] + 40.0) < 1e-6,
            f"delta clamp: deg={d['robot_deg']} clamped={d['clamped']}")
    _assert(d["tick"] == int(round(2048 - 40 * TICKS_PER_DEG)),
            f"clamped tick={d['tick']}")
    j = _joint_cfg(4, "upper_arm_abduction_signed", robot_neutral_tick=50)
    m = MimicMapper(_cfg([j]))
    d = m.update(_schema(horiz), (W, H), now=0.0)["details"][4]
    _assert(d["tick"] == 50, f"global tick floor should clamp, got {d['tick']}")

    # 8) One Euro：阶跃后第一帧必须被拖慢；丢帧后重新起步
    j = _joint_cfg(4, "upper_arm_abduction_signed", smooth=True)
    m = MimicMapper(_cfg([j], smooth=True))
    m.update(_schema(down), (W, H), now=0.0)
    d = m.update(_schema(horiz), (W, H), now=1.0 / 30.0)["details"][4]
    _assert(d["smoothed"], "smoothing flag missing")
    _assert(-90.0 < d["robot_deg_raw"] < 0.0,
            f"filter should lag the step, got {d['robot_deg_raw']}")
    m.update(None, (W, H), now=2.0 / 30.0)          # 丢帧 → 清滤波状态
    d = m.update(_schema(down), (W, H), now=3.0 / 30.0)["details"][4]
    _assert_close(d["robot_deg_raw"], 0.0, 1e-9,
                  "after a dropped frame the filter must restart from raw")

    # 9) 禁用 / 无中立位 / 无 source
    j = _joint_cfg(4, "upper_arm_abduction_signed", enabled=False)
    _assert(MimicMapper(_cfg([j])).update(_schema(down), (W, H), now=0.0)
            ["details"][4]["reason"] == "disabled", "disabled reason")
    j = _joint_cfg(4, "upper_arm_abduction_signed", robot_neutral_tick=None)
    d = MimicMapper(_cfg([j])).update(_schema(down), (W, H), now=0.0)["details"][4]
    _assert(d["tick"] is None and d["reason"] == "no_robot_neutral",
            f"no neutral reason={d['reason']}")
    _assert(d["robot_deg"] is not None, "robot_deg should still be available")

    # 10) 头部（唯一用 schema.angle_deg 的 source；不在白名单也能算，
    #     白名单由 safety 在运行期过滤，所以这里 enforce_whitelist=False）
    head_pose = {0: (360, 100), 3: (280, 100), 4: (360, 100)}
    j = _joint_cfg(1, "head_yaw", min_deg=-45.0, max_deg=45.0)
    m = MimicMapper(_cfg([j]))
    d = m.update(_schema(head_pose), (W, H), now=0.0)["details"][1]
    _assert_close(d["human_angle"], 26.565, 0.1, "head yaw from schema")

    # 11) 白名单外的禁用关节可以留在配置里（模板里就有头）
    print("  selftest_mapper: OK")


if __name__ == "__main__":
    selftest_mapper()
    print("mapper selftest: ALL PASSED")
