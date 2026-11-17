"""mimic_config 层：人体→机器人映射配置的加载 / 校验 / 保存（纯标准库）。

配置文件是 `mimic_config.json`。JSON 不能写注释，所以本项目的约定是：
**以 `_` 开头的键就是注释**（`_comment` / `_notes` / `_placeholder` …），
校验时原样保留、加载后可直接读给人看、保存时也不丢。其余未知键一律报错
（防止把 `sign` 拼成 `sing` 这种静默失效）。

这一层只做"结构 + 安全边界"的校验，**不做映射计算**：
    * 单位：human_neutral / min_deg / max_deg 都是**度**；其中 min_deg/max_deg
      是**相对 robot_neutral 的偏移范围**（初始规则：中立位 ±40°）。
    * robot_neutral_tick 是原始 tick（0-4095），None = 还没标定。
    * sign ∈ {-1, +1}，scale > 0。
    * 全局白名单只能**收紧**硬编码的 `safety.ALLOWED_SERVO_IDS`。
    * 启用的关节必须在有效白名单里；没启用的可以写（比如头、肩P 先占位）。

接口约定：
    load_config(path) -> dict                    读 + 校验（返回补全默认值的副本）
    validate_config(cfg, source_label=..., enforce_whitelist=True) -> dict
    save_config(path, cfg) -> None               写回（阶段 3 标定用）
    effective_whitelist(cfg) -> frozenset[int]   硬编码 ∩ 配置
    get_joint(cfg, servo_id) -> dict | None

可独立测试：python mimic_config.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

from safety import ALLOWED_SERVO_IDS, MAX_TICK, MIN_TICK

# 支持的人体量来源。新增 source 要同时改 mapper 的取值逻辑，别只加字符串。
SUPPORTED_SOURCES = (
    "upper_arm_abduction_signed",   # 上臂相对"向下竖直轴"的有符号角（正面：外展）
    "upper_arm_flexion_signed",     # 侧视：上臂相对"向下竖直轴"的有符号角（前/后抬）
    "upper_arm_elevation",          # 上臂相对"向下竖直轴"的无符号幅度（0-180）
    "elbow_flex",                   # 肘弯曲量（0=伸直，180=完全折叠）
    "elbow_inner",                  # 肘内角（180=伸直）
    "head_yaw",                     # 头左右偏转（human schema 已算好的角度）
)

DEFAULT_GLOBAL: Dict[str, Any] = {
    "whitelist": sorted(ALLOWED_SERVO_IDS),
    "tick_min": MIN_TICK,
    "tick_max": MAX_TICK,
    "max_step_deg": 6.0,             # 单帧最大步进
    "max_speed_deg_s": 45.0,         # 最大角速度
    "watchdog_s": 0.5,               # 多久没数据就停
    "max_consecutive_errors": 3,     # 连续失败熔断
    "visibility_min": 0.3,           # 关键点门控（与 features 同口径）
    "edge_margin": 0.02,             # 画面边界门控（与 features 同口径）
    "min_segment_ratio": 0.01,       # 最短可信线段（与 features 同口径）
}

DEFAULT_SMOOTHING: Dict[str, Any] = {
    "enabled": True,
    "min_cutoff": 1.0,
    "beta": 0.05,
    "d_cutoff": 1.0,
}

DEFAULT_JOINT: Dict[str, Any] = {
    "name": "",
    "enabled": True,
    "source": None,
    "sign": 1,
    "scale": 1.0,
    "human_neutral": 0.0,
    "robot_neutral_tick": None,
    "min_deg": -40.0,
    "max_deg": 40.0,
    "max_step_deg": None,          # None = 用 global.max_step_deg
    "max_speed_deg_s": None,       # None = 用 global.max_speed_deg_s
    "smooth": True,
    "calibrated": False,
}


class MimicConfigError(ValueError):
    """配置结构 / 安全边界不合法。消息里带字段名，便于定位。"""


def _fail(label: str, msg: str) -> None:
    raise MimicConfigError(f"{label}: {msg}")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _copy_comments(src: dict, dst: dict) -> None:
    """把 `_` 开头的注释键原样搬到规范化后的 dict 里。"""
    for key, val in src.items():
        if isinstance(key, str) and key.startswith("_"):
            dst[key] = copy.deepcopy(val)


def _check_unknown_keys(label: str, src: dict, allowed: set) -> None:
    """未知键直接报错：`sign` 写成 `sing` 这种错法必须当场暴露。"""
    unknown = [k for k in src if not str(k).startswith("_") and k not in allowed]
    if unknown:
        _fail(label, f"未知字段 {unknown}，允许：{sorted(allowed)}")


def _check_int(label: str, name: str, value: Any, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(label, f"{name} 必须是整数，实际 {value!r}")
    if not (lo <= value <= hi):
        _fail(label, f"{name}={value} 越界，允许 [{lo}, {hi}]")
    return value


def _check_number(label: str, name: str, value: Any,
                  lo: float = None, hi: float = None) -> float:
    if not _is_number(value):
        _fail(label, f"{name} 必须是数字，实际 {value!r}")
    value = float(value)
    if lo is not None and value < lo:
        _fail(label, f"{name}={value} 小于下限 {lo}")
    if hi is not None and value > hi:
        _fail(label, f"{name}={value} 大于上限 {hi}")
    return value


def effective_whitelist(cfg: dict) -> FrozenSet[int]:
    """硬编码白名单 ∩ 配置白名单 = 真正允许写的舵机（只能收紧）。"""
    configured = (cfg.get("global") or {}).get("whitelist")
    if configured is None:
        return ALLOWED_SERVO_IDS
    return frozenset(int(x) for x in configured) & ALLOWED_SERVO_IDS


def get_joint(cfg: dict, servo_id: int) -> Optional[dict]:
    """按舵机号取关节配置（取不到返回 None）。"""
    for joint in cfg.get("joints") or []:
        if int(joint.get("servo_id", -1)) == int(servo_id):
            return joint
    return None


def validate_config(cfg: dict, source_label: str = "<config>",
                    enforce_whitelist: bool = True) -> dict:
    """校验并补全默认值，返回**新 dict**（不改入参）。

    enforce_whitelist=True（load_config 用）：启用的关节若不在有效白名单里就报错。
    mapper 内部以 False 调用（只做结构校验），运行时由 safety 再兜一层。
    """
    label = source_label
    if not isinstance(cfg, dict):
        _fail(label, f"顶层必须是 JSON 对象，实际 {type(cfg).__name__}")

    out: Dict[str, Any] = {}
    _copy_comments(cfg, out)
    _check_unknown_keys(label, cfg, {"schema_version", "mount", "global",
                                     "smoothing", "joints"})
    # 这两个是元信息，原样保留（save_config 写回时不能丢）
    for meta in ("schema_version", "mount"):
        if meta in cfg:
            out[meta] = copy.deepcopy(cfg[meta])

    # ---------- global ----------
    raw_global = cfg.get("global") or {}
    if not isinstance(raw_global, dict):
        _fail(label, "global 必须是对象")
    _check_unknown_keys(label, raw_global, set(DEFAULT_GLOBAL) | {"whitelist"})
    g = dict(DEFAULT_GLOBAL)
    g.update({k: v for k, v in raw_global.items() if not str(k).startswith("_")})
    _copy_comments(raw_global, g)

    # 白名单：只能是硬编码集合的子集（写入配置也放宽不了）
    wl = g.get("whitelist")
    if not isinstance(wl, (list, tuple)) or not wl:
        _fail(label, "global.whitelist 必须是非空数组")
    wl_clean = []
    for sid in wl:
        if not isinstance(sid, int) or isinstance(sid, bool):
            _fail(label, f"global.whitelist 里 {sid!r} 不是整数")
        if sid not in ALLOWED_SERVO_IDS:
            _fail(label, f"global.whitelist 含 {sid}，它不在硬编码白名单 "
                         f"{sorted(ALLOWED_SERVO_IDS)} 里 —— 白名单只能收紧，不能放宽")
        if sid not in wl_clean:
            wl_clean.append(sid)
    g["whitelist"] = sorted(wl_clean)

    g["tick_min"] = _check_int(label, "global.tick_min", g["tick_min"], MIN_TICK, MAX_TICK)
    g["tick_max"] = _check_int(label, "global.tick_max", g["tick_max"], MIN_TICK, MAX_TICK)
    if g["tick_min"] >= g["tick_max"]:
        _fail(label, "global.tick_min 必须小于 tick_max")
    g["max_step_deg"] = _check_number(label, "global.max_step_deg",
                                      g["max_step_deg"], lo=0.1)
    g["max_speed_deg_s"] = _check_number(label, "global.max_speed_deg_s",
                                         g["max_speed_deg_s"], lo=0.1)
    g["watchdog_s"] = _check_number(label, "global.watchdog_s", g["watchdog_s"], lo=0.05)
    g["max_consecutive_errors"] = _check_int(label, "global.max_consecutive_errors",
                                             g["max_consecutive_errors"], 1, 100)
    g["visibility_min"] = _check_number(label, "global.visibility_min",
                                        g["visibility_min"], lo=0.0, hi=1.0)
    g["edge_margin"] = _check_number(label, "global.edge_margin",
                                     g["edge_margin"], lo=0.0, hi=0.5)
    g["min_segment_ratio"] = _check_number(label, "global.min_segment_ratio",
                                           g["min_segment_ratio"], lo=0.0, hi=0.5)
    out["global"] = g

    # ---------- smoothing ----------
    raw_smooth = cfg.get("smoothing") or {}
    if not isinstance(raw_smooth, dict):
        _fail(label, "smoothing 必须是对象")
    _check_unknown_keys(label, raw_smooth, set(DEFAULT_SMOOTHING))
    s = dict(DEFAULT_SMOOTHING)
    s.update({k: v for k, v in raw_smooth.items() if not str(k).startswith("_")})
    _copy_comments(raw_smooth, s)
    s["enabled"] = bool(s["enabled"])
    s["min_cutoff"] = _check_number(label, "smoothing.min_cutoff", s["min_cutoff"], lo=0.0)
    s["beta"] = _check_number(label, "smoothing.beta", s["beta"], lo=0.0)
    s["d_cutoff"] = _check_number(label, "smoothing.d_cutoff", s["d_cutoff"], lo=0.0)
    if s["min_cutoff"] <= 0.0:
        _fail(label, "smoothing.min_cutoff 必须大于 0")
    if s["d_cutoff"] <= 0.0:
        _fail(label, "smoothing.d_cutoff 必须大于 0")
    out["smoothing"] = s

    # ---------- joints ----------
    raw_joints = cfg.get("joints")
    if not isinstance(raw_joints, list) or not raw_joints:
        _fail(label, "joints 必须是非空数组")
    joints = []
    seen = set()
    allowed = effective_whitelist(out)
    for idx, raw in enumerate(raw_joints):
        jlabel = f"{label}: joints[{idx}]"
        if not isinstance(raw, dict):
            _fail(jlabel, "必须是对象")
        _check_unknown_keys(jlabel, raw, set(DEFAULT_JOINT) | {"servo_id"})
        j = copy.deepcopy(DEFAULT_JOINT)
        j.update({k: v for k, v in raw.items() if not str(k).startswith("_")})
        _copy_comments(raw, j)

        sid = j.get("servo_id")
        if not isinstance(sid, int) or isinstance(sid, bool):
            _fail(jlabel, f"servo_id 必须是整数，实际 {sid!r}")
        if sid in seen:
            _fail(jlabel, f"servo_id={sid} 重复")
        seen.add(sid)

        j["enabled"] = bool(j["enabled"])
        src = j.get("source")
        if src is None:
            if j["enabled"]:
                _fail(jlabel, f"servo {sid} 已启用但没配 source")
        elif src not in SUPPORTED_SOURCES:
            _fail(jlabel, f"未知 source {src!r}，支持：{list(SUPPORTED_SOURCES)}")

        if j["sign"] not in (-1, 1) or isinstance(j["sign"], bool):
            _fail(jlabel, f"sign 只能是 -1 或 +1，实际 {j['sign']!r}")
        j["sign"] = int(j["sign"])
        j["scale"] = _check_number(jlabel, "scale", j["scale"], lo=0.0)
        if j["scale"] <= 0.0:
            _fail(jlabel, "scale 必须大于 0")
        j["human_neutral"] = _check_number(jlabel, "human_neutral", j["human_neutral"])
        j["min_deg"] = _check_number(jlabel, "min_deg", j["min_deg"])
        j["max_deg"] = _check_number(jlabel, "max_deg", j["max_deg"])
        if j["min_deg"] > j["max_deg"]:
            _fail(jlabel, f"min_deg={j['min_deg']} 大于 max_deg={j['max_deg']}")

        rnt = j.get("robot_neutral_tick")
        if rnt is not None and (not isinstance(rnt, int) or isinstance(rnt, bool)
                                or not (0 <= rnt <= 4095)):
            _fail(jlabel, f"robot_neutral_tick 必须是 0-4095 的整数或 null，实际 {rnt!r}")

        for opt in ("max_step_deg", "max_speed_deg_s"):
            if j[opt] is not None:
                j[opt] = _check_number(jlabel, opt, j[opt], lo=0.1)

        j["smooth"] = bool(j["smooth"])
        j["calibrated"] = bool(j["calibrated"])
        j["servo_id"] = sid
        if not isinstance(j.get("name"), str):
            _fail(jlabel, "name 必须是字符串")

        if enforce_whitelist and j["enabled"] and sid not in allowed:
            _fail(jlabel, f"servo {sid} 已启用但不在有效白名单 {sorted(allowed)} 里"
                          f"（白名单是安全边界，先改代码再用）")
        joints.append(j)

    out["joints"] = joints
    return out


def load_config(path: str) -> dict:
    """读取 + 校验配置。返回补全默认值的 dict（原文件不动）。"""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise MimicConfigError(f"读不到配置文件 {p}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MimicConfigError(f"{p} 不是合法 JSON: {exc}") from exc
    return validate_config(data, source_label=str(p))


def save_config(path: str, cfg: dict) -> None:
    """写回配置（阶段 3 标定工具用）。写之前按同一套规则校验。

    注意：`_` 开头的注释键会保留；**手写在文件里的格式（缩进/空行）会被重排**。
    """
    p = Path(path)
    normalized = validate_config(cfg, source_label=str(p))
    p.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")


# ---------------------------------------------------------------------------
# 自测：不碰摄像头/机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _expect_error(cfg: dict, needle: str = "") -> None:
    try:
        validate_config(cfg)
    except MimicConfigError as exc:
        if needle and needle not in str(exc):
            raise AssertionError(f"错误信息里缺少 {needle!r}: {exc}") from exc
        return
    raise AssertionError(f"应当报错但没有：{cfg}")


def _minimal_cfg(joints=None) -> dict:
    return {
        "global": dict(DEFAULT_GLOBAL),
        "smoothing": dict(DEFAULT_SMOOTHING),
        "joints": joints or [{
            "servo_id": 3, "name": "左肘", "enabled": True,
            "source": "elbow_flex", "sign": 1, "scale": 1.0,
            "human_neutral": 0.0, "robot_neutral_tick": 2048,
            "min_deg": -40.0, "max_deg": 40.0,
        }],
    }


def selftest_mimic_config() -> None:
    """模板可加载 + 各类非法配置被拒绝 + 保存往返。"""
    import tempfile

    # 1) 仓库里的模板必须能过
    template = Path(__file__).with_name("mimic_config.json")
    cfg = load_config(str(template))
    _assert(len(cfg["joints"]) >= 4, f"template joints={len(cfg['joints'])}")
    ids = {j["servo_id"] for j in cfg["joints"]}
    _assert(ids == {3, 6, 8, 13}, f"template joint ids={ids}")
    _assert(effective_whitelist(cfg) == ALLOWED_SERVO_IDS,
            f"template whitelist={effective_whitelist(cfg)}")

    # 2) 重复舵机号
    dup = _minimal_cfg()
    dup["joints"].append(copy.deepcopy(dup["joints"][0]))
    _expect_error(dup, "重复")

    # 3) sign 非法
    bad = _minimal_cfg()
    bad["joints"][0]["sign"] = 0
    _expect_error(bad, "sign")

    # 4) min_deg > max_deg
    bad = _minimal_cfg()
    bad["joints"][0]["min_deg"] = 40.0
    bad["joints"][0]["max_deg"] = -40.0
    _expect_error(bad, "min_deg")

    # 5) 未知 source
    bad = _minimal_cfg()
    bad["joints"][0]["source"] = "arm_angle"
    _expect_error(bad, "source")

    # 6) 启用白名单外的舵机 → 报错；禁用则允许
    bad = _minimal_cfg()
    bad["joints"].append({"servo_id": 12, "enabled": True,
                          "source": "head_yaw", "robot_neutral_tick": 2048})
    _expect_error(bad, "白名单")
    ok = _minimal_cfg()
    ok["joints"].append({"servo_id": 12, "enabled": False, "source": None})
    validate_config(ok)

    # 7) 配置白名单不能放宽
    bad = _minimal_cfg()
    bad["global"]["whitelist"] = [3, 12]
    _expect_error(bad, "不能放宽")

    # 8) tick_min 不能低于硬限位
    bad = _minimal_cfg()
    bad["global"]["tick_min"] = 10
    _expect_error(bad, "tick_min")

    # 9) 启用的关节必须有 source
    bad = _minimal_cfg()
    bad["joints"][0]["source"] = None
    _expect_error(bad, "source")

    # 10) save/load 往返（临时目录）
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.json"
        save_config(str(p), cfg)
        back = load_config(str(p))
        _assert(back["joints"][0]["servo_id"] == cfg["joints"][0]["servo_id"],
                "save/load roundtrip lost data")
        _assert("_comment" in back, "comments should survive a roundtrip")

    print("  selftest_mimic_config: OK")


if __name__ == "__main__":
    selftest_mimic_config()
    print("mimic_config selftest: ALL PASSED")
