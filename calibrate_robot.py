"""calibrate_robot 工具：阶段 3 的机器人侧标定（只做这一步，不进入阶段 4）。

它只做四件事，**一条命令只做一步**：

    status   只读：桥接 / safety / 各关节 raw·zero·配置中立位
    reset    Motion_Reset（**会让机器人动**，必须 --yes）
    neutral  等姿态稳定后完整回读，把 27 个 raw tick 存成中立位快照
    probe    单关节小幅点动（**会让机器人动**，必须 --yes），用来判断 sign
    apply    把中立位快照写回 mimic_config.json；只对显式给出 --sign 的关节
             置 calibrated=true（不动机器人）

为什么不做成"一键跑完"：
    复位和点动都会让真机动起来，而 sign 的方向必须由人在现场看（tick 增大时
    手臂是向外抬还是向内收）。所以本工具每次运行只做一个动作，运动前由人
    确认；工具不会自己连续点动。

典型顺序（每一步之间由人确认）：
    python calibrate_robot.py status
    python calibrate_robot.py reset --yes
    python calibrate_robot.py neutral
    python calibrate_robot.py probe --joint 3 --delta 60 --yes
    python calibrate_robot.py probe --joint 8 --delta 60 --yes
    python calibrate_robot.py probe --joint 6 --delta 60 --yes
    python calibrate_robot.py probe --joint 13 --delta 60 --yes
    python calibrate_robot.py apply --sign 3=+1 --sign 8=-1 --sign 6=+1 --sign 13=+1

    （2026-11-17 实测：3=左肘、8=右肘、6=右肩外展、13=左肩前后 pitch（侧视）；
      4/7/9 是腕、11 是头，已移出白名单。）

安全边界：
    * 只允许白名单 ALLOWED_SERVO_IDS（3/6/8/13）里的关节；
    * 点动步长 |delta| <= MAX_PROBE_TICKS（默认 120 tick ≈ 10.5°）；
    * 目标 tick 夹在配置的 global.tick_min/tick_max 内；
    * safety.level == danger / 关节离线 → 拒绝运动；
    * 运动命令必须显式 --yes，否则只打印"将要做什么"然后退出。

依赖：robot_link（HTTP 传输）+ mimic_config（配置读写）+ safety（白名单/单位）。
不 import cv2 / pose_estimation，不需要摄像头。

可独立测试：python calibrate_robot.py --selftest（不碰机器人）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from mimic_config import (MimicConfigError, get_joint, load_config,  # noqa: E402
                          save_config)
from robot_link import (DEFAULT_BASE_URL, RobotLink,               # noqa: E402
                        RobotLinkError)
from safety import ALLOWED_SERVO_IDS, TICKS_PER_DEG, clamp_tick    # noqa: E402

DEFAULT_SNAPSHOT = SCRIPT_DIR / "data" / "robot_neutral.json"
DEFAULT_TIMEOUT = 10.0          # 完整回读实测约 5~6s，统一给足
MAX_PROBE_TICKS = 120           # ≈10.5°，够肉眼分辨方向，又不会大幅动
STABLE_TOL_TICKS = 8            # 相邻两次回读最大差 <= 8 tick 视为稳定
STABLE_INTERVAL_S = 0.5
STABLE_TIMEOUT_S = 10.0


class CalibError(RuntimeError):
    """标定前置条件不满足 / 桥接返回异常。"""


class _CalibLink(RobotLink):
    """robot_link 的阶段 3 扩展：补两个桥接已有、但 RobotLink 未暴露的端点。

    放在本文件里而不是改 robot_link，保持"阶段 3 只动 calibrate_robot.py"。
    """

    def reset_motion(self) -> dict:
        """POST /api/reset（Motion_Reset）——**会让机器人动**。"""
        return self._request("POST", "/api/reset", {})

    def joint_enable(self, servo_id: int, enable: bool) -> dict:
        """POST /api/joint/enable。是否使能不能由本层替你决定，只负责发。"""
        return self._request("POST", "/api/joint/enable",
                             {"id": int(servo_id), "enable": bool(enable)})


# ---------------------------------------------------------------------------
# 通用检查
# ---------------------------------------------------------------------------

def _require_yes(args: argparse.Namespace, what: str) -> None:
    """运动类命令的硬闸：没有 --yes 就只说明将要做什么，然后退出。"""
    if getattr(args, "yes", False):
        return
    print(f"[calib] 这一步会让机器人动：{what}")
    print("[calib] 未带 --yes，什么都没有做。确认现场安全后再加 --yes 重跑。")
    sys.exit(3)


def _safety_level(pose: dict) -> str:
    return str((pose.get("safety") or {}).get("level") or "unknown")


def _positions_from_pose(pose: dict) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for key, joint in (pose.get("joints") or {}).items():
        if not isinstance(joint, dict) or not joint.get("online"):
            continue
        if joint.get("raw") is None:
            continue
        try:
            out[int(key)] = int(joint["raw"])
        except (TypeError, ValueError):
            continue
    return out


def _read_ready(link: RobotLink, require_ids=()) -> tuple:
    """连接 + safety + 必需关节在线。返回 (pose, positions)。"""
    status = link.status()
    if not status.get("connected"):
        raise CalibError(f"桥接在线但机器人未连接：{status}")
    pose = link.read_pose(fast=True)
    level = _safety_level(pose)
    if level not in ("safe", "warn"):
        issues = (pose.get("safety") or {}).get("issues")
        raise CalibError(f"当前姿态 safety={level}（{issues}），拒绝任何动作。"
                         "先人工处理超限/极限位/离线关节")
    positions = _positions_from_pose(pose)
    missing = [sid for sid in require_ids if int(sid) not in positions]
    if missing:
        raise CalibError(f"这些关节回读不到位置：{missing}。"
                         "确认在线、串口没被 motion_main 独占")
    return pose, positions


def _wait_stable(link: RobotLink, timeout: float = STABLE_TIMEOUT_S,
                 interval: float = STABLE_INTERVAL_S,
                 tol: int = STABLE_TOL_TICKS) -> tuple:
    """轮询回读，直到相邻两次最大变化 <= tol（连续 2 次）或超时。

    返回 (最后一次 positions, stable: bool, 最大变化 tick 或 None)。
    读取失败不抛：真机运动期间串口可能被 motion_main 占用，下一轮继续。
    """
    t0 = time.monotonic()
    prev = None
    stable_hits = 0
    max_diff: Optional[int] = None
    while True:
        try:
            cur = link.read_positions(fast=True)
        except RobotLinkError as exc:
            cur = None
            print(f"[calib] 等待稳定时回读失败（继续重试）：{exc}")
        if cur:
            if prev:
                common = set(prev) & set(cur)
                if common:
                    max_diff = max(abs(cur[s] - prev[s]) for s in common)
                    if max_diff <= tol:
                        stable_hits += 1
                        if stable_hits >= 3:
                            return cur, True, max_diff
                    else:
                        stable_hits = 0
            prev = cur
        if time.monotonic() - t0 >= timeout:
            return (prev or {}), False, max_diff
        time.sleep(interval)


def _fmt_cfg_joint(cfg: dict, sid: int) -> str:
    joint = get_joint(cfg, sid)
    if joint is None:
        return "（配置里没有这个关节）"
    flags = []
    if joint.get("enabled"):
        flags.append("enabled")
    flags.append("calibrated" if joint.get("calibrated") else "UNCAL")
    return (f"{joint.get('name', '')} sign={joint.get('sign'):+d} "
            f"neutral={joint.get('robot_neutral_tick')} "
            f"({','.join(flags)})")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    link = _CalibLink(args.bridge, args.timeout)
    pose, positions = _read_ready(link)
    safety = pose.get("safety") or {}
    print(f"[calib] 桥接 {args.bridge}：connected=True "
          f"serial={link.status().get('serial')}")
    print(f"[calib] safety={safety.get('level')} "
          f"online={safety.get('online_count')}/{safety.get('total')}")
    zeros = {int(k): int(v) for k, v in (pose.get("zero_values") or {}).items()}
    joints = pose.get("joints") or {}

    print(f"[calib] {'id':>2} {'name':<7}{'raw':>6}{'angle':>8}{'zero':>6}"
          f"{'neutral':>8}{'raw-zero':>9}  cfg")
    for key in sorted(joints, key=lambda k: int(k)):
        sid = int(key)
        joint = joints[key]
        cfg_joint = get_joint(cfg, sid)
        name = (cfg_joint or {}).get("name", "")
        raw = joint.get("raw")
        zero = zeros.get(sid)
        neutral = (cfg_joint or {}).get("robot_neutral_tick")
        delta = "--" if raw is None or zero is None else f"{raw - zero:+d}"
        raw_s = "--" if raw is None else str(raw)
        zero_s = "--" if zero is None else str(zero)
        neutral_s = "--" if neutral is None else str(neutral)
        angle = joint.get("angle")
        flags = []
        if cfg_joint is not None:
            if cfg_joint.get("enabled"):
                flags.append("enabled")
            if not cfg_joint.get("calibrated"):
                flags.append("UNCAL")
        print(f"[calib] {sid:>2} {name:<7}{raw_s:>6}"
              f"{('--' if angle is None else f'{float(angle):.1f}'):>8}"
              f"{zero_s:>6}{neutral_s:>8}{delta:>9}  "
              + (",".join(flags) if flags else ""))

    if Path(args.snapshot).exists():
        info = _load_snapshot(args.snapshot, strict=False)
        print(f"[calib] 中立位快照：{args.snapshot}"
              f"（{info.get('created_at', '?')}，"
              f"{len(info.get('positions') or {})} 个关节）")
    else:
        print(f"[calib] 中立位快照：{args.snapshot} 还不存在（先 reset → neutral）")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def cmd_reset(args: argparse.Namespace) -> None:
    _require_yes(args, "Motion_Reset（会让机器人回到安全姿态）")
    load_config(args.config)                # 提前校验配置，避免复位后才发现配置坏了
    link = _CalibLink(args.bridge, args.timeout)
    pose, _ = _read_ready(link)
    print(f"[calib] reset 前 safety={_safety_level(pose)}")
    print("[calib] 发送 Motion_Reset ...")
    resp = link.reset_motion()
    print(f"[calib] 桥接返回：{resp}")
    if args.settle > 0:
        print(f"[calib] 先等 {args.settle:g}s 让复位动作启动 ...")
        time.sleep(args.settle)
    print(f"[calib] 等待姿态稳定（最多 {STABLE_TIMEOUT_S:g}s）...")
    positions, stable, max_diff = _wait_stable(link)
    print(f"[calib] {'稳定' if stable else '未确认稳定'}"
          f"（相邻回读最大变化 {max_diff if max_diff is not None else '--'} tick）")
    if positions:
        arms = {sid: positions[sid] for sid in (3, 4, 6, 7, 9, 11)
                if sid in positions}
        print(f"[calib] 臂部回读：{arms}")
    print("[calib] 下一步：python calibrate_robot.py neutral")


# ---------------------------------------------------------------------------
# neutral
# ---------------------------------------------------------------------------

def _load_snapshot(path: Path, strict: bool = True) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if strict:
            raise CalibError(f"读不到中立位快照 {path}：{exc}") from exc
        return {}
    if strict and not isinstance(data.get("positions"), dict):
        raise CalibError(f"{path} 里没有 positions，不像中立位快照")
    return data if isinstance(data, dict) else {}


def _print_neutral_table(cfg: dict, positions: Dict[int, int]) -> None:
    print(f"[calib] {'id':>2} {'name':<7}{'neutral':>8}{'old':>6}  cfg")
    for key in sorted(positions):
        sid = int(key)
        cfg_joint = get_joint(cfg, sid)
        name = (cfg_joint or {}).get("name", "")
        old = (cfg_joint or {}).get("robot_neutral_tick")
        print(f"[calib] {sid:>2} {name:<7}{positions[sid]:>8}"
              f"{('--' if old is None else str(old)):>6}  {_fmt_cfg_joint(cfg, sid)}")


def cmd_neutral(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    link = _CalibLink(args.bridge, args.timeout)
    pose, _ = _read_ready(link)
    print(f"[calib] 当前 safety={_safety_level(pose)}；等待稳定 ...")
    positions, stable, max_diff = _wait_stable(link)
    print(f"[calib] {'稳定' if stable else '未确认稳定'}"
          f"（最大变化 {max_diff if max_diff is not None else '--'} tick）")
    if not stable and not args.force:
        raise CalibError("姿态未稳定，拒绝写中立位快照。等机器人停稳后重跑，"
                         "或确认没问题时加 --force")

    # 稳定后做一次完整回读（fast=False）：拿完整的 safety / zero_values / raw
    pose = link.read_pose(fast=False)
    positions = _positions_from_pose(pose)
    if not positions:
        raise CalibError("完整回读没有拿到任何关节位置")
    print(f"[calib] 完整回读 safety={_safety_level(pose)}，"
          f"{len(positions)} 个关节在线")

    zeros = {int(k): int(v) for k, v in (pose.get("zero_values") or {}).items()}
    angles = {int(k): (None if v.get("angle") is None else float(v["angle"]))
              for k, v in (pose.get("joints") or {}).items()
              if isinstance(v, dict)}
    snapshot = {
        "schema_version": "robot-neutral-0.1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bridge": args.bridge,
        "serial": link.status().get("serial"),
        "safety": pose.get("safety"),
        "zero_values": zeros,
        "positions": {str(sid): int(tick) for sid, tick in sorted(positions.items())},
        "angles": {str(sid): angle for sid, angle in sorted(angles.items())},
        "config_before": {
            str(j["servo_id"]): {
                "name": j.get("name"),
                "enabled": bool(j.get("enabled")),
                "robot_neutral_tick": j.get("robot_neutral_tick"),
                "calibrated": bool(j.get("calibrated")),
            } for j in cfg["joints"]
        },
    }
    out = Path(args.snapshot)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(out)
    print(f"[calib] 已保存中立位快照 → {out}")
    _print_neutral_table(cfg, positions)
    print("[calib] 下一步：逐个 probe 关节判断 sign")


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

_SIGN_HINTS = {
    3: "tick 增大 = 左小臂向上/前抬起（肘弯）→ sign=+1；伸直 → sign=-1",
    8: "tick 增大 = 右小臂伸直 → sign=-1；向上/前抬起（肘弯）→ sign=+1",
    6: "tick 增大 = 右臂向内/向下 → sign=+1；tick 减小 = 右臂向外抬（外展）"
       "（人体右臂外展的有符号角为负，负角 → tick 减小，故 sign=+1）",
    13: "tick 增大 = 左臂向后；tick 减小 = 向前（侧视 flexion 源）。"
        "人左臂向后时机器人也向后 → sign=+1；相反则 -1",
}


def _clamp_target(target: float, cfg: dict) -> int:
    g = cfg["global"]
    return int(round(clamp_tick(target, g["tick_min"], g["tick_max"])))


def cmd_probe(args: argparse.Namespace) -> None:
    sid = int(args.joint)
    if sid not in ALLOWED_SERVO_IDS:
        raise CalibError(f"关节 {sid} 不在硬编码白名单 {sorted(ALLOWED_SERVO_IDS)} 里")
    if args.delta == 0:
        raise CalibError("--delta 不能为 0")
    if abs(args.delta) > MAX_PROBE_TICKS:
        raise CalibError(f"|--delta| 不能超过 {MAX_PROBE_TICKS} tick"
                         f"（≈{MAX_PROBE_TICKS / TICKS_PER_DEG:.1f}°）")

    cfg = load_config(args.config)
    link = _CalibLink(args.bridge, args.timeout)
    pose, positions = _read_ready(link, require_ids=(sid,))
    before = int(positions[sid])
    target = _clamp_target(before + args.delta, cfg)
    actual_delta = target - before
    if actual_delta == 0:
        raise CalibError(f"目标被限位夹回原位（before={before}，"
                         f"tick 范围 {cfg['global']['tick_min']}-{cfg['global']['tick_max']}）")

    joint_cfg = get_joint(cfg, sid)
    name = (joint_cfg or {}).get("name", "")
    what = (f"{sid} 号 {name}：{before} → {target} tick"
            f"（{actual_delta:+d}，≈{actual_delta / TICKS_PER_DEG:+.1f}°），"
            f"保持 {args.hold:g}s 后"
            + ("保持不动（--no-return）" if args.no_return else "回原位"))
    _require_yes(args, what)
    print(f"[calib] {what}")
    print(f"[calib] 观察提示：{_SIGN_HINTS.get(sid, '看关节朝哪个方向动')}")

    if args.enable:
        resp = link.joint_enable(sid, True)
        print(f"[calib] 已请求使能：{resp}")

    during = None
    returned = None
    try:
        wrote = link.write_joint(sid, target)
        if not wrote.get("ok"):
            raise CalibError(f"写目标失败：{wrote}")
        print(f"[calib] 已下发 target={target}，保持 {args.hold:g}s ...")
        time.sleep(args.hold)
        try:
            during = link.read_positions(fast=True).get(sid)
        except RobotLinkError as exc:
            print(f"[calib] 点动后回读失败：{exc}")

        if not args.no_return:
            back = link.write_joint(sid, before)
            if not back.get("ok"):
                raise CalibError(f"回原位失败，机器人当前停在 {target} tick：{back}")
            time.sleep(args.return_hold)
            try:
                returned = link.read_positions(fast=True).get(sid)
            except RobotLinkError as exc:
                print(f"[calib] 回位后回读失败：{exc}")
    finally:
        if args.disable_after:
            try:
                resp = link.joint_enable(sid, False)
                print(f"[calib] 已请求失能：{resp}")
            except RobotLinkError as exc:
                print(f"[calib] 失能失败（需要人工确认关节状态）：{exc}")

    print(f"[calib] before={before} target={target} during="
          f"{during if during is not None else '--'} "
          f"returned={returned if returned is not None else '--'}")
    if during is None:
        print("[calib] 没读到点动后的值，方向无法确认")
    else:
        moved = during - before
        print(f"[calib] 实际变化 {moved:+d} tick（目标 {actual_delta:+d}）")
        if abs(moved) < max(5, abs(actual_delta) // 3):
            print("[calib] 注意：几乎没动。可能没使能、被 idle 动作抢占、"
                  "或机械上卡住——先别下结论，检查后再试")
    print("[calib] 把观察到的方向记下来，最后用 apply --sign 写入")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _parse_signs(specs) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for spec in specs or []:
        text = str(spec).replace("=", ":").strip()
        parts = text.split(":")
        if len(parts) != 2:
            raise CalibError(f"--sign 格式应为 SID=+1/-1，实际 {spec!r}")
        try:
            sid, value = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise CalibError(f"--sign 格式应为 SID=+1/-1，实际 {spec!r}") from exc
        if value not in (-1, 1):
            raise CalibError(f"--sign {sid} 只能是 +1 或 -1，实际 {value}")
        out[sid] = value
    return out


def cmd_apply(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    snap = _load_snapshot(Path(args.snapshot))
    positions = {int(k): int(v) for k, v in snap["positions"].items()}
    signs = _parse_signs(args.sign)

    for sid in signs:
        joint = get_joint(cfg, sid)
        if joint is None:
            print(f"[calib] 警告：--sign 指定了配置里没有的关节 {sid}，忽略")
        elif not joint.get("enabled"):
            print(f"[calib] 警告：关节 {sid} 在配置里是 disabled，sign 照样写入但不会下发")

    print(f"[calib] 中立位快照：{args.snapshot}（{snap.get('created_at', '?')}）")
    print(f"[calib] {'id':>2} {'name':<7}{'neutral':>8}{'old':>6}{'sign':>6}  status")
    changed = []
    for joint in cfg["joints"]:
        sid = int(joint["servo_id"])
        if not joint.get("enabled"):
            continue
        if sid not in positions:
            raise CalibError(f"快照里没有启用关节 {sid}（{joint.get('name')}）的 raw tick")
        new = int(positions[sid])
        if not (0 <= new <= 4095):
            raise CalibError(f"关节 {sid} 的快照 tick={new} 越界")
        old = joint.get("robot_neutral_tick")
        joint["robot_neutral_tick"] = new
        if sid in signs:
            joint["sign"] = int(signs[sid])
            joint["calibrated"] = True
        changed.append((sid, joint.get("name", ""), old, new,
                        joint.get("sign"), bool(joint.get("calibrated"))))
        print(f"[calib] {sid:>2} {joint.get('name', ''):<7}{new:>8}"
              f"{('--' if old is None else str(old)):>6}"
              f"{joint.get('sign'):>+6}  "
              + ("calibrated" if joint.get("calibrated") else "仍 UNCAL（没给 --sign）"))

    enabled = [j for j in cfg["joints"] if j.get("enabled")]
    remaining = [f"{j['name']}({j['servo_id']})" for j in enabled
                 if not j.get("calibrated")]
    if remaining:
        print("[calib] 仍未标定（--live 会继续拒绝）：" + "、".join(remaining))

    if args.dry_run:
        print("[calib] --dry-run：配置没有写回")
        return
    save_config(args.config, cfg)
    print(f"[calib] 已写回 {args.config}")
    print("[calib] 提醒：human_neutral 仍要在运行时按 n 采集；"
          "robot_neutral_tick 只是机器人侧零点")


# ---------------------------------------------------------------------------
# 自测：不碰机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest() -> None:
    import copy
    import tempfile

    cfg = load_config(str(SCRIPT_DIR / "mimic_config.json"))

    # 1) sign 解析
    signs = _parse_signs(["3=+1", "8:-1", "6=+1"])
    _assert(signs == {3: 1, 8: -1, 6: 1}, f"signs={signs}")
    for bad in ("3", "3=0", "x=1", "3=+1=2"):
        try:
            _parse_signs([bad])
        except CalibError:
            pass
        else:
            raise AssertionError(f"bad sign spec accepted: {bad}")

    # 2) 目标夹限位
    _assert(_clamp_target(2044 + 60, cfg) == 2104, "target should be exact")
    _assert(_clamp_target(10, cfg) == cfg["global"]["tick_min"], "low clamp")
    _assert(_clamp_target(4000, cfg) == cfg["global"]["tick_max"], "high clamp")

    # 3) apply 的纯逻辑：用临时配置 + 假快照
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cpath = td / "cfg.json"
        cpath.write_text(json.dumps(load_config(str(SCRIPT_DIR / "mimic_config.json")),
                                    ensure_ascii=False, indent=2), encoding="utf-8")
        spath = td / "neutral.json"
        spath.write_text(json.dumps({
            "schema_version": "robot-neutral-0.1",
            "created_at": "test",
            "positions": {"3": 1981, "6": 2200, "8": 2133, "13": 1178},
        }), encoding="utf-8")
        args = argparse.Namespace(config=str(cpath), snapshot=str(spath),
                                  sign=["3=+1", "8=-1", "6=+1", "13=-1"],
                                  dry_run=False)
        cmd_apply(args)
        back = load_config(str(cpath))
        for sid, expect in ((3, 1981), (6, 2200), (8, 2133), (13, 1178)):
            joint = get_joint(back, sid)
            _assert(joint["robot_neutral_tick"] == expect,
                    f"servo {sid} neutral={joint['robot_neutral_tick']}")
            _assert(joint["calibrated"] is True, f"servo {sid} should be calibrated")
        _assert(get_joint(back, 13)["sign"] == -1, "left abduction sign")
        _assert(get_joint(back, 3)["sign"] == 1, "left elbow sign")

        # 4) 没给 --sign 的关节保持 UNCAL（构造一份全未标定的模板）
        base = load_config(str(SCRIPT_DIR / "mimic_config.json"))
        for joint in base["joints"]:
            joint["calibrated"] = False
            joint["robot_neutral_tick"] = None
            joint["sign"] = 1
        cpath2 = td / "cfg2.json"
        cpath2.write_text(json.dumps(base, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        args2 = copy.deepcopy(args)
        args2.config = str(cpath2)
        args2.sign = ["3=+1"]
        cmd_apply(args2)
        back2 = load_config(str(cpath2))
        _assert(get_joint(back2, 3)["calibrated"] is True, "3 should be calibrated")
        _assert(get_joint(back2, 6)["calibrated"] is False, "6 should stay UNCAL")

        # 5) 快照缺启用关节 → 拒绝
        bad = td / "bad.json"
        bad.write_text(json.dumps({"positions": {"3": 1981}}), encoding="utf-8")
        args3 = copy.deepcopy(args)
        args3.snapshot = str(bad)
        try:
            cmd_apply(args3)
        except CalibError as exc:
            _assert("快照里没有启用关节" in str(exc), f"unexpected: {exc}")
        else:
            raise AssertionError("incomplete snapshot should be rejected")

    print("  selftest_calibrate_robot: OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="阶段 3 机器人侧标定（复位 / 中立位回读 / 单关节点动 / 写回配置）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=str(SCRIPT_DIR / "mimic_config.json"),
                   help="映射配置")
    p.add_argument("--bridge", default=DEFAULT_BASE_URL,
                   help="figurobot 桥接地址")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="单次桥接请求超时（完整回读约 5~6s）")
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT),
                   help="中立位快照路径（neutral 写、apply 读、status 显示）")
    p.add_argument("--selftest", action="store_true",
                   help="纯软件自测（不碰机器人）")

    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("status", help="只读：桥接 / safety / 关节回读")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("reset", help="Motion_Reset（会让机器人动，必须 --yes）")
    s.add_argument("--yes", action="store_true", help="确认现场安全，执行复位")
    s.add_argument("--settle", type=float, default=1.5,
                   help="发送复位后先等多少秒，再开始判断是否稳定")
    s.set_defaults(func=cmd_reset)

    s = sub.add_parser("neutral", help="等稳后完整回读并保存中立位快照（不动机器人）")
    s.add_argument("--force", action="store_true", help="姿态未稳定也强制保存")
    s.set_defaults(func=cmd_neutral)

    s = sub.add_parser("probe", help="单关节小幅点动（会让机器人动，必须 --yes）")
    s.add_argument("--joint", type=int, required=True,
                   help="舵机 ID（白名单 3/6/8/13）")
    s.add_argument("--delta", type=int, default=60,
                   help=f"相对当前位置的 tick 偏移，|delta| <= {MAX_PROBE_TICKS}")
    s.add_argument("--hold", type=float, default=1.5,
                   help="到目标后保持多少秒，便于观察")
    s.add_argument("--return-hold", type=float, default=0.8,
                   help="回原位后等多少秒再回读")
    s.add_argument("--no-return", action="store_true",
                   help="不自动回原位（默认会回）")
    s.add_argument("--enable", action="store_true",
                   help="点动前请求使能该关节（不确定是否已使能时用）")
    s.add_argument("--disable-after", action="store_true",
                   help="点动结束后请求失能（与 --enable 配对）")
    s.add_argument("--yes", action="store_true", help="确认现场安全，执行点动")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("apply", help="把中立位快照写回配置（不动机器人）")
    s.add_argument("--sign", action="append", default=[],
                   help="SID=+1/-1，可重复；给了 sign 的关节才置 calibrated=true")
    s.add_argument("--dry-run", action="store_true", help="只打印，不写回配置")
    s.set_defaults(func=cmd_apply)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        print("calibrate_robot selftest: ALL PASSED")
        return
    if not getattr(args, "func", None):
        args.cmd = "status"
        args.func = cmd_status
    try:
        args.func(args)
    except (CalibError, MimicConfigError, RobotLinkError) as exc:
        print(f"[calib] 失败：{exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
