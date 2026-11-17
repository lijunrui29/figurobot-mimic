"""safety 层：把"想下发的目标 tick"变成"允许下发的目标 tick"。

这一层是**唯一的刹车**，所有写机器人的路径都必须经过它。它做五件事：

1. **白名单**（硬编码）：只放行双臂肩 P/R + 肘 P = 舵机 3,4,6,7,9,11。
   腰/胸/腿/腕/头一律过滤。配置文件里的白名单**只能收紧、不能放宽** ——
   与硬编码集合取交集，写进配置也开不了别的舵机。
2. **全局软限位**：clamp 到 [50, 3600] tick（≈[4.39°, 316.48°]），与
   控制盒/网页控制台同一组数。**每个关节更紧的角度范围由 mapper 负责**
   （它是配置驱动的 `min_deg/max_deg`，相对 robot_neutral 的偏移）；这里只兜底。
3. **限速**：单帧最大步进（默认 6°）+ 最大角速度（默认 45°/s），两者取更紧的。
   三个单位的关系是线性的：`TICKS_PER_DEG = 4095/360`。
4. **看门狗**：超过 `watchdog_s`（默认 0.5 s）没有有效数据 → 不再下发新目标
   （= 保持当前位置，让舵机自己停在最后一个目标上）。
5. **错误熔断**：连续 N 次下游失败（默认 3）→ 整体停止，直到 `reset()`。

本层**不算角度、不读配置、不碰网络**：输入 `{servo_id: 目标 tick | None}`，
输出"可以发"的那部分 + 诊断。`None` 的语义是"本帧没有新目标"（保持），
不是"去 0 位"。

限速依赖"上一次下发的 tick"。第一次见某个舵机时不知道它现在在哪，
此时**直接放行**（记 `seeded=False`）——所以实时驱动在开始前应当用
`seed_current()` 把读到的当前位置塞进来，否则第一个目标可能是一次跳变。
这是阶段 4 的调用责任，本层只如实标注。

纯标准库，可独立测试：python safety.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional

# ---------- 硬编码白名单（改这里 = 改安全边界，必须过用户） ----------
# 2026-11-17 真机实测后收紧：只保留 mimic 实际使用的四个关节。
# 3=左肘、8=右肘、6=右肩外展、13=左肩外展。
# 腕(4,7,9)、头(1,2,11)、腰/胸/腿 全部不在里面。
ALLOWED_SERVO_IDS: FrozenSet[int] = frozenset({3, 6, 8, 13})

# 中文名，只用于日志/标定报告，不参与判断
SERVO_NAMES: Dict[int, str] = {
    3: "左肘", 8: "右肘", 6: "右肩外展", 13: "左肩外展",
}

# ---------- 与控制盒一致的软限位与单位换算 ----------
MIN_TICK = 50            # 约 4.39°
MAX_TICK = 3600          # 约 316.48°
POS_MAX = 4095.0         # 0-4095 对应 0-360°
DEG_MAX = 360.0
TICKS_PER_DEG = POS_MAX / DEG_MAX    # 11.375 tick/度
DEG_PER_TICK = DEG_MAX / POS_MAX     # 0.08791 度/tick


def clamp(value: float, lo: float, hi: float) -> float:
    """数值 clamp；lo > hi 时（配置写反）取 lo，避免静默越界。"""
    if lo > hi:
        return lo
    return lo if value < lo else (hi if value > hi else value)


def deg_to_tick(deg: float) -> float:
    """相对角（度）→ tick 增量。**不含零点**，零点由调用方加。"""
    return float(deg) * TICKS_PER_DEG


def tick_to_deg(tick: float) -> float:
    """tick 增量 → 相对角（度）。**不含零点**。"""
    return float(tick) * DEG_PER_TICK


def clamp_tick(tick: float, lo: float = MIN_TICK, hi: float = MAX_TICK) -> float:
    """全局软限位（控制盒侧同一组数）。"""
    return clamp(float(tick), float(lo), float(hi))


def filter_targets(targets: Dict[int, Optional[float]],
                   whitelist: Iterable[int] = ALLOWED_SERVO_IDS
                   ) -> Dict[int, Optional[float]]:
    """只保留白名单内的目标（写之前强制过滤）。"""
    allowed = frozenset(int(x) for x in whitelist)
    return {int(sid): val for sid, val in targets.items() if int(sid) in allowed}


@dataclass
class SafetyResult:
    """一次 govern 的结果：要发的目标 + 发生了什么。"""
    targets: Dict[int, int] = field(default_factory=dict)
    holding: bool = False            # 看门狗：本帧不下发（保持当前位置）
    tripped: bool = False            # 熔断：整体停止
    rate_limited: Dict[int, bool] = field(default_factory=dict)
    clamped: Dict[int, bool] = field(default_factory=dict)
    filtered: List[int] = field(default_factory=list)   # 白名单外被丢弃的舵机

    def to_debug(self) -> dict:
        return {
            "targets": dict(self.targets),
            "holding": self.holding,
            "tripped": self.tripped,
            "rate_limited": dict(self.rate_limited),
            "clamped": dict(self.clamped),
            "filtered": list(self.filtered),
        }


class SafetyGovernor:
    """白名单 / 限位 / 限速 / 看门狗 / 熔断的集中实现。

    典型用法（每帧）：
        res = gov.update(mapper_targets, now, data_valid=pose is not None)
        link.send_targets(res.targets)      # 只发 res.targets
        gov.notify_result(ok)               # 每次下发后回报成功/失败

    **一个数据源一个实例**（它持有"上次下发值"这类跨帧状态）。
    """

    def __init__(self, whitelist: Optional[Iterable[int]] = None,
                 min_tick: int = MIN_TICK, max_tick: int = MAX_TICK,
                 default_max_step_deg: float = 6.0,
                 default_max_speed_deg_s: float = 45.0,
                 watchdog_s: float = 0.5,
                 max_consecutive_errors: int = 3) -> None:
        """
        参数:
            whitelist: `None` = 用硬编码的 ALLOWED_SERVO_IDS；传入集合会与它
                       **取交集**（配置只能收紧，不能放宽）。
            min_tick/max_tick: 全局软限位（tick）。
            default_max_step_deg: 单帧最大步进（度），可被 configure_joint 覆盖。
            default_max_speed_deg_s: 最大角速度（度/秒），可被覆盖。
            watchdog_s: 多久没有有效数据就停止下发（秒）。
            max_consecutive_errors: 连续多少次下游失败就熔断。
        """
        if whitelist is None:
            self.whitelist: FrozenSet[int] = ALLOWED_SERVO_IDS
        else:
            self.whitelist = frozenset(int(x) for x in whitelist) & ALLOWED_SERVO_IDS
        self.min_tick = int(min_tick)
        self.max_tick = int(max_tick)
        self.default_max_step_deg = float(default_max_step_deg)
        self.default_max_speed_deg_s = float(default_max_speed_deg_s)
        self.watchdog_s = float(watchdog_s)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.reset()

    # ---------- 配置 ----------

    def configure_joint(self, servo_id: int, max_step_deg: Optional[float] = None,
                        max_speed_deg_s: Optional[float] = None) -> None:
        """给单个关节收紧步进/速度（None = 用全局默认）。"""
        sid = int(servo_id)
        if max_step_deg is not None:
            self._max_step_deg[sid] = max(0.0, float(max_step_deg))
        if max_speed_deg_s is not None:
            self._max_speed_deg_s[sid] = max(0.0, float(max_speed_deg_s))

    def seed_current(self, positions: Dict[int, int],
                     now: Optional[float] = None) -> None:
        """把**读到的当前位置**作为限速起点。

        实时驱动开始前必须先调这个：否则第一次 `update()` 不知道手臂在哪，
        会直接放行到目标位置（可能是一次大跳变）。
        """
        ts = now if now is not None else 0.0
        for sid, tick in positions.items():
            sid = int(sid)
            self._last_cmd[sid] = int(clamp_tick(tick, self.min_tick, self.max_tick))
            self._last_cmd_ts[sid] = float(ts)

    # ---------- 主流程 ----------

    def update(self, targets: Dict[int, Optional[float]], now: float,
               data_valid: bool = True) -> SafetyResult:
        """输入"期望目标 tick"，返回"允许下发 tick"。

        targets: {servo_id: 目标 tick | None}。`None` = 本帧没有新目标（保持）。
        now:     单调时钟（秒）。
        data_valid: 本帧是否有有效姿态数据（没人/低置信度 → False）。
        """
        res = SafetyResult()
        if self._tripped:
            res.tripped = True
            return res

        # --- 看门狗 ---
        if data_valid:
            self._last_valid = float(now)
        if (self._last_valid is None
                or float(now) - self._last_valid > self.watchdog_s):
            res.holding = True
            return res

        # --- 逐舵机：白名单 → 限位 → 限速 ---
        for raw_sid, raw_target in targets.items():
            sid = int(raw_sid)
            if sid not in self.whitelist:
                res.filtered.append(sid)
                continue
            if raw_target is None:
                continue                     # 保持，不发
            tick = clamp_tick(float(raw_target), self.min_tick, self.max_tick)
            res.clamped[sid] = abs(tick - float(raw_target)) > 1e-6

            prev = self._last_cmd.get(sid)
            limited = False
            if prev is not None:
                dt = float(now) - self._last_cmd_ts.get(sid, float(now))
                if dt <= 0.0:
                    limit_tick = 0.0     # 同一时刻重复调用：不许动
                else:
                    max_step = self._max_step_deg.get(
                        sid, self.default_max_step_deg) * TICKS_PER_DEG
                    max_speed = (self._max_speed_deg_s.get(
                        sid, self.default_max_speed_deg_s) * dt * TICKS_PER_DEG)
                    limit_tick = min(max_step, max_speed)
                delta = tick - prev
                if abs(delta) > limit_tick:
                    tick = prev + (limit_tick if delta > 0 else -limit_tick)
                    limited = True
            # 四舍五入到整数 tick（协议是整数）
            out = int(round(tick))
            # 再夹一次：round 不会越界，但半径 0.5 tick 的舍入要守住边界
            out = int(clamp(out, self.min_tick, self.max_tick))
            res.targets[sid] = out
            res.rate_limited[sid] = limited
            self._last_cmd[sid] = out
            self._last_cmd_ts[sid] = float(now)
        return res

    def notify_result(self, ok: bool) -> None:
        """下游每下发一次回报一次；连续失败到阈值就熔断。"""
        if ok:
            self._consecutive_errors = 0
            return
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.max_consecutive_errors:
            self.trip(f"{self._consecutive_errors} consecutive downstream errors")

    def trip(self, reason: str = "emergency stop") -> None:
        """立即停止一切下发（急停 / ESC / 不可恢复错误）。"""
        self._tripped = True
        self._trip_reason = str(reason)

    def reset(self, clear_positions: bool = True) -> None:
        """解除熔断 / 看门狗状态。

        clear_positions=True（默认）连"上次下发值"一起清掉 —— 急停后手臂
        可能已经被重力带走了旧位置，继续拿旧值做限速基准是错的，应重新
        seed_current()。只有明确知道机器人没动时才传 False。
        """
        self._last_valid: Optional[float] = None
        self._tripped = False
        self._trip_reason: Optional[str] = None
        self._consecutive_errors = 0
        self._max_step_deg: Dict[int, float] = {}
        self._max_speed_deg_s: Dict[int, float] = {}
        if clear_positions:
            self._last_cmd: Dict[int, int] = {}
            self._last_cmd_ts: Dict[int, float] = {}

    # ---------- 只读属性 ----------
    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> Optional[str]:
        return self._trip_reason

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors


# ---------------------------------------------------------------------------
# 自测：纯数值，不碰机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest_safety() -> None:
    """白名单 / 限位 / 限速 / 看门狗 / 熔断 / reset 六组用例。"""
    # 1) 白名单：12（腰Y）必须被丢弃，3（左肘）必须保留
    gov = SafetyGovernor()
    res = gov.update({3: 2048, 12: 2048}, now=0.0)
    _assert(12 in res.filtered, "servo 12 must be filtered by whitelist")
    _assert(3 in res.targets, "servo 3 must pass whitelist")
    _assert(res.filtered == [12], f"filtered={res.filtered}")

    # 2) 配置里的白名单只能收紧：传 {3,12} 后 3 还在、12 永远进不来
    gov = SafetyGovernor(whitelist={3, 12})
    _assert(gov.whitelist == frozenset({3}),
            f"whitelist intersection wrong: {gov.whitelist}")

    # 3) 全局软限位
    gov = SafetyGovernor()
    res = gov.update({3: 10.0, 6: 4000.0}, now=0.0)
    _assert(res.targets[3] == MIN_TICK, f"low clamp: {res.targets[3]}")
    _assert(res.targets[6] == MAX_TICK, f"high clamp: {res.targets[6]}")
    _assert(res.clamped.get(3) and res.clamped.get(6), "clamped flags missing")

    # 4) 限速：seed 2048，0.0333s 后给 3000
    #    单帧上限 6° = 68.25 tick；速度上限 45°/s × 0.0333s = 1.5° = 17.06 tick
    #    → 取更紧的 17 tick
    gov = SafetyGovernor()
    gov.seed_current({3: 2048}, now=0.0)
    res = gov.update({3: 3000}, now=1.0 / 30.0)
    _assert(res.targets[3] == 2048 + 17,
            f"speed/step limit should move 17 ticks, got {res.targets[3]}")
    _assert(res.rate_limited[3], "rate_limited flag missing")

    # 5) 限速不该拦住小步
    gov = SafetyGovernor()
    gov.seed_current({3: 2048}, now=0.0)
    res = gov.update({3: 2050}, now=1.0 / 30.0)
    _assert(res.targets[3] == 2050 and not res.rate_limited[3],
            f"small step should pass, got {res.targets[3]}")

    # 6) 看门狗：0.4s 内仍下发，超过 0.5s 停止
    gov = SafetyGovernor()
    gov.update({3: 2048}, now=0.0, data_valid=True)
    res = gov.update({3: 2048}, now=0.4, data_valid=False)
    _assert(3 in res.targets and not res.holding, "watchdog fired too early")
    res = gov.update({3: 2048}, now=0.6, data_valid=False)
    _assert(res.holding and not res.targets, "watchdog should hold after 0.5s")

    # 7) 熔断：连续 3 次失败后停止；reset 后恢复
    gov = SafetyGovernor()
    gov.seed_current({3: 2048}, now=0.0)
    _assert(3 in gov.update({3: 2048}, now=0.01).targets, "should still send")
    gov.notify_result(False)
    gov.notify_result(False)
    _assert(not gov.tripped, "should not trip after 2 errors")
    gov.notify_result(False)
    _assert(gov.tripped, "should trip after 3 errors")
    res = gov.update({3: 2048}, now=0.02)
    _assert(res.tripped and not res.targets, "tripped governor must not send")
    gov.reset()
    res = gov.update({3: 2100}, now=0.03)
    _assert(not res.tripped and 3 in res.targets, "reset should resume sending")
    _assert(res.targets[3] == 2100, "reset(clear_positions) drops the rate base")

    # 8) trip/reset 与 notify_result(True) 清零
    gov = SafetyGovernor()
    gov.notify_result(False)
    gov.notify_result(False)
    gov.notify_result(True)
    gov.notify_result(False)
    _assert(not gov.tripped, "success should clear the consecutive error count")

    # 9) 单位换算往返
    _assert(abs(tick_to_deg(deg_to_tick(37.5)) - 37.5) < 1e-9,
            "deg<->tick roundtrip")

    print("  selftest_safety: OK")


if __name__ == "__main__":
    selftest_safety()
    print("safety selftest: ALL PASSED")
