"""one_euro 层：One Euro Filter（Casiez et al., CHI 2012）—— 低延迟自适应低通。

为什么在 EMA（smoothing.AngleSmoother）之外再要一个滤波器：
    EMA 的 alpha 是固定的，抖动压得越狠、跟随就越慢；对人机模仿这种
    "手一动机器人就要跟"的链路，固定 alpha 只能二选一。One Euro 把
    "截止频率"跟信号速度挂钩：
        cutoff = min_cutoff + beta * |dx_hat|
    人几乎不动时 cutoff 低 → 狠狠压抖动；动起来时 cutoff 抬高 →
    当帧就跟上。它只有一个状态（上一输出 + 上一导数），没有窗口、
    没有未来数据，适合逐帧实时链路。

本模块只做"标量序列 → 标量序列"的滤波，**不知道角度、不知道关节**：
    * 输入 None 不是它的职责 —— 丢帧由调用方（mapper）直接输出 None，
      并且**清掉该通道的滤波状态**（不跨丢帧维持旧值，与本仓库一贯纪律一致）。
    * 单位无关：喂度就是度、喂 tick 就是 tick，但**同一实例不要换单位**。

接口约定（保持稳定，别改签名）：
    OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0, default_dt=1/30)
    update(x, timestamp=None) -> float
        喂一个样本，返回滤波值。timestamp 建议传秒（time.monotonic()）；
        不传就按 default_dt / 上一次的 dt 推进（离线回放用）。
    reset() -> None            清空状态，下一次 update 原样返回输入

参数取值的直觉：
    min_cutoff 越小越"黏"（静止时更稳）；beta 越大越"跟手"（快速动作时延迟小）。
    30fps 人体关键点场景的常用起点是 min_cutoff≈1.0 / beta≈0.05 / d_cutoff≈1.0，
    这三个数**未经实拍标定**，改之前先看动作跟随延迟。

纯标准库（math），可独立测试：python one_euro.py
"""

from __future__ import annotations

import math

# 数值下限：dt 太小会让导数爆炸（同一毫秒内连喂两帧时会发生）
_MIN_DT = 1e-3
_EPS = 1e-9


def _alpha(cutoff_hz: float, dt: float) -> float:
    """一阶低通的平滑系数 α = 1 / (1 + τ/dt)，τ = 1/(2π·fc)。"""
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, _EPS))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """单通道 One Euro 滤波器。

    === 两段低通 ===
    1. 先对**导数**做一次低通（d_cutoff）得到 dx_hat；
    2. 用 cutoff = min_cutoff + beta·|dx_hat| 得到自适应截止频率，
       再对**信号本身**做一次低通。

    导数用的是"原始样本之差"，不是"滤波值之差"：否则滤波把导数也拖慢，
    自适应就失去意义（快速动作的第一帧恰恰要靠原始跳变把 cutoff 抬起来）。

    === 时间戳 ===
    timestamp 用来算真实 dt。**不要传 None 又指望它自动测速**：
    None 时按 default_dt 推进（离线回放/自测够用），实时链路请传 time.monotonic()。
    dt 非正（时间戳回退 / 同一时刻喂两帧）时沿用上一次的 dt，不重置状态。
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0, default_dt: float = 1.0 / 30.0) -> None:
        """
        参数:
            min_cutoff: 静止时的截止频率（Hz）。越小越稳、跟随越慢。
            beta:       速度对截止频率的增益。越大越跟手、抖动越多。
            d_cutoff:   导数低通的截止频率（Hz）。
            default_dt: 不传 timestamp 时使用的步长（秒）。
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.default_dt = max(float(default_dt), _MIN_DT)
        self.reset()

    def reset(self) -> None:
        """清空跨帧状态（丢帧 / 换人 / 重新标定后调用）。"""
        self._x = None          # 上一滤波输出
        self._dx = 0.0          # 上一滤波导数
        self._t = None          # 上一 timestamp（可能为 None）
        self._last_dt = self.default_dt

    def update(self, x: float, timestamp: float = None) -> float:
        """喂一个样本，返回滤波值（第一次调用原样返回 x）。"""
        x = float(x)
        if self._x is None:
            self._x = x
            self._dx = 0.0
            self._t = timestamp
            self._last_dt = self.default_dt
            return x

        dt = self._step_dt(timestamp)
        # 1) 导数低通
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        # 2) 自适应截止频率 + 信号低通
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = timestamp
        return self._x

    # ---------- 内部实现 ----------

    def _step_dt(self, timestamp: float) -> float:
        """本次步长（秒）。时间戳缺失/回退时沿用上一次的 dt。"""
        if timestamp is None or self._t is None:
            dt = self._last_dt
        else:
            dt = float(timestamp) - float(self._t)
        if not (dt > _MIN_DT):     # 含 NaN / 0 / 负值
            dt = self._last_dt
        self._last_dt = dt
        return dt


# ---------------------------------------------------------------------------
# 自测：纯合成序列，不碰摄像头/机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest_one_euro() -> None:
    """常量 / 阶跃 / 噪声 / beta 跟随 / reset 五组用例。"""
    fps = 30.0
    dt = 1.0 / fps

    # 1) 常量输入：输出必须原样稳定（没有过冲、没有漂移）
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    t = 0.0
    vals = []
    for _ in range(90):
        t += dt
        vals.append(f.update(10.0, t))
    _assert(abs(vals[-1] - 10.0) < 1e-6,
            f"constant input should stay 10.0, got {vals[-1]}")
    _assert(max(vals) - min(vals) < 1e-6, "constant input should not wobble")

    # 2) 阶跃输入：单调上升、有延迟、最终收敛
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    f.update(0.0, 0.0)
    step_out = []
    for i in range(1, 91):
        step_out.append(f.update(10.0, i * dt))
    for a, b in zip(step_out, step_out[1:]):
        _assert(b >= a - 1e-9,
                f"step response should be monotonic, got {a} -> {b}")
    _assert(1.0 < step_out[0] < 9.5,
            f"step response should lag at the first frame, got {step_out[0]}")
    _assert(step_out[-1] > 9.9,
            f"step response should converge, got {step_out[-1]}")

    # 3) 高频噪声：输出峰峰值必须明显小于输入
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    f.update(5.0, 0.0)
    noisy = []
    for i in range(1, 121):
        noisy.append(f.update(5.0 + (0.2 if i % 2 else -0.2), i * dt))
    p2p = max(noisy[10:]) - min(noisy[10:])
    _assert(p2p < 0.20,
            f"noise should be attenuated (input p2p 0.4), output p2p={p2p:.3f}")

    # 4) beta 越大跟随越快（同样的阶跃，高 beta 在早期走得更远）
    f0 = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    f1 = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
    f0.update(0.0, 0.0)
    f1.update(0.0, 0.0)
    o0 = o1 = 0.0
    for i in range(1, 6):
        o0 = f0.update(10.0, i * dt)
        o1 = f1.update(10.0, i * dt)
    _assert(o1 > o0 + 1e-6,
            f"higher beta should follow faster ({o1} vs {o0})")

    # 5) reset：下一次 update 原样返回输入
    f = OneEuroFilter()
    f.update(1.0, 0.0)
    f.update(50.0, dt)
    f.reset()
    _assert(f.update(7.5, 2.0) == 7.5, "reset should return raw value")

    print("  selftest_one_euro: OK")


if __name__ == "__main__":
    selftest_one_euro()
    print("one_euro selftest: ALL PASSED")
