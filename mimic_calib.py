"""mimic_calib 层：人体中立位采集（多帧中位数，写回配置的 human_neutral）。

为什么不用"按一下记一帧"：
    单帧关键点抖动会原样变成整条映射的中立位偏移（`delta = 当前角 − 中立角`），
    之后所有读数都跟着偏。用一段时间的**中位数**（不是均值）抗瞬时丢点/抖动。

采集的姿势（写进用法提示）：正对镜头、双臂自然下垂、肘微屈、双脚与肩同宽。
**机器人侧的中立位不在这里采集** —— 那是 `robot_neutral_tick`，要阶段 3
Motion_Reset 后从 `/api/pose` 回读。

接口约定：
    NeutralCollector(duration_s=5.0, min_samples=30)
        start(now)                     开始计时
        add(details, now)              每帧喂 mapper 的 details
        done(now) -> bool              采集窗口是否结束
        progress(now) -> float         0-1
        counts() -> dict               每关节样本数（给画面显示）
        medians() -> {servo_id: 中位数}
        report() -> str                收集结果的可读报告
        apply(cfg) -> (new_cfg, applied, skipped)
                                       返回新配置（深拷贝），不改入参

纯标准库，可独立测试：python mimic_calib.py
"""

from __future__ import annotations

import copy
import statistics
from typing import Dict, Optional

DEFAULT_DURATION_S = 5.0
DEFAULT_MIN_SAMPLES = 30


class NeutralCollector:
    """一段时间的采样 + 中位数。一个采集会话一个实例。"""

    def __init__(self, duration_s: float = DEFAULT_DURATION_S,
                 min_samples: int = DEFAULT_MIN_SAMPLES) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s 必须大于 0")
        if min_samples < 1:
            raise ValueError("min_samples 必须 >= 1")
        self.duration_s = float(duration_s)
        self.min_samples = int(min_samples)
        self._t0: Optional[float] = None
        self._samples: Dict[int, list] = {}
        self._missing: Dict[int, int] = {}

    # ---------- 生命周期 ----------

    def start(self, now: float) -> None:
        self._t0 = float(now)
        self._samples = {}
        self._missing = {}

    @property
    def active(self) -> bool:
        return self._t0 is not None

    def progress(self, now: float) -> float:
        if not self.active:
            return 0.0
        return max(0.0, min(1.0, (float(now) - self._t0) / self.duration_s))

    def done(self, now: float) -> bool:
        return self.active and (float(now) - self._t0) >= self.duration_s

    # ---------- 采样 ----------

    def add(self, details: dict, now: float) -> None:
        """喂一帧 mapper details。窗口结束后不再收集。"""
        if not self.active or self.done(now):
            return
        for sid, detail in (details or {}).items():
            if not detail.get("enabled"):
                continue
            sid = int(sid)
            value = detail.get("human_angle")
            if value is None:
                self._missing[sid] = self._missing.get(sid, 0) + 1
                continue
            self._samples.setdefault(sid, []).append(float(value))

    def counts(self) -> Dict[int, dict]:
        """每个采到样本的关节：{'n', 'missing', 'median'}（未采到则不出现）。"""
        out = {}
        for sid, values in self._samples.items():
            out[sid] = {
                "n": len(values),
                "missing": self._missing.get(sid, 0),
                "median": statistics.median(values),
            }
        for sid, missing in self._missing.items():
            out.setdefault(sid, {"n": 0, "missing": missing, "median": None})
        return out

    def medians(self) -> Dict[int, float]:
        """样本数达标的关节 → 中位数（度）。"""
        return {sid: statistics.median(values)
                for sid, values in self._samples.items()
                if len(values) >= self.min_samples}

    # ---------- 应用 ----------

    def report(self, names: Optional[Dict[int, str]] = None) -> str:
        names = names or {}
        lines = []
        for sid, info in sorted(self.counts().items()):
            name = names.get(sid, str(sid))
            if info["median"] is None:
                lines.append(f"  {name:<8} samples=0 missing={info['missing']}  SKIP")
            elif info["n"] < self.min_samples:
                lines.append(f"  {name:<8} samples={info['n']} "
                             f"(< {self.min_samples})  SKIP")
            else:
                lines.append(f"  {name:<8} samples={info['n']:>3} "
                             f"missing={info['missing']:>3}  "
                             f"median={info['median']:+.2f}deg")
        return "\n".join(lines) if lines else "  (no samples)"

    def apply(self, cfg: dict):
        """把中位数写进配置副本的 human_neutral。

        返回 (new_cfg, applied, skipped)：
            applied: {servo_id: 中位数}
            skipped: {servo_id: 原因}（样本不够 / 该关节没采到）
        """
        new_cfg = copy.deepcopy(cfg)
        medians = self.medians()
        applied: Dict[int, float] = {}
        skipped: Dict[int, str] = {}
        for joint in new_cfg.get("joints") or []:
            sid = int(joint.get("servo_id", -1))
            if not joint.get("enabled"):
                continue
            if sid not in medians:
                info = self.counts().get(sid)
                n = 0 if not info else info["n"]
                skipped[sid] = f"samples={n} < {self.min_samples}"
                continue
            value = round(float(medians[sid]), 2)
            joint["human_neutral"] = value
            applied[sid] = value
        return new_cfg, applied, skipped


# ---------------------------------------------------------------------------
# 自测：合成 details，不碰摄像头/机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _detail(angle, enabled=True):
    return {"enabled": enabled, "human_angle": angle}


def selftest_mimic_calib() -> None:
    # 1) 时间窗
    col = NeutralCollector(duration_s=5.0, min_samples=3)
    col.start(100.0)
    _assert(not col.done(104.9), "should not finish before 5s")
    _assert(col.done(105.0), "should finish at 5s")
    _assert(abs(col.progress(102.5) - 0.5) < 1e-9, "progress")

    # 2) 采样 + 中位数（中位数要抗离群点）
    col = NeutralCollector(duration_s=5.0, min_samples=3)
    col.start(0.0)
    for i, a in enumerate([10.0, 12.0, 11.0, 1000.0]):
        col.add({4: _detail(a)}, now=0.1 + i)
    _assert(col.medians()[4] == 11.5, f"median={col.medians()[4]}")
    _assert(col.counts()[4]["n"] == 4, "sample count")

    # 3) 丢帧只记 missing，不进中位数
    col = NeutralCollector(min_samples=1)
    col.start(0.0)
    col.add({4: _detail(5.0), 6: _detail(None)}, now=0.1)
    _assert(col.medians()[4] == 5.0, "valid sample kept")
    _assert(6 not in col.medians(), "missing joint must not get a median")
    _assert(col.counts()[6]["missing"] == 1, "missing count")

    # 4) 禁用关节不采
    col = NeutralCollector(min_samples=1)
    col.start(0.0)
    col.add({7: _detail(3.0, enabled=False)}, now=0.1)
    _assert(not col.medians(), "disabled joints must be ignored")

    # 5) apply：只改启用的、样本够的
    cfg = {
        "joints": [
            {"servo_id": 4, "enabled": True, "human_neutral": 0.0},
            {"servo_id": 6, "enabled": True, "human_neutral": 0.0},
            {"servo_id": 7, "enabled": False, "human_neutral": 0.0},
        ]
    }
    col = NeutralCollector(min_samples=2)
    col.start(0.0)
    for _ in range(3):
        col.add({4: _detail(20.0), 7: _detail(99.0)}, now=0.1)
    new_cfg, applied, skipped = col.apply(cfg)
    _assert(applied == {4: 20.0}, f"applied={applied}")
    _assert(6 in skipped, f"skipped={skipped}")
    _assert(new_cfg["joints"][0]["human_neutral"] == 20.0, "neutral written")
    _assert(new_cfg["joints"][1]["human_neutral"] == 0.0, "untouched joint")
    _assert(new_cfg["joints"][2]["human_neutral"] == 0.0, "disabled untouched")
    _assert(cfg["joints"][0]["human_neutral"] == 0.0, "input must not be mutated")

    # 6) report 不出异常、能区分 SKIP（6 号只有丢帧记录）
    col.add({6: _detail(None)}, now=0.3)
    text = col.report(names={4: "L-shR", 6: "L-elb"})
    _assert("L-shR" in text and "SKIP" in text, text)

    print("  selftest_mimic_calib: OK")


if __name__ == "__main__":
    selftest_mimic_calib()
    print("mimic_calib selftest: ALL PASSED")
