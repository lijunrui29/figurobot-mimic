"""robot_link 层：通过 figurobot 桥接的 HTTP API 与机器人通信（阶段 4）。

这一层**只做传输，不做任何安全判断**：
    * 输入必须是 `safety.SafetyGovernor.update()` 已经允许下发的 tick；
    * 白名单 / 限位 / 限速 / 看门狗 / 熔断全部在 safety 层；
    * 本层不读配置、不算角度、不决定"该不该发"。

对应的桥接服务是 `figurobot-console/robot_bridge.py`（默认 127.0.0.1:8888）：
    GET  /api/status         连接状态
    POST /api/pose           读全部关节姿态（含 safety / zero_values）
    POST /api/joint/write    写单个关节目标 tick

**为什么是一颗一颗写**：桥接目前只暴露 `/api/joint/write`，每次写入在控制盒
上都要跑一次 adb + 串口脚本，耗时明显。所以 `send_targets()` 默认按频率
由入口限流（`--live-hz`），不要按摄像头帧率调用。批量 / SYNC_WRITE 端点
要改桥接（另一个仓库），不在本层偷偷做。

接口约定（保持稳定，别改签名）：
    RobotLink(base_url=..., timeout=...)
        status() -> dict                       GET /api/status
        read_pose(fast=True) -> dict           POST /api/pose
        read_positions(fast=True) -> {sid: raw tick}
        write_joint(servo_id, tick) -> dict    POST /api/joint/write
        send_targets(targets, stop_on_error=True) -> LinkResult
    LinkResult
        ok / sent / failed / error / elapsed_s

传输失败（桥接没起、超时、JSON 不对）在 status/read_pose 里抛
`RobotLinkError`；`send_targets()` 把失败收进 LinkResult，**不抛**，方便入口
按帧调用 `safety.notify_result(ok)` 走熔断。

纯标准库（urllib/json），可独立测试：python robot_link.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8888"
DEFAULT_TIMEOUT = 3.0


class RobotLinkError(RuntimeError):
    """桥接不可达 / 返回不合法。消息里带地址和可执行的排查提示。"""


@dataclass
class LinkResult:
    """一次 `send_targets()` 的结果。

    ok:      所有关节都写成功（没有任何 failed）
    sent:    {servo_id: 实际写入的 tick}
    failed:  {servo_id: 错误短码/消息}
    error:   整体错误（例如桥接不可达）；没有则为 None
    elapsed_s: 这次下发在墙钟上花了多久（串口写入慢，入口要据此限流）
    """
    ok: bool = True
    sent: Dict[int, int] = field(default_factory=dict)
    failed: Dict[int, str] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "sent": dict(self.sent),
            "failed": dict(self.failed),
            "error": self.error,
            "elapsed_s": self.elapsed_s,
        }


class RobotLink:
    """figurobot 桥接的薄客户端。一个桥接地址一个实例（无跨帧状态）。"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)

    # ---------- 底层 HTTP ----------

    def _request(self, method: str, path: str,
                 payload: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # 桥接在业务错误时也会回 JSON（例如 {"error": ...}），尽量把它带出来
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            raise RobotLinkError(
                f"{url} 返回 HTTP {exc.code}"
                + (f"：{detail}" if detail else "")) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RobotLinkError(
                f"连不上机器人桥接 {self.base_url}（{exc}）。"
                "先在 figurobot-console 目录启动：python robot_bridge.py --port 8888；"
                "若桥接已在跑，可能是完整回读较慢（实测约 5~6s），"
                "把 --bridge-timeout 调大再试"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RobotLinkError(f"{url} 返回的不是 JSON：{body[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise RobotLinkError(f"{url} 返回的 JSON 不是对象：{type(parsed).__name__}")
        return parsed

    # ---------- 只读 ----------

    def status(self) -> dict:
        """GET /api/status → {'connected': bool, 'serial': ..., ...}。"""
        return self._request("GET", "/api/status")

    def read_pose(self, fast: bool = True) -> dict:
        """POST /api/pose → 桥接原始返回（joints / safety / zero_values / limits）。

        fast=True 走 PING 缓存路径，快但错误码可能来自缓存；标定读中立位时
        建议用 fast=False 拿一次完整读数。
        """
        resp = self._request("POST", "/api/pose", {"fast": bool(fast)})
        if not resp.get("ok", False) and not resp.get("joints"):
            raise RobotLinkError(f"读姿态失败：{resp.get('error') or resp}")
        return resp

    def read_positions(self, fast: bool = True) -> Dict[int, int]:
        """只取在线关节的当前 tick：{servo_id: int(raw)}。

        离线 / 没回读到的关节**不出现**在结果里（调用方自己决定怎么处理）。
        """
        pose = self.read_pose(fast=fast)
        out: Dict[int, int] = {}
        for key, joint in (pose.get("joints") or {}).items():
            if not isinstance(joint, dict) or not joint.get("online"):
                continue
            raw = joint.get("raw")
            if raw is None:
                continue
            try:
                out[int(key)] = int(raw)
            except (TypeError, ValueError):
                continue
        return out

    # ---------- 写入 ----------

    def write_joint(self, servo_id: int, tick: int) -> dict:
        """写单个关节目标位置。返回桥接 JSON；失败由调用方看 `ok`。"""
        return self._request("POST", "/api/joint/write",
                             {"id": int(servo_id), "position": int(tick)})

    def write_joint_batch(self, targets: Dict[int, int],
                          ramp_ms: int = 0, steps: int = 1) -> dict:
        """一次请求写多个关节（桥接的 /api/joint/write_batch）。

        桥接会在设备上只启动一次 adb + python，再逐个发串口帧；比
        `send_targets` 的逐关节调用快一个数量级（实时模仿必需）。
        ramp_ms/steps > 0 时桥接会把目标铺成连续轨迹（设备端按时间发
        中间帧），解决"一秒跳一次"的卡顿；老桥接忽略这两个字段。
        """
        payload = {"targets": {str(int(sid)): int(tick)
                               for sid, tick in (targets or {}).items()},
                   "ramp_ms": int(ramp_ms or 0),
                   "steps": int(steps or 1)}
        return self._request("POST", "/api/joint/write_batch", payload)

    def send_targets(self, targets: Dict[int, Optional[float]],
                     stop_on_error: bool = True,
                     ramp_ms: int = 0, steps: int = 1) -> LinkResult:
        """把 safety 允许的目标写给机器人。

        优先走**批量端点**（一次 adb 调用写多个关节）；老桥接没有这个
        端点（HTTP 404）时自动退回逐关节写，保证兼容。ramp_ms/steps 走
        批量端点时生效；逐关节兜底不支持插值（老桥接本来也没有该能力）。

        targets 里值为 None 的条目跳过（= 本帧保持）。**调用方必须先过
        safety**；本层不再过滤白名单。

        stop_on_error=True：失败就停，避免桥接掉线时每个关节都等超时。
        失败不会抛异常，而是写进 LinkResult → 入口调 safety.notify_result(False)。
        """
        t0 = time.monotonic()
        result = LinkResult()
        pending = [(int(sid), int(tick)) for sid, tick in (targets or {}).items()
                   if tick is not None]
        # 确定性顺序，便于日志对齐
        pending.sort()
        if not pending:
            result.elapsed_s = time.monotonic() - t0
            return result

        # ---- 1) 批量端点 ----
        try:
            resp = self.write_joint_batch(dict(pending), ramp_ms=ramp_ms,
                                          steps=steps)
        except RobotLinkError as exc:
            if "HTTP 404" not in str(exc):
                result.failed = {sid: str(exc) for sid, _ in pending}
                result.error = str(exc)
                result.ok = False
                result.elapsed_s = time.monotonic() - t0
                return result
            # 404 → 老桥接，走逐关节兜底
        else:
            if resp.get("ok"):
                result.sent = dict(pending)
                result.ok = True
                result.elapsed_s = time.monotonic() - t0
                return result
            msg = str(resp.get("error") or "batch write failed")
            result.failed = {sid: msg for sid, _ in pending}
            result.error = msg
            result.ok = False
            result.elapsed_s = time.monotonic() - t0
            return result

        # ---- 2) 逐关节兜底（老桥接 / 批量端点不可用） ----
        for sid, tick in pending:
            try:
                resp = self.write_joint(sid, tick)
            except RobotLinkError as exc:
                result.failed[sid] = str(exc)
                result.error = str(exc)
                if stop_on_error:
                    break
                continue
            if resp.get("ok"):
                result.sent[sid] = tick
            else:
                msg = str(resp.get("error") or "write failed")
                result.failed[sid] = msg
                result.error = msg
                if stop_on_error:
                    break
        result.ok = not result.failed
        result.elapsed_s = time.monotonic() - t0
        return result


# ---------------------------------------------------------------------------
# 自测：本地假桥接（HTTP），不碰真机器人
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def selftest_robot_link() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    calls = {"write": [], "batch": [], "ramp": [], "pose": 0, "status": 0}
    fail_ids = {6}
    batch_enabled = {"on": True}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/status":
                calls["status"] += 1
                self._json({"connected": True, "serial": "FAKE",
                            "protocol": "test", "baud": 1000000})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length).decode() or "{}")
            if self.path == "/api/pose":
                calls["pose"] += 1
                self._json({
                    "ok": True,
                    "joints": {
                        "4": {"raw": 2048, "angle": 180.0, "online": True},
                        "6": {"raw": None, "angle": None, "online": False},
                        "9": {"raw": 2100, "angle": 184.5, "online": True},
                    },
                    "safety": {"level": "safe"},
                    "zero_values": {"4": 2048},
                })
            elif self.path == "/api/joint/write":
                sid = int(body.get("id", 0))
                calls["write"].append((sid, int(body.get("position", 0))))
                if sid in fail_ids:
                    self._json({"ok": False, "id": sid, "error": "fake failure"})
                else:
                    self._json({"ok": True, "id": sid,
                                "position": int(body.get("position", 0))})
            elif self.path == "/api/joint/write_batch":
                if not batch_enabled["on"]:
                    self._json({"error": "not found"}, 404)
                    return
                targets = {int(k): int(v)
                           for k, v in (body.get("targets") or {}).items()}
                calls["batch"].append(dict(targets))
                calls["ramp"].append((body.get("ramp_ms", 0),
                                      body.get("steps", 1)))
                if any(sid in fail_ids for sid in targets):
                    self._json({"ok": False, "sent": {},
                                "error": "fake batch failure"})
                else:
                    self._json({"ok": True, "sent": targets})
            else:
                self._json({"error": "not found"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    link = RobotLink(f"http://127.0.0.1:{port}", timeout=2.0)
    try:
        # 1) status / pose / 在线关节过滤
        st = link.status()
        _assert(st["connected"] is True and st["serial"] == "FAKE", f"status={st}")
        pos = link.read_positions(fast=True)
        _assert(pos == {4: 2048, 9: 2100}, f"positions={pos}")
        _assert(calls["pose"] == 1, f"pose calls={calls['pose']}")

        # 2) 批量端点：一次请求写两颗，不走逐关节
        res = link.send_targets({4: 2050, 9: 2101, 6: None})
        _assert(res.ok, f"send should succeed: {res}")
        _assert(res.sent == {4: 2050, 9: 2101}, f"sent={res.sent}")
        _assert(calls["batch"] == [{4: 2050, 9: 2101}],
                f"batches={calls['batch']}")
        _assert(calls["write"] == [], f"writes={calls['write']}")

        # 2b) ramp 参数透传到批量端点
        calls["batch"].clear()
        calls["ramp"].clear()
        res = link.send_targets({4: 2050, 9: 2101}, ramp_ms=800, steps=8)
        _assert(res.ok and calls["ramp"] == [(800, 8)],
                f"ramp={calls['ramp']}")

        # 3) 批量端点整体失败 → 所有 pending 都记 failed
        calls["batch"].clear()
        res = link.send_targets({4: 2050, 6: 2100})
        _assert(not res.ok and set(res.failed) == {4, 6},
                f"batch failure: {res}")

        # 4) 老桥接（无批量端点）→ 404 自动退回逐关节
        batch_enabled["on"] = False
        calls["write"].clear()
        res = link.send_targets({4: 2050, 6: 2100, 9: 2101})
        _assert(not res.ok and 6 in res.failed, f"expected failure: {res}")
        _assert(res.sent == {4: 2050}, f"sent={res.sent}")
        _assert(calls["write"] == [(4, 2050), (6, 2100)], f"writes={calls['write']}")

        # 5) stop_on_error=False：失败后继续写 9
        calls["write"].clear()
        res = link.send_targets({4: 2050, 6: 2100, 9: 2101}, stop_on_error=False)
        _assert(res.sent == {4: 2050, 9: 2101} and 6 in res.failed,
                f"continue-on-error: {res}")
        _assert(calls["write"] == [(4, 2050), (6, 2100), (9, 2101)],
                f"writes={calls['write']}")

        # 6) 桥接不可达 → status 抛 RobotLinkError（用已关闭端口的地址）
        dead = RobotLink(f"http://127.0.0.1:{port}", timeout=0.5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
        try:
            dead.status()
        except RobotLinkError:
            pass
        else:
            raise AssertionError("unreachable bridge should raise RobotLinkError")

        # 7) send_targets 不抛异常，把错误收进 LinkResult
        res = dead.send_targets({4: 2050})
        _assert(not res.ok and res.error and 4 in res.failed,
                f"dead bridge send: {res}")
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass

    print("  selftest_robot_link: OK")


if __name__ == "__main__":
    selftest_robot_link()
    print("robot_link selftest: ALL PASSED")
