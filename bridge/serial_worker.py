"""figurobot 常驻串口 worker（在设备上由 robot_bridge.py 启动）。

和一次性的 `serial_exec.py` 不同：本进程**打开串口后一直不退出**，从 stdin
逐行读取 JSON 请求，把帧写到舵机总线，然后回一行 JSON ack。这样主机不再为
每次写都启动一次 `adb shell + python3`，实时模仿的延迟从 ~0.8s 降到毫秒级。

请求（一行一个 JSON）：
    {"frames": ["ff00fd...", ...], "delay_ms": 0}
    frames 已经是完整的串口帧（hex），delay_ms 是相邻帧之间的延时。

响应（一行一个 JSON）：
    启动：{"ok": true, "port": "/dev/serial/by-id/..."}
    成功：{"ok": true, "sent": N}
    失败：{"ok": false, "error": "..."}

启动：python3 -u /tmp/figurobot_serial_worker.py [串口路径]
不传串口路径时用 find_port() 探测（PING id=1）。
"""

import glob
import json
import os
import sys
import termios
import time
import fcntl


def crc16(d):
    c = 0
    for b in d:
        c ^= (b << 8) & 0xFFFF
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return [c & 0xFF, (c >> 8) & 0xFF]


def ping(did):
    body = [0xFF, 0x00, 0xFD, 0x00, did, 3, 0, 1]
    return bytes(body + crc16(body))


def openp(port):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl & ~os.O_NONBLOCK)
    a = termios.tcgetattr(fd)
    a[2] = (a[2] & ~(termios.CSIZE | termios.PARENB | termios.CSTOPB | termios.PARODD)) \
           | termios.CS8 | termios.CREAD | termios.CLOCAL
    a[0] = 0
    a[1] = 0
    a[3] = 0
    a[6][termios.VMIN] = 0
    a[6][termios.VTIME] = 1
    a[4] = termios.B1000000
    a[5] = termios.B1000000
    termios.tcsetattr(fd, termios.TCSANOW, a)
    return fd


def drain(fd, max_iters=10):
    out = b''
    for _ in range(max_iters):
        try:
            c = os.read(fd, 512)
            if not c:
                break
            out += c
        except Exception:
            break
    return out


def find_port():
    cands = glob.glob('/dev/serial/by-id/usb-1a86*') + glob.glob('/dev/ttyACM*')
    seen = set()
    ports = []
    for p in cands:
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            ports.append(real)
    for p in ports:
        try:
            fd = openp(p)
            os.write(fd, ping(1))
            time.sleep(0.15)
            out = drain(fd)
            os.close(fd)
            if b'\xfd\x00' in out:
                return p
        except Exception:
            pass
    return None


def _reply(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SERIAL_PORT')
    if not port:
        port = find_port()
    if not port:
        _reply({"ok": False, "error": "no_motor_serial"})
        return
    try:
        fd = openp(port)
        # 写路径不需要等舵机应答；把读超时设成 0，drain 立即返回，
        # 避免每次请求多等 0.1s。
        a = termios.tcgetattr(fd)
        a[6][termios.VTIME] = 0
        a[6][termios.VMIN] = 0
        termios.tcsetattr(fd, termios.TCSANOW, a)
    except Exception as exc:
        _reply({"ok": False, "error": "open_failed: {}".format(exc)})
        return
    _reply({"ok": True, "port": port})
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                frames = req.get("frames") or []
                delay = float(req.get("delay_ms") or 0.0) / 1000.0
                for f in frames:
                    os.write(fd, bytes.fromhex(f))
                    if delay > 0:
                        time.sleep(delay)
                # 清掉舵机应答，避免缓冲区累积；写失败不影响 ack 的主流程
                try:
                    drain(fd, max_iters=10)
                except Exception:
                    pass
                _reply({"ok": True, "sent": len(frames)})
            except Exception as exc:
                _reply({"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)})
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


if __name__ == "__main__":
    main()
