"""mimic 专用的骨架绘制（从动作线 output.py 精简出来的骨架部分）。

动作线的 `output.py` 还带有 11 个动作的读数/状态/几何叠加，依赖
`decision.py`；mimic 只用 `FrameRenderer.draw_skeleton_frame`，所以这里
只保留骨架绘制，避免把整套动作线代码带进本仓库。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# 骨架：17 个 COCO 关键点之间怎么连线（RTMPose 输出的 COCO 17 点）。
# 只保留躯干 + 四肢连线；头部/面部 0-4 不画（脸只用于角度计算）。
POSE_CONNECTIONS = frozenset([
    (5, 6), (5, 7), (7, 9),         # 肩 + 左臂
    (6, 8), (8, 10),                # 右臂
    (5, 11), (6, 12), (11, 12),     # 躯干
    (11, 13), (13, 15),             # 左腿
    (12, 14), (14, 16),             # 右腿
])


class FrameRenderer:
    """画面输出器（只画骨架）。"""

    def __init__(self, draw_skeleton: bool = True) -> None:
        self.draw_skeleton = draw_skeleton

    def draw_skeleton_frame(self, frame: np.ndarray,
                            pose: Optional[dict]) -> None:
        """只叠骨架（关键点 + 连线 + pid/置信度标签）。"""
        if self.draw_skeleton and pose is not None:
            self._draw_skeleton(frame, pose)

    def _draw_skeleton(self, frame: np.ndarray, pose: dict) -> None:
        """把姿态骨架（关键点 + 连接线 + pid/置信度标签）叠到画面上。

        pose: {'landmarks': [(x,y,z,vis) x17], 'image_size': (w,h)}
        """
        landmarks = pose.get("landmarks")
        image_size = pose.get("image_size")
        if not landmarks or image_size is None:
            return

        w, h = image_size
        VIS_LOW = 0.3    # < 此值 = 没检测到（灰）
        VIS_OK = 0.5     # >= 此值 = 置信度足够（绿）

        def px(pid: int) -> Optional[tuple]:
            if pid >= len(landmarks):
                return None
            x, y, _z, _vis = landmarks[pid]
            return int(x * w), int(y * h)

        def vis_of(pid: int) -> float:
            return landmarks[pid][3] if pid < len(landmarks) else 0.0

        # 关键点：画 3-16 号；躯干/四肢绿，耳 3/4 黄，没检测到画灰空心。
        for pid in range(3, 17):
            p = px(pid)
            if p is None:
                continue
            if vis_of(pid) < VIS_LOW:
                cv2.circle(frame, p, 3, (150, 150, 150), 1, cv2.LINE_AA)
            elif pid <= 4:
                cv2.circle(frame, p, 4, (0, 215, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, 4, (0, 255, 0), -1, cv2.LINE_AA)

        # 标签：`pid:置信度`，字色按置信度分档（黑描边保证可读）。
        for pid in range(3, 17):
            p = px(pid)
            if p is None:
                continue
            vis = vis_of(pid)
            if vis >= VIS_OK:
                color = (60, 200, 60)
            elif vis >= VIS_LOW:
                color = (0, 215, 255)
            else:
                color = (150, 150, 150)
            label = f"{pid}:{vis:.2f}"
            tx = min(p[0] + 6, w - 42)
            ty = p[1] + 12
            cv2.putText(frame, label, (tx, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                        cv2.LINE_AA)

        # 连接线（两端都可见才画）
        for a, b in POSE_CONNECTIONS:
            if vis_of(a) < VIS_LOW or vis_of(b) < VIS_LOW:
                continue
            pa, pb = px(a), px(b)
            if pa is not None and pb is not None:
                cv2.line(frame, pa, pb, (255, 0, 0), 2, cv2.LINE_AA)
