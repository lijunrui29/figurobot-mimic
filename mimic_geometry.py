"""mimic 用到的几何门控常量（与动作线 features.py 同口径）。

动作线仓库里这些常量在 `features.py`；mimic 只需要这两个值，单独放一个
小模块，避免把整套动作特征类带进本仓库。改动作线的门控时要同步这里。
"""

FRAME_EDGE_MARGIN = 0.02      # 归一化坐标越出 ±2% 视为画面外（RTMPose 外推点）
MIN_SEGMENT_RATIO = 0.01      # 角度依赖线段最短 = 1% 画面对角线
