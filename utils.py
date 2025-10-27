import math
import cv2
import numpy as np

def draw_interpolated_line(canvas, prev, cur, color, thickness=5, step_px = 5, max_step=50):
    prev_x, prev_y = prev
    x, y= cur

    dx= x-prev_x
    dy = y - prev_y

    dist = math.hypot(dx, dy)

    if dist<1:
        return (x, y)
    
    # ステップ数
    steps = min(max_step, max(1, int(dist/step_px)))

    pts =[] # cac diem noi suy
    for i in range(1, steps + 1):
        t = i/ steps
        idx = (prev_x + dx*t)
        idy = (prev_y + dy*t)
        pts.append((idx, idy))

    all_pts = [(int(prev_x), int(prev_y))] + pts
    # False: 最後のポイントを最初のポイントに接続する 
    # line_AA: muot
    cv2.polylines(canvas, [np.array(all_pts, dtype = np.int32)], False, color,thickness, lineType=cv2.LINE_AA) 

    # 今のポイントは前のポイントになる
    return (x, y)