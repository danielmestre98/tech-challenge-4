def iou(a, b):
    # a,b: (x,y,w,h)
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def clamp_box(box, W, H):
    x, y, w, h = box
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return (x, y, w, h)


def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    ss = seconds - 60 * m
    return f"{m:02d}:{ss:05.2f}"
