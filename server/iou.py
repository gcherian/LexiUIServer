# ---- IoU helpers ------------------------------------------------------------

def _rect_from_vertices(verts):
    """verts: list/tuple of 4 points [[x,y], ...]. Returns (x1,y1,x2,y2)."""
    xs = [p[0] for p in verts]
    ys = [p[1] for p in verts]
    return (min(xs), min(ys), max(xs), max(ys))

def iou_rect(a, b):
    """Axis-aligned IoU. a,b are (x1,y1,x2,y2) in the SAME coordinate system."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def iou_from_vertices(verts_a, verts_b):
    """IoU from 4-point quads; both abs or both normalized."""
    return iou_rect(_rect_from_vertices(verts_a), _rect_from_vertices(verts_b))

# Optional: generalized IoU (helps when boxes don't overlap)
def giou_rect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iou = iou_rect(a, b)
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    # union area:
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = area_a + area_b - inter
    return iou - ((c_area - union) / c_area if c_area > 0 else 0.0)