"""Classical lane detection: colour mask + Canny edges -> region of interest -> Hough lines -> left/right lane fit."""
import cv2
import numpy as np


def colour_mask(bgr):
    """Keep yellow and white paint (the two lane-marking colours) in HSV space."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 190), (180, 40, 255))
    return cv2.bitwise_or(yellow, white)


def roi_polygon(shape, top=0.60, left=0.08, right=0.95, top_left=0.43, top_right=0.57):
    h, w = shape[:2]
    return np.array([[(int(left * w), h), (int(top_left * w), int(top * h)),
                      (int(top_right * w), int(top * h)), (int(right * w), h)]], np.int32)


def detect_edges(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 150)
    mask = colour_mask(bgr)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.bitwise_and(edges, mask)


def apply_roi(img, polygon):
    m = np.zeros_like(img)
    cv2.fillPoly(m, polygon, 255)
    return cv2.bitwise_and(img, m)


def hough_segments(edges):
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=60)
    return [] if segs is None else np.asarray(segs).reshape(-1, 4)


def split_left_right(segs, min_abs_slope=0.4, max_abs_slope=3.0):
    left, right = [], []
    for x1, y1, x2, y2 in segs:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not (min_abs_slope < abs(slope) < max_abs_slope):
            continue  # reject near-horizontal clutter and near-vertical noise
        (left if slope < 0 else right).append((x1, y1, x2, y2, slope))
    return left, right


def fit_lane(group, y_bottom, y_top):
    """Length-weighted average of slope/intercept, returned as endpoints at y_bottom and y_top."""
    if not group:
        return None
    g = np.array(group)
    lengths = np.hypot(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1])
    slope = np.average(g[:, 4], weights=lengths)
    mid_x = np.average((g[:, 0] + g[:, 2]) / 2, weights=lengths)
    mid_y = np.average((g[:, 1] + g[:, 3]) / 2, weights=lengths)
    x_at = lambda y: int(mid_x + (y - mid_y) / slope)
    return (x_at(y_bottom), y_bottom, x_at(y_top), y_top)


def process(bgr):
    """Return an annotated frame, intermediate stages and the lateral offset (fraction of lane half-width)."""
    h, w = bgr.shape[:2]
    poly = roi_polygon(bgr.shape)
    edges = apply_roi(detect_edges(bgr), poly)
    segs = hough_segments(edges)
    left, right = split_left_right(segs)
    y_top = int(0.62 * h)
    L, R = fit_lane(left, h, y_top), fit_lane(right, h, y_top)
    out = bgr.copy()
    if L and R:
        cv2.fillPoly(out, [np.array([(L[0], L[1]), (L[2], L[3]), (R[2], R[3]), (R[0], R[1])])], (0, 255, 0))
        out = cv2.addWeighted(bgr, 0.65, out, 0.35, 0)
    for lane in (L, R):
        if lane:
            cv2.line(out, lane[:2], lane[2:], (0, 0, 255), 6)
    offset = None
    if L and R:
        centre = (L[0] + R[0]) / 2
        half = (R[0] - L[0]) / 2
        offset = float((w / 2 - centre) / half)  # >0: vehicle right of lane centre
        cv2.putText(out, f"offset {offset:+.2f} x half-lane", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return {"annotated": out, "edges": edges, "segments": segs, "left": L, "right": R, "offset": offset, "roi": poly}
