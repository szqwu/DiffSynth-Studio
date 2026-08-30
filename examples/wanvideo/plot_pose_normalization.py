#!/usr/bin/env python3
"""
Generate a figure illustrating query-centered pose normalization (c_q = I_4x4).
Coordinate system at origin, query camera at identity, input cameras at
various relative poses. No text labels. Style follows make_supplementary_video.py.
"""

import cv2
import numpy as np


BG = 255


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _make_c2w(R, t):
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def _frustum_pts(c2w, fov_x, fov_y, size):
    """5 pts [apex, BL, BR, TR, TL] in world coords. Camera looks along +X."""
    pos = c2w[:3, 3]
    fwd = c2w[:3, 0]
    up = c2w[:3, 1]
    right = c2w[:3, 2]
    hx = size * np.tan(fov_x / 2)
    hy = size * np.tan(fov_y / 2)
    ctr = pos + fwd * size
    return np.array([pos,
                     ctr - right * hx - up * hy,
                     ctr + right * hx - up * hy,
                     ctr + right * hx + up * hy,
                     ctr - right * hx + up * hy])


def _view_matrix(eye, target, up_hint=None):
    if up_hint is None:
        up_hint = np.array([0., 1., 0.])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_hint)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, fwd)
    return np.stack([right, up, fwd])


def _project(pts, view_rot, center, fov_deg=45, d=5.0):
    rotated = (view_rot @ (pts - center).T).T
    f = d / np.tan(np.radians(fov_deg) / 2.0)
    x2d = f * rotated[:, 0] / (rotated[:, 2] + d)
    y2d = f * rotated[:, 1] / (rotated[:, 2] + d)
    return x2d, y2d


def _draw_frustum(canvas, pts_px, color, thickness):
    """Draw wireframe frustum, matching reference script style exactly."""
    pts = pts_px.astype(np.int32)
    apex = tuple(pts[0])
    corners = [tuple(pts[i]) for i in range(1, 5)]
    for c in corners:
        cv2.line(canvas, apex, c, color, thickness, cv2.LINE_AA)
    for i in range(4):
        cv2.line(canvas, corners[i], corners[(i + 1) % 4],
                 color, thickness, cv2.LINE_AA)


def _draw_dashed_line(canvas, pt1, pt2, color, thickness=1,
                      dash_len=8, gap_len=6):
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dist = np.hypot(dx, dy)
    if dist < 1:
        return
    dx, dy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end = min(pos + dash_len, dist)
        x1, y1 = int(pt1[0] + dx * pos), int(pt1[1] + dy * pos)
        x2, y2 = int(pt1[0] + dx * end), int(pt1[1] + dy * end)
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        pos = end + gap_len


def render(fig_h=1200, fig_w=1400, padding=100):
    canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * BG

    fov_x, fov_y = np.radians(55), np.radians(38)
    frust_sz = 0.18
    axis_len = 0.65

    # Query camera: identity (c_q = I_4x4), looks along +X
    query = np.eye(4, dtype=np.float64)

    # Input cameras with diverse positions and orientations
    # yaw = rotation around Y, pitch = rotation around Z
    input_configs = [
        (np.array([0.9, 0.2, -0.4]), -0.50, 0.20),
        (np.array([-0.8, 0.1, 0.7]), -0.30, 0.10),
        (np.array([0.2, -0.15, -0.9]), -0.35, 0.12),
        (np.array([0.6, 0.2, -0.8]), -0.80, 0.15),
        (np.array([-0.2, 0.6, 0.3]), 0.45, 0.30),
    ]
    inputs = [_make_c2w(_rot_y(yaw) @ _rot_z(pitch), t)
              for t, yaw, pitch in input_configs]

    all_c2ws = [query] + inputs

    # Virtual camera: behind-right-above for clear 3/4 view
    eye = np.array([2.5, 3.5, 3.0])
    target = np.array([0.0, 0.0, -0.05])
    view_rot = _view_matrix(eye, target)

    # Gather all 3D points for projection scaling
    all_3d = np.concatenate(
        [_frustum_pts(c, fov_x, fov_y, frust_sz) for c in all_c2ws] +
        [np.array([[0, 0, 0],
                   [axis_len, 0, 0],
                   [0, axis_len, 0],
                   [0, 0, axis_len]])],
        axis=0)

    x2d, y2d = _project(all_3d, view_rot, target)
    pts2 = np.stack([x2d, y2d], axis=1)
    mn, mx = pts2.min(0), pts2.max(0)
    rng = mx - mn
    rng[rng == 0] = 1
    mn -= rng * 0.15
    mx += rng * 0.15
    rng = mx - mn
    scale = (min(fig_h, fig_w) - 2 * padding) / rng.max()
    c2d = (mn + mx) / 2
    cc = np.array([fig_w / 2., fig_h / 2.])

    def px(pts):
        x, y = _project(pts, view_rot, target)
        return np.stack([(x - c2d[0]) * scale + cc[0],
                         -(y - c2d[1]) * scale + cc[1]], axis=1)

    # 1. Dashed lines from each input camera apex to origin (background)
    o = px(np.array([[0, 0, 0]]))[0]
    for c2w in inputs:
        apex_3d = c2w[:3, 3]
        apex_px = px(np.array([apex_3d]))[0]
        _draw_dashed_line(canvas,
                          tuple(apex_px.astype(int)),
                          tuple(o.astype(int)),
                          (205, 205, 205), thickness=1,
                          dash_len=10, gap_len=8)

    # 2. Input camera frustums (dark blue)
    # in_col = (180, 60, 0)
    in_col = (203, 118, 44)
    for c2w in inputs:
        pts_px = px(_frustum_pts(c2w, fov_x, fov_y, frust_sz))
        _draw_frustum(canvas, pts_px, in_col, 3)
        apex = tuple(pts_px[0].astype(int))
        cv2.circle(canvas, apex, 3, in_col, -1, cv2.LINE_AA)

    # 3. Coordinate axes on top (X=red, Y=green, Z=blue)
    axis_defs = [
        ([axis_len, 0, 0], (20, 20, 200)),
        ([0, axis_len, 0], (20, 180, 20)),
        ([0, 0, axis_len], (200, 80, 20)),
    ]
    for ep, col in axis_defs:
        e = px(np.array([ep]))[0]
        cv2.arrowedLine(canvas,
                        tuple(o.astype(int)), tuple(e.astype(int)),
                        col, 3, cv2.LINE_AA, tipLength=0.08)

    # 4. Query camera frustum (topmost, orange)
    # q_col = (50, 120, 200)
    q_col = (0, 68, 189) 
    qpts = px(_frustum_pts(query, fov_x, fov_y, frust_sz))
    _draw_frustum(canvas, qpts, q_col, 3)
    apex = tuple(qpts[0].astype(int))
    cv2.circle(canvas, apex, 5, q_col, -1)
    cv2.circle(canvas, apex, 6, q_col, 2, cv2.LINE_AA)

    # Auto-crop to content with margin
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    mask = gray < 250
    if mask.any():
        ys, xs = np.where(mask)
        margin = 60
        y0 = max(ys.min() - margin, 0)
        y1 = min(ys.max() + margin, fig_h)
        x0 = max(xs.min() - margin, 0)
        x1 = min(xs.max() + margin, fig_w)
        canvas = canvas[y0:y1, x0:x1]

    return canvas


if __name__ == "__main__":
    img = render()
    path = "/data2/qiwu2/DiffSynth-Studio/examples/wanvideo/query_centered_pose_normalization.png"
    cv2.imwrite(path, img)
    print(f"Saved: {path}  ({img.shape[1]}x{img.shape[0]})")
