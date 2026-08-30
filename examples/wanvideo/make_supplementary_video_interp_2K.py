#!/usr/bin/env python3
"""
Create supplementary videos for video interpolation results on DL3DV-2K scenes.

Layout (--camviz, default):
  Top row:   6 keyframe thumbnails
  Bottom-left:  Camera trajectory visualization (colored trajectory line)
  Bottom-right: Generated video frame

Layout (--no-camviz):
  Top:    Generated video frame
  Bottom: 6 keyframe thumbnails

The trajectory is drawn as a continuous color-changing line (blue → red)
with keyframe frustums + thumbnails and the current frame marked.
"""

import argparse
import json
import os
import re

import cv2
import imageio
import numpy as np
from PIL import Image


BG = 255
SEED = 42


# ─── Utility helpers ────────────────────────────────────────────────────────

def resize_crop_to_rect(img, target_h, target_w):
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    scale = max(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cy, cx = (new_h - target_h) // 2, (new_w - target_w) // 2
    return img_resized[cy:cy + target_h, cx:cx + target_w]


def make_input_strip(views, strip_w, gap=6):
    n = len(views)
    aspect = views[0].shape[1] / views[0].shape[0]
    total_gaps = gap * (n - 1)
    thumb_w = (strip_w - total_gaps) // n
    thumb_h = int(thumb_w / aspect)
    strip = np.ones((thumb_h, strip_w, 3), dtype=np.uint8) * BG
    x = 0
    for v in views:
        thumb = cv2.resize(v, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        end_x = min(x + thumb_w, strip_w)
        strip[:, x:end_x] = thumb[:, :end_x - x]
        x += thumb_w + gap
    return strip


def add_label(canvas, text, y_offset, font_scale=0.6, color=(60, 60, 60)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    tx = (canvas.shape[1] - tw) // 2
    cv2.putText(canvas, text, (tx, y_offset), font, font_scale, color,
                1, cv2.LINE_AA)


# ─── Camera visualization with colored trajectory ──────────────────────────

def _frustum_pts_world(c2w, fov_x, fov_y, size, opengl=True):
    pos = c2w[:3, 3]
    right = c2w[:3, 0]
    up = c2w[:3, 1] if opengl else -c2w[:3, 1]
    fwd = -c2w[:3, 2] if opengl else c2w[:3, 2]
    hx = size * np.tan(fov_x / 2)
    hy = size * np.tan(fov_y / 2)
    ctr = pos + fwd * size
    return np.array([
        pos,
        ctr - right * hx - up * hy,
        ctr + right * hx - up * hy,
        ctr + right * hx + up * hy,
        ctr - right * hx + up * hy,
    ])


def _build_view_matrix(c2ws, elev_deg=55, opengl=True):
    positions = c2ws[:, :3, 3]
    center = positions.mean(0)

    ups = c2ws[:, :3, 1] if opengl else -c2ws[:, :3, 1]
    fwds = -c2ws[:, :3, 2] if opengl else c2ws[:, :3, 2]

    avg_up = ups.mean(0)
    avg_up /= max(np.linalg.norm(avg_up), 1e-12)
    avg_fwd = fwds.mean(0)
    avg_fwd -= np.dot(avg_fwd, avg_up) * avg_up
    avg_fwd /= max(np.linalg.norm(avg_fwd), 1e-12)

    a = np.radians(elev_deg)
    view_z = avg_fwd * np.cos(a) - avg_up * np.sin(a)
    view_z /= max(np.linalg.norm(view_z), 1e-12)

    cam_right = np.cross(avg_fwd, avg_up)
    cam_right /= max(np.linalg.norm(cam_right), 1e-12)
    view_x = cam_right - np.dot(cam_right, view_z) * view_z
    view_x /= max(np.linalg.norm(view_x), 1e-12)

    view_y = np.cross(view_x, view_z)
    if np.dot(view_y, avg_up) < 0:
        view_x = -view_x
        view_y = -view_y

    return np.stack([view_x, view_y, view_z]), center


def _project_perspective(pts_3d, view_rot, center, fov_deg=50):
    rotated = (view_rot @ (pts_3d - center).T).T
    d = 6.0
    f = d / np.tan(np.radians(fov_deg) / 2.0)
    x2d = f * rotated[:, 0] / (rotated[:, 2] + d)
    y2d = f * rotated[:, 1] / (rotated[:, 2] + d)
    return x2d, y2d


def _color_for_t(t):
    """Return BGR color for parameter t in [0, 1]: blue(0) → cyan → green → yellow → red(1)."""
    r = np.clip(min(4 * t - 2, 1.0), 0, 1)
    g = np.clip(min(4 * t, 4 - 4 * t), 0, 1)
    b = np.clip(1 - 4 * t, 0, 1)
    return (int(b * 220 + 30), int(g * 220 + 30), int(r * 220 + 30))


def render_camera_viz_trajectory(viz_h, viz_w, all_c2ws, keyframe_local,
                                 current_local, input_thumbs,
                                 opengl=True, padding=35):
    """Render camera trajectory with color gradient, keyframe frustums, and current marker."""
    canvas = np.ones((viz_h, viz_w, 3), dtype=np.uint8) * BG

    fov_x, fov_y = np.radians(73), np.radians(46)
    n_frames = len(all_c2ws)
    positions = all_c2ws[:, :3, 3]
    spread = np.linalg.norm(positions.max(0) - positions.min(0))

    kf_dists = [np.linalg.norm(positions[keyframe_local[i]] - positions[keyframe_local[i + 1]])
                for i in range(len(keyframe_local) - 1)]
    min_dist = min(kf_dists) if kf_dists else spread
    frustum_size = max(min(spread * 0.08, min_dist * 0.45), spread * 0.02, 1e-4)

    view_rot, center = _build_view_matrix(all_c2ws, elev_deg=70, opengl=opengl)

    # Fixed bounding box from all trajectory positions + keyframe frustums
    kf_frust_3d = np.concatenate([
        _frustum_pts_world(all_c2ws[k], fov_x, fov_y, frustum_size, opengl)
        for k in keyframe_local], axis=0)
    fixed_3d = np.concatenate([kf_frust_3d, positions], axis=0)
    fix_x, fix_y = _project_perspective(fixed_3d, view_rot, center, fov_deg=50)
    fix_2d = np.stack([fix_x, fix_y], axis=1)
    mn, mx = fix_2d.min(0), fix_2d.max(0)
    rng = mx - mn
    rng[rng == 0] = 1
    mn -= rng * 0.1
    mx += rng * 0.1
    rng = mx - mn
    usable = min(viz_h, viz_w) - 2 * padding
    scale = usable / rng.max()
    c2d = (mn + mx) / 2
    cc = np.array([viz_w / 2.0, viz_h / 2.0])

    def proj_to_px(pts_3d):
        x2d, y2d = _project_perspective(pts_3d, view_rot, center, fov_deg=50)
        px = (x2d - c2d[0]) * scale + cc[0]
        py = -(y2d - c2d[1]) * scale + cc[1]
        return np.stack([px, py], axis=1)

    # Project all trajectory positions to 2D
    traj_px = proj_to_px(positions)

    # 1) Draw trajectory with color gradient (full trajectory, dimmed for future)
    for i in range(n_frames - 1):
        t = i / max(n_frames - 1, 1)
        pt1 = tuple(traj_px[i].astype(int))
        pt2 = tuple(traj_px[i + 1].astype(int))
        color = _color_for_t(t)
        if i > current_local:
            # Dim future segments
            color = tuple(int(c * 0.3 + BG * 0.7) for c in color)
        thickness = 2 if i <= current_local else 1
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 2) Warp thumbnails into keyframe frustum image planes
    kf_projected = {}
    for k in keyframe_local:
        pts3 = _frustum_pts_world(all_c2ws[k], fov_x, fov_y, frustum_size, opengl)
        kf_projected[k] = proj_to_px(pts3)

    input_color = (50, 120, 200)
    for i, k in enumerate(keyframe_local):
        thumb = cv2.resize(input_thumbs[i], (80, 48), interpolation=cv2.INTER_AREA)
        th_h, th_w = thumb.shape[:2]
        src = np.array([[0, 0], [th_w, 0], [th_w, th_h], [0, th_h]], dtype=np.float32)
        dst = kf_projected[k][[3, 4, 1, 2]].astype(np.float32)
        try:
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(thumb, M, (viz_w, viz_h))
            mask = cv2.warpPerspective(
                np.ones((th_h, th_w), dtype=np.uint8) * 255, M, (viz_w, viz_h))
            canvas[mask > 0] = warped[mask > 0]
        except cv2.error:
            pass

    # 3) Keyframe frustum wireframes
    for k in keyframe_local:
        pts = kf_projected[k].astype(np.int32)
        apex = tuple(pts[0])
        corners = [tuple(pts[j]) for j in range(1, 5)]
        for c in corners:
            cv2.line(canvas, apex, c, input_color, 2, cv2.LINE_AA)
        for j in range(4):
            cv2.line(canvas, corners[j], corners[(j + 1) % 4], input_color, 2, cv2.LINE_AA)

    # 4) Current frame camera frustum
    cur_t = current_local / max(n_frames - 1, 1)
    cur_color = _color_for_t(cur_t)
    cur_pts3 = _frustum_pts_world(all_c2ws[current_local], fov_x, fov_y, frustum_size, opengl)
    cur_pts = proj_to_px(cur_pts3).astype(np.int32)
    cur_apex = tuple(cur_pts[0])
    cur_corners = [tuple(cur_pts[j]) for j in range(1, 5)]
    for c in cur_corners:
        cv2.line(canvas, cur_apex, c, cur_color, 2, cv2.LINE_AA)
    for j in range(4):
        cv2.line(canvas, cur_corners[j], cur_corners[(j + 1) % 4], cur_color, 2, cv2.LINE_AA)
    cv2.circle(canvas, cur_apex, 5, cur_color, -1)
    cv2.circle(canvas, cur_apex, 6, (40, 40, 40), 2, cv2.LINE_AA)

    return canvas


# ─── Scene data loading ─────────────────────────────────────────────────────

def parse_metrics_window(metrics_path):
    """Parse metrics.txt to get window start and keyframe gap."""
    window_start = None
    keyframe_gap = None
    total_frames = None
    with open(metrics_path) as f:
        for line in f:
            m = re.match(r"Window: \[(\d+), (\d+)\]", line.strip())
            if m:
                window_start = int(m.group(1))
            m2 = re.match(r"Keyframe gap: (\d+), Total frames: (\d+)", line.strip())
            if m2:
                keyframe_gap = int(m2.group(1))
                total_frames = int(m2.group(2))
    return window_start, keyframe_gap, total_frames


def load_scene_data(scene_hash, data_path, results_dir, model_h, model_w):
    """Load camera poses and keyframe images for a video interpolation scene."""
    metrics_path = os.path.join(results_dir, scene_hash, "metrics.txt")
    window_start, keyframe_gap, total_frames = parse_metrics_window(metrics_path)
    if window_start is None:
        raise ValueError(f"Could not parse window from {metrics_path}")

    keyframe_positions = [keyframe_gap * i for i in range(6)]
    window_indices = list(range(window_start, window_start + total_frames))

    tf_path = os.path.join(data_path, scene_hash, "transforms.json")
    with open(tf_path) as f:
        tf = json.load(f)
    frames = tf["frames"]
    img_dir = os.path.join(data_path, scene_hash, "images_4")

    # Camera poses for all frames in the window
    all_c2ws = np.stack([
        np.array(frames[gi]["transform_matrix"], dtype=np.float32)
        for gi in window_indices
    ])

    # Load keyframe images at model resolution
    keyframe_views = []
    for kp in keyframe_positions:
        gi = window_indices[kp]
        fname = os.path.basename(
            frames[gi]["file_path"].replace("images/", "images_4/"))
        img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
        keyframe_views.append(resize_crop_to_rect(img, model_h, model_w))

    return all_c2ws, keyframe_views, keyframe_positions, total_frames


# ─── Main ────────────────────────────────────────────────────────────────────

def make_supplementary_video(
    results_dir,
    data_path="/data2/qiwu2/2K",
    output_dir=None,
    scenes=None,
    model_h=480,
    model_w=832,
    crf=18,
    camviz=True,
    viz_w=480,
    label_h=28,
    gap=6,
):
    if output_dir is None:
        output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)

    all_scene_hashes = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and os.path.isfile(os.path.join(results_dir, d, "metrics.txt"))
        and os.path.isfile(os.path.join(results_dir, d, "gen_video.mp4"))
    ])

    if scenes is not None:
        scene_hashes = [all_scene_hashes[i] for i in scenes]
    else:
        scene_hashes = all_scene_hashes

    print(f"Found {len(all_scene_hashes)} scenes, processing {len(scene_hashes)}")

    for scene_hash in scene_hashes:
        scene_dir = os.path.join(results_dir, scene_hash)
        print(f"Processing {scene_hash[:12]}...")

        gen_video_path = os.path.join(scene_dir, "gen_video.mp4")
        reader = imageio.get_reader(gen_video_path)
        gen_meta = reader.get_meta_data()
        gen_fps = gen_meta.get("fps", 16)
        gen_frames_list = []
        for frame in reader:
            gen_frames_list.append(frame)
        reader.close()
        n_gen = len(gen_frames_list)
        print(f"  gen_video.mp4: {n_gen} frames @ {gen_fps} fps")

        if camviz:
            all_c2ws, keyframe_views, keyframe_positions, total_frames = \
                load_scene_data(scene_hash, data_path, results_dir, model_h, model_w)
            opengl_conv = True

            if n_gen != total_frames:
                print(f"  Warning: gen_video has {n_gen} frames but expected {total_frames}")
                total_frames = min(n_gen, total_frames)
                all_c2ws = all_c2ws[:total_frames]

            total_w = viz_w + gap + model_w
            input_strip = make_input_strip(keyframe_views, total_w, gap=gap)
            strip_h = input_strip.shape[0]
            canvas_h = label_h + strip_h + gap + model_h
        else:
            # Load keyframes for strip (without camera data)
            all_c2ws, keyframe_views, keyframe_positions, total_frames = \
                load_scene_data(scene_hash, data_path, results_dir, model_h, model_w)
            total_w = model_w
            input_strip = make_input_strip(keyframe_views, total_w, gap=gap)
            strip_h = input_strip.shape[0]
            canvas_h = model_h + gap + label_h + strip_h

        canvas_h += canvas_h % 2
        total_w += total_w % 2

        suffix = "camviz" if camviz else ""
        name = f"supplementary_interp_{suffix + '_' if suffix else ''}{scene_hash[:12]}.mp4"
        out_path = os.path.join(output_dir, name)
        writer = imageio.get_writer(
            out_path, fps=gen_fps, codec="libx264",
            output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
            macro_block_size=1,
        )

        for local_idx in range(min(n_gen, total_frames if camviz else n_gen)):
            gen_img = gen_frames_list[local_idx]
            if gen_img.shape[:2] != (model_h, model_w):
                gen_img = cv2.resize(gen_img, (model_w, model_h), interpolation=cv2.INTER_AREA)

            canvas = np.ones((canvas_h, total_w, 3), dtype=np.uint8) * BG

            if camviz:
                cam_panel = render_camera_viz_trajectory(
                    model_h, viz_w, all_c2ws,
                    keyframe_positions, local_idx, keyframe_views,
                    opengl=opengl_conv)

                y = 0
                add_label(canvas, "6 Keyframe Views",
                          y + label_h - 6, font_scale=0.5)
                y += label_h
                canvas[y:y + strip_h, :total_w] = input_strip
                y += strip_h + gap
                canvas[y:y + model_h, :viz_w] = cam_panel
                canvas[y:y + model_h, viz_w + gap:viz_w + gap + model_w] = gen_img
            else:
                y = 0
                canvas[y:y + model_h, :model_w] = gen_img
                y += model_h + gap
                add_label(canvas, "6 Keyframe Views",
                          y + label_h - 8, font_scale=0.55)
                y += label_h
                canvas[y:y + strip_h, :model_w] = input_strip

            writer.append_data(canvas)

        writer.close()
        actual_dur = min(n_gen, total_frames if camviz else n_gen) / gen_fps
        print(f"  Saved: {out_path}  ({min(n_gen, total_frames if camviz else n_gen)} frames "
              f"@ {gen_fps} fps = {actual_dur:.1f}s)")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create supplementary videos for video interpolation on DL3DV-2K")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to video interpolation results directory")
    parser.add_argument("--data_path", type=str, default="/data2/qiwu2/2K",
                        help="Path to DL3DV-2K data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenes", type=int, nargs="+", default=None,
                        help="Scene indices to process (0-based). Default: all.")
    parser.add_argument("--model_h", type=int, default=480)
    parser.add_argument("--model_w", type=int, default=832)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--camviz", action="store_true", default=True,
                        help="Add camera trajectory visualization (default: on)")
    parser.add_argument("--no-camviz", dest="camviz", action="store_false")
    parser.add_argument("--viz_w", type=int, default=480)
    args = parser.parse_args()

    make_supplementary_video(
        results_dir=args.results_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        scenes=args.scenes,
        model_h=args.model_h,
        model_w=args.model_w,
        crf=args.crf,
        camviz=args.camviz,
        viz_w=args.viz_w,
    )
