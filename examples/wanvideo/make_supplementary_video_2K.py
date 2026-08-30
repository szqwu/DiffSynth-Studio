#!/usr/bin/env python3
"""
Create supplementary videos for 6-to-1 NVS results on DL3DV-2K random scenes.

Reads selection.json from each scene's output directory to determine
input/target frame indices. Data is loaded from /data2/qiwu2/2K/{hash}/.

Two layout modes:
  --camviz (default): camera pose visualization on the left, generated view on the right,
                      input views strip on top.
  --no-camviz:        generated view on top, input views strip on the bottom.
"""

import argparse
import json
import os
import random
import re

import cv2
import imageio
import numpy as np
from PIL import Image


BG = 255


# ─── Utility helpers ────────────────────────────────────────────────────────

def resize_crop_to_rect(img, target_h, target_w):
    """Resize to cover target_h x target_w then center crop."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    scale = max(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cy, cx = (new_h - target_h) // 2, (new_w - target_w) // 2
    return img_resized[cy:cy + target_h, cx:cx + target_w]


def make_input_strip(views, strip_w, gap=6):
    """Arrange N views in a single row scaled to fit strip_w total width."""
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
    """Put a centered label at y_offset on the canvas."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    tx = (canvas.shape[1] - tw) // 2
    cv2.putText(canvas, text, (tx, y_offset), font, font_scale, color,
                1, cv2.LINE_AA)


def subsample_frames(frames, max_frames):
    """Evenly subsample to at most max_frames entries."""
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]


def collect_generated_frames(scene_dir, model_h, model_w):
    """Collect generated_frame_XXXX_HxW.png sorted by frame index."""
    pattern = re.compile(rf"generated_frame_(\d+)_{model_h}x{model_w}\.png")
    entries = []
    for fname in os.listdir(scene_dir):
        m = pattern.match(fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(scene_dir, fname)))
    entries.sort(key=lambda x: x[0])
    return entries


def select_best_psnr(gen_frames, psnr_map, max_frames):
    """Select up to max_frames with the highest PSNR, returned in frame-index order."""
    scored = [(idx, path, psnr_map.get(idx, 0.0)) for idx, path in gen_frames]
    scored.sort(key=lambda x: x[2], reverse=True)
    selected = scored[:max_frames]
    selected.sort(key=lambda x: x[0])
    return [(idx, path) for idx, path, _ in selected]


# ─── Camera pose visualization (3D wireframe frustums) ───────────────────────

def _frustum_pts_world(c2w, fov_x, fov_y, size, opengl=True):
    """Return 5 wireframe points [apex, BL, BR, TR, TL] in world coordinates."""
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
    """Build a view matrix looking at cameras from behind-and-above."""
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
    """Perspective-project 3D points -> normalised 2D (before canvas scaling)."""
    rotated = (view_rot @ (pts_3d - center).T).T
    d = 6.0
    f = d / np.tan(np.radians(fov_deg) / 2.0)
    x2d = f * rotated[:, 0] / (rotated[:, 2] + d)
    y2d = f * rotated[:, 1] / (rotated[:, 2] + d)
    return x2d, y2d


def _draw_dashed_line(canvas, pt1, pt2, color, thickness=1,
                      dash_len=8, gap_len=6):
    """Draw a dashed line between two points."""
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


def render_camera_viz(viz_h, viz_w, c2ws, train_k, test_k, query_k,
                      input_thumbs, opengl=True, padding=35):
    """Render 3D wireframe camera frustums from a bird's-eye view."""
    canvas = np.ones((viz_h, viz_w, 3), dtype=np.uint8) * BG

    fov_x, fov_y = np.radians(73), np.radians(46)
    positions = c2ws[:, :3, 3]
    spread = np.linalg.norm(positions.max(0) - positions.min(0))
    dists = [np.linalg.norm(positions[train_k[i]] - positions[train_k[i + 1]])
             for i in range(len(train_k) - 1)]
    min_dist = min(dists) if dists else spread
    frustum_size = max(min(spread * 0.08, min_dist * 0.45), spread * 0.02, 1e-4)

    view_rot, center = _build_view_matrix(c2ws, elev_deg=70, opengl=opengl)

    input_frust_3d = np.concatenate([
        _frustum_pts_world(c2ws[k], fov_x, fov_y, frustum_size, opengl)
        for k in train_k], axis=0)
    test_pos_3d = positions[test_k]
    fixed_3d = np.concatenate([input_frust_3d, test_pos_3d], axis=0)
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

    draw_ks = list(train_k) + [query_k]
    projected = {}
    for k in draw_ks:
        pts3 = _frustum_pts_world(c2ws[k], fov_x, fov_y, frustum_size, opengl)
        projected[k] = proj_to_px(pts3)

    def draw_frustum(k, color, thickness):
        pts = projected[k].astype(np.int32)
        apex = tuple(pts[0])
        corners = [tuple(pts[i]) for i in range(1, 5)]
        for c in corners:
            cv2.line(canvas, apex, c, color, thickness, cv2.LINE_AA)
        for i in range(4):
            cv2.line(canvas, corners[i], corners[(i + 1) % 4],
                     color, thickness, cv2.LINE_AA)

    dash_color = (200, 200, 200)
    for i in range(len(train_k) - 1):
        p1 = tuple(projected[train_k[i]][0].astype(int))
        p2 = tuple(projected[train_k[i + 1]][0].astype(int))
        _draw_dashed_line(canvas, p1, p2, dash_color, thickness=1,
                          dash_len=10, gap_len=8)

    input_color = (50, 120, 200)
    for i, k in enumerate(train_k):
        thumb = cv2.resize(input_thumbs[i], (80, 48),
                           interpolation=cv2.INTER_AREA)
        th_h, th_w = thumb.shape[:2]
        src = np.array([[0, 0], [th_w, 0], [th_w, th_h], [0, th_h]],
                       dtype=np.float32)
        dst = projected[k][[3, 4, 1, 2]].astype(np.float32)
        try:
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(thumb, M, (viz_w, viz_h))
            mask = cv2.warpPerspective(
                np.ones((th_h, th_w), dtype=np.uint8) * 255,
                M, (viz_w, viz_h))
            canvas[mask > 0] = warped[mask > 0]
        except cv2.error:
            pass

    for k in train_k:
        draw_frustum(k, input_color, 2)

    query_color = (180, 60, 0)
    draw_frustum(query_k, query_color, 2)
    apex = projected[query_k][0].astype(int)
    cv2.circle(canvas, tuple(apex), 5, query_color, -1)
    cv2.circle(canvas, tuple(apex), 6, query_color, 2, cv2.LINE_AA)

    return canvas


# ─── Scene data loader for 2K random ────────────────────────────────────────

def load_2k_scene_cameras(scene_hash, data_path, results_dir,
                          num_input_frames, model_h, model_w):
    """Load input views and camera poses for a DL3DV-2K random scene.

    Reads selection.json from the results directory for frame indices,
    and transforms.json + images from the 2K data directory.
    """
    selection_path = os.path.join(results_dir, scene_hash, "selection.json")
    with open(selection_path) as f:
        sel = json.load(f)
    input_indices = sel["input_indices"][:num_input_frames]
    target_indices = sel["target_indices"]

    tf_path = os.path.join(data_path, scene_hash, "transforms.json")
    with open(tf_path) as f:
        tf = json.load(f)
    frames = tf["frames"]
    img_dir = os.path.join(data_path, scene_hash, "images_4")

    all_ids = input_indices + target_indices
    c2ws = np.stack([np.array(frames[i]["transform_matrix"], dtype=np.float32)
                     for i in all_ids])

    input_views = []
    for idx in input_indices:
        fname = os.path.basename(
            frames[idx]["file_path"].replace("images/", "images_4/"))
        img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
        input_views.append(resize_crop_to_rect(img, model_h, model_w))

    id_to_k = {idx: k for k, idx in enumerate(all_ids)}
    train_k = [id_to_k[i] for i in input_indices]
    test_k = [id_to_k[i] for i in target_indices]
    return c2ws, input_views, train_k, test_k, id_to_k


def load_2k_input_views(scene_hash, data_path, results_dir,
                        num_input_frames, model_h, model_w):
    """Load 2K input views cropped to model resolution (no camera data)."""
    selection_path = os.path.join(results_dir, scene_hash, "selection.json")
    with open(selection_path) as f:
        sel = json.load(f)
    input_indices = sel["input_indices"][:num_input_frames]

    tf_path = os.path.join(data_path, scene_hash, "transforms.json")
    with open(tf_path) as f:
        tf = json.load(f)
    frames = tf["frames"]
    img_dir = os.path.join(data_path, scene_hash, "images_4")

    views = []
    for idx in input_indices:
        fname = os.path.basename(
            frames[idx]["file_path"].replace("images/", "images_4/"))
        img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
        views.append(resize_crop_to_rect(img, model_h, model_w))
    return views


# ─── Main ────────────────────────────────────────────────────────────────────

def make_supplementary_video(
    results_dir,
    data_path="/data2/qiwu2/2K",
    output_dir=None,
    scenes=None,
    num_input_frames=6,
    model_h=480,
    model_w=832,
    fps=2,
    duration=5.0,
    crf=18,
    shuffle=False,
    best_psnr=False,
    scores_json=None,
    seed=42,
    camviz=False,
    viz_w=480,
    label_h=28,
    gap=6,
):
    if output_dir is None:
        output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Auto-discover scenes from results directory
    all_scene_hashes = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and os.path.isfile(os.path.join(results_dir, d, "selection.json"))
    ])

    if scenes is not None:
        scene_hashes = [all_scene_hashes[i] for i in scenes]
    else:
        scene_hashes = all_scene_hashes

    print(f"Found {len(all_scene_hashes)} scenes, processing {len(scene_hashes)}")

    psnr_data = {}
    if best_psnr:
        if scores_json is None:
            candidates = [f for f in os.listdir(results_dir)
                          if f.startswith("scores") and f.endswith(".json")]
            if candidates:
                scores_json = os.path.join(results_dir, sorted(candidates)[0])
        if scores_json and os.path.isfile(scores_json):
            print(f"Loading PSNR scores from {scores_json}")
            with open(scores_json) as f:
                sdata = json.load(f)
            for sd in sdata["per_scene"]:
                psnr_data[sd["scene"]] = {
                    f["frame_idx"]: f["psnr"] for f in sd["per_frame"]}
        else:
            print("Warning: --best_psnr set but no scores JSON found, "
                  "falling back to even subsampling")
            best_psnr = False

    for scene_hash in scene_hashes:
        scene_dir = os.path.join(results_dir, scene_hash)
        if not os.path.isdir(scene_dir):
            print(f"Skipping {scene_hash[:12]}... (no results dir)")
            continue

        print(f"Processing {scene_hash[:12]}...")

        if camviz:
            c2ws, input_views, train_k, test_k, id_to_k = \
                load_2k_scene_cameras(scene_hash, data_path, results_dir,
                                      num_input_frames, model_h, model_w)
            opengl_conv = True  # 2K data uses OpenGL convention
        else:
            input_views = load_2k_input_views(
                scene_hash, data_path, results_dir,
                num_input_frames, model_h, model_w)

        gen_frames = collect_generated_frames(scene_dir, model_h, model_w)
        if not gen_frames:
            print(f"  No generated frames found, skipping")
            continue

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(gen_frames)
            rng.shuffle(input_views)

        max_frames = int(fps * duration)
        if best_psnr and scene_hash in psnr_data:
            gen_frames = select_best_psnr(gen_frames, psnr_data[scene_hash],
                                          max_frames)
        else:
            gen_frames = subsample_frames(gen_frames, max_frames)

        if camviz:
            total_w = viz_w + gap + model_w
        else:
            total_w = model_w

        input_strip = make_input_strip(input_views, total_w, gap=gap)
        strip_h = input_strip.shape[0]

        if camviz:
            canvas_h = label_h + strip_h + gap + model_h
        else:
            canvas_h = model_h + gap + label_h + strip_h

        canvas_h += canvas_h % 2
        total_w += total_w % 2

        suffix = "camviz" if camviz else ""
        name = f"supplementary_{suffix + '_' if suffix else ''}{scene_hash[:12]}.mp4"
        out_path = os.path.join(output_dir, name)
        writer = imageio.get_writer(
            out_path, fps=fps, codec="libx264",
            output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
            macro_block_size=1,
        )

        for frame_idx, frame_path in gen_frames:
            gen_img = np.array(Image.open(frame_path).convert("RGB"))
            if gen_img is None:
                continue

            canvas = np.ones((canvas_h, total_w, 3), dtype=np.uint8) * BG

            if camviz:
                query_k = id_to_k.get(frame_idx)
                if query_k is None:
                    continue
                cam_panel = render_camera_viz(
                    model_h, viz_w, c2ws,
                    train_k, test_k, query_k, input_views,
                    opengl=opengl_conv)

                y = 0
                add_label(canvas, f"{num_input_frames} Input Views",
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
                add_label(canvas, f"{num_input_frames} Input Views",
                          y + label_h - 8, font_scale=0.55)
                y += label_h
                canvas[y:y + strip_h, :model_w] = input_strip

            writer.append_data(canvas)

        writer.close()
        actual_dur = len(gen_frames) / fps
        print(f"  Saved: {out_path}  ({len(gen_frames)} frames @ {fps} fps = {actual_dur:.1f}s)")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create supplementary videos for 6-to-1 NVS on DL3DV-2K random scenes")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to evaluation results directory")
    parser.add_argument("--data_path", type=str,
                        default="/data2/qiwu2/2K",
                        help="Path to DL3DV-2K data (contains transforms.json per scene)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to results_dir)")
    parser.add_argument("--scenes", type=int, nargs="+", default=None,
                        help="Scene indices to process (0-based). Default: all.")
    parser.add_argument("--num_input_frames", type=int, default=6)
    parser.add_argument("--model_h", type=int, default=480)
    parser.add_argument("--model_w", type=int, default=832)
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Output video fps (low for per-frame generation)")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="Target video duration in seconds (frames subsampled to fit)")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 quality: 0=lossless, 18=visually lossless, 23=default")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomly shuffle input view order and output frame order")
    parser.add_argument("--best_psnr", action="store_true",
                        help="Select frames with highest PSNR (requires scores JSON)")
    parser.add_argument("--scores_json", type=str, default=None,
                        help="Path to scores JSON file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--camviz", action="store_true",
                        help="Add camera pose visualization panel on the left")
    parser.add_argument("--viz_w", type=int, default=480,
                        help="Width of camera visualization panel (default: 480)")
    args = parser.parse_args()

    make_supplementary_video(
        results_dir=args.results_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        scenes=args.scenes,
        num_input_frames=args.num_input_frames,
        model_h=args.model_h,
        model_w=args.model_w,
        fps=args.fps,
        duration=args.duration,
        crf=args.crf,
        shuffle=args.shuffle,
        best_psnr=args.best_psnr,
        scores_json=args.scores_json,
        seed=args.seed,
        camviz=args.camviz,
        viz_w=args.viz_w,
    )
