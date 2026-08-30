#!/usr/bin/env python3
"""
Create supplementary videos for RE10K boomerang/forward video interpolation results.

Layout: generated video frame on top, K input views strip on the bottom.
Input views are read from {scene_dir}/input_views/ (already at model resolution).
"""

import argparse
import os
import re

import cv2
import imageio
import numpy as np
from PIL import Image


BG = 255


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


def make_supplementary_video(
    results_dir,
    output_dir=None,
    model_h=480,
    model_w=832,
    crf=18,
    label_h=28,
    gap=6,
    k_values=None,
    scenes=None,
):
    if output_dir is None:
        output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Discover K subdirectories
    all_k_dirs = sorted([
        d for d in os.listdir(results_dir)
        if d.startswith("K") and os.path.isdir(os.path.join(results_dir, d))
    ])
    if k_values:
        all_k_dirs = [f"K{k}" for k in k_values if f"K{k}" in all_k_dirs]

    print(f"K directories: {all_k_dirs}")

    for k_dir in all_k_dirs:
        k_path = os.path.join(results_dir, k_dir)
        k_val = int(k_dir[1:])

        scene_list = sorted([
            d for d in os.listdir(k_path)
            if os.path.isdir(os.path.join(k_path, d))
            and os.path.isfile(os.path.join(k_path, d, "gen_video.mp4"))
        ])

        if scenes is not None:
            scene_list = [scene_list[i] for i in scenes if i < len(scene_list)]

        print(f"\n{k_dir}: {len(scene_list)} scenes")

        for scene_id in scene_list:
            scene_dir = os.path.join(k_path, scene_id)
            print(f"  Processing {scene_id}...")

            # Load input views
            input_dir = os.path.join(scene_dir, "input_views")
            view_files = sorted([
                f for f in os.listdir(input_dir) if f.endswith(".png")
            ])
            input_views = [
                np.array(Image.open(os.path.join(input_dir, f)).convert("RGB"))
                for f in view_files
            ]

            # Read generated video
            gen_video_path = os.path.join(scene_dir, "gen_video.mp4")
            reader = imageio.get_reader(gen_video_path)
            gen_fps = reader.get_meta_data().get("fps", 16)
            gen_frames = [frame for frame in reader]
            reader.close()

            total_w = model_w
            input_strip = make_input_strip(input_views, total_w, gap=gap)
            strip_h = input_strip.shape[0]
            canvas_h = model_h + gap + label_h + strip_h
            canvas_h += canvas_h % 2
            total_w += total_w % 2

            name = f"supplementary_{k_dir}_{scene_id}.mp4"
            out_path = os.path.join(output_dir, name)
            writer = imageio.get_writer(
                out_path, fps=gen_fps, codec="libx264",
                output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
                macro_block_size=1,
            )

            for gen_img in gen_frames:
                if gen_img.shape[:2] != (model_h, model_w):
                    gen_img = cv2.resize(gen_img, (model_w, model_h),
                                         interpolation=cv2.INTER_AREA)

                canvas = np.ones((canvas_h, total_w, 3), dtype=np.uint8) * BG
                y = 0
                canvas[y:y + model_h, :model_w] = gen_img
                y += model_h + gap
                add_label(canvas, f"{k_val} Input Views",
                          y + label_h - 8, font_scale=0.55)
                y += label_h
                canvas[y:y + strip_h, :model_w] = input_strip

                writer.append_data(canvas)

            writer.close()
            actual_dur = len(gen_frames) / gen_fps
            print(f"    Saved: {out_path}  ({len(gen_frames)} frames "
                  f"@ {gen_fps} fps = {actual_dur:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create supplementary videos for RE10K video interpolation")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_h", type=int, default=480)
    parser.add_argument("--model_w", type=int, default=832)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--k_values", type=int, nargs="+", default=None,
                        help="Only process these K values (e.g. 3 5)")
    parser.add_argument("--scenes", type=int, nargs="+", default=None,
                        help="Scene indices within each K group (0-based)")
    args = parser.parse_args()

    make_supplementary_video(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        model_h=args.model_h,
        model_w=args.model_w,
        crf=args.crf,
        k_values=args.k_values,
        scenes=args.scenes,
    )
