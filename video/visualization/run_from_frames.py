"""
Pipeline: read subfolders of frames → run VIS visualization → save predictions + videos.

Each subfolder under --frames-root should contain raw image frames (.jpg/.png).
For each subfolder the script creates under --output-root/<subfolder_name>:
    frames/          – predicted/visualised frames
    <name>_org.mp4   – video assembled from the raw input frames
    <name>_pred.mp4  – video assembled from the predicted frames

Usage:
    python run_from_frames.py \
        --config-file /path/to/config.yaml \
        --weights /path/to/weight.pth \
        --frames-root /path/to/input_frames_root \
        --output-root /path/to/output \
        [--val-list /path/to/val.txt] \
        [--fps 30] \
        [--windows_size 20]
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import cv2


def _natural_key(path):
    """Sort key that orders frame2 before frame10."""
    name = os.path.basename(path)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', name)]


def collect_frames(folder):
    """Return naturally-sorted list of image paths in *folder*."""
    for ext in ("jpg", "png", "jpeg"):
        paths = sorted(glob.glob(os.path.join(folder, f"*.{ext}")), key=_natural_key)
        if paths:
            return paths
    return []


def run_visualization(config_file, weights, frames_dir, output_dir, windows_size):
    """Run video_demo.py on the frames."""
    os.makedirs(output_dir, exist_ok=True)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_demo.py")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, script,
        "--config-file", config_file,
        "--input", frames_dir,
        "--output", output_dir,
        "--windows_size", str(windows_size),
        "--opts", "MODEL.WEIGHTS", weights,
    ]
    print(f"[vis]     {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=project_root)


def frames_to_video(frame_paths, output_video, fps):
    """Assemble a list of frame images into an mp4 video."""
    if not frame_paths:
        print(f"[encode]  WARNING: no frames provided for {output_video}")
        return

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    sample = cv2.imread(frame_paths[0])
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for p in frame_paths:
        frame = cv2.imread(p)
        writer.write(frame)

    writer.release()
    print(f"[encode]  {len(frame_paths)} frames → {output_video}  ({w}x{h} @ {fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Frames-based VIS pipeline")
    parser.add_argument("--config-file", required=True, help="Path to model config yaml")
    parser.add_argument("--weights", required=True, help="Path to model weights .pth")
    parser.add_argument(
        "--frames-root",
        required=True,
        help="Root folder whose subfolders each contain raw frames",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root folder for all outputs",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos")
    parser.add_argument("--windows_size", type=int, default=20, help="Window size for semi-offline mode")
    parser.add_argument(
        "--val-list",
        default=None,
        help="Path to a text file listing folder names to process (one per line). "
             "If not provided, all subfolders under --frames-root are processed.",
    )
    args = parser.parse_args()

    # discover subfolders that contain frames
    if args.val_list:
        with open(args.val_list, "r") as f:
            val_names = {line.strip() for line in f if line.strip()}
        subdirs = sorted(
            d for d in val_names
            if os.path.isdir(os.path.join(args.frames_root, d))
        )
        print(f"Loaded {len(val_names)} entries from {args.val_list}, "
              f"{len(subdirs)} found in {args.frames_root}")
    else:
        subdirs = sorted(
            d for d in os.listdir(args.frames_root)
            if os.path.isdir(os.path.join(args.frames_root, d))
        )

    if not subdirs:
        print(f"No subfolders found in {args.frames_root}")
        return

    for name in subdirs:
        src_frames_dir = os.path.join(args.frames_root, name)
        src_frames = collect_frames(src_frames_dir)

        if not src_frames:
            print(f"[SKIP] no image frames found in {src_frames_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {name}  ({len(src_frames)} frames)")
        print(f"{'='*60}")

        out_dir = os.path.join(args.output_root, name)
        pred_frames_dir = os.path.join(out_dir, "frames")
        org_video = os.path.join(out_dir, f"{name}_org.mp4")
        pred_video = os.path.join(out_dir, f"{name}_pred.mp4")

        # 1) run visualization → predicted frames into frames/
        existing_preds = collect_frames(pred_frames_dir)
        if existing_preds:
            print(f"[SKIP] predictions already exist ({len(existing_preds)} files)")
        else:
            run_visualization(args.config_file, args.weights, src_frames_dir, pred_frames_dir, args.windows_size)
            existing_preds = collect_frames(pred_frames_dir)

        # 2) create video from original raw frames
        if os.path.isfile(org_video):
            print(f"[SKIP] original video already exists: {org_video}")
        else:
            frames_to_video(src_frames, org_video, args.fps)
            print(f"[DONE] original video → {org_video}")

        # 3) create video from predicted frames
        if os.path.isfile(pred_video):
            print(f"[SKIP] prediction video already exists: {pred_video}")
        else:
            frames_to_video(existing_preds, pred_video, args.fps)
            print(f"[DONE] prediction video → {pred_video}")

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()
