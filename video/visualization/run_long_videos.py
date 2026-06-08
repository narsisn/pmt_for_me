"""
Pipeline: extract frames → run VIS visualization → reassemble to video.

Usage:
    python run_long_videos.py \
        --config-file /path/to/config.yaml \
        --weights /path/to/weight.pth \
        [--video-root /path/to/long_videos] \
        [--output-root /path/to/output] \
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


def extract_frames(video_path, frames_dir, fps):
    """Extract frames from a video at the given FPS using OpenCV."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = fps  # fallback
    frame_interval = src_fps / fps  # e.g. 1.0 when src==target

    frame_idx = 0
    saved = 0
    next_sample = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= next_sample:
            saved += 1
            out_path = os.path.join(frames_dir, f"{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            next_sample += frame_interval
        frame_idx += 1

    cap.release()
    print(f"[extract] {saved} frames from {video_path} (src {src_fps:.1f}fps → {fps}fps)")


def run_visualization(config_file, weights, frames_dir, output_dir, windows_size):
    """Run video_demo.py on extracted frames."""
    os.makedirs(output_dir, exist_ok=True)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_demo.py")
    # run from the project root so relative config paths resolve correctly
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


def frames_to_video(frames_dir, output_video, fps):
    """Reassemble predicted frames into an mp4 video using OpenCV."""
    # collect frame paths (natural sort so frame2 < frame10)
    paths = []
    for ext in ("jpg", "png", "jpeg"):
        paths = sorted(glob.glob(os.path.join(frames_dir, f"*.{ext}")), key=_natural_key)
        if paths:
            break
    if not paths:
        print(f"[encode]  WARNING: no frames found in {frames_dir}")
        return

    sample = cv2.imread(paths[0])
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for p in paths:
        frame = cv2.imread(p)
        writer.write(frame)

    writer.release()
    print(f"[encode]  {len(paths)} frames → {output_video}  ({w}x{h} @ {fps}fps)")


def find_frames(folder):
    """Find existing image frames in a folder or its 'frames' subfolder."""
    for search_dir in [os.path.join(folder, "frames"), folder]:
        for ext in ("jpg", "png", "jpeg"):
            hits = sorted(glob.glob(os.path.join(search_dir, f"*.{ext}")), key=_natural_key)
            if hits:
                return search_dir
    return None


def find_video(folder):
    """Return the first video file found inside *folder* (recursive one level)."""
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4"):
        hits = glob.glob(os.path.join(folder, "**", ext), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Long-video VIS pipeline")
    parser.add_argument("--config-file", required=True, help="Path to model config yaml")
    parser.add_argument("--weights", required=True, help="Path to model weights .pth")
    parser.add_argument(
        "--video-root",
        default="/projects/0/tesei0745/Narges/vis_for_gijs/VidEoMT_VIS/Video_instance_segmentation/long_videos",
        help="Root folder containing video sub-folders",
    )
    parser.add_argument(
        "--output-root",
        default="/projects/0/tesei0745/Narges/vis_for_gijs/VidEoMT_VIS/Video_instance_segmentation/long_videos_output",
        help="Root folder for all outputs",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--windows_size", type=int, default=20, help="Window size for semi-offline mode")
    args = parser.parse_args()

    # discover sub-folders that contain videos
    subdirs = sorted(
        d for d in os.listdir(args.video_root)
        if os.path.isdir(os.path.join(args.video_root, d))
    )

    for name in subdirs:
        video_dir = os.path.join(args.video_root, name)
        video_path = find_video(video_dir)
        existing_frames_dir = find_frames(video_dir)

        if video_path is None and existing_frames_dir is None:
            print(f"[SKIP] no video or frames found in {video_dir}")
            continue

        print(f"\n{'='*60}")
        if video_path:
            print(f"Processing: {name}  (video: {video_path})")
        else:
            print(f"Processing: {name}  (frames: {existing_frames_dir})")
        print(f"{'='*60}")

        frames_dir = os.path.join(args.output_root, name, "frames")
        pred_dir = os.path.join(args.output_root, name, "predictions")
        out_video = os.path.join(args.output_root, name, f"{name}_pred.mp4")

        # check if already done
        if os.path.isfile(out_video):
            print(f"[SKIP] output video already exists: {out_video}")
            continue

        # 1) extract frames (skip if frames already exist)
        existing_frames = glob.glob(os.path.join(frames_dir, "*.jpg")) + \
                          glob.glob(os.path.join(frames_dir, "*.png"))
        if existing_frames:
            print(f"[SKIP] frames already extracted ({len(existing_frames)} files)")
        elif existing_frames_dir and existing_frames_dir != frames_dir:
            # frames exist in the input folder — use them directly
            frames_dir = existing_frames_dir
            n = len(glob.glob(os.path.join(frames_dir, "*.*")))
            print(f"[USE]  using existing frames from {frames_dir} ({n} files)")
        elif video_path:
            extract_frames(video_path, frames_dir, args.fps)
        else:
            print(f"[SKIP] no frames and no video for {name}")
            continue

        # 2) run visualization (skip if predictions already exist)
        existing_preds = glob.glob(os.path.join(pred_dir, "*.jpg"))
        if existing_preds:
            print(f"[SKIP] predictions already exist ({len(existing_preds)} files)")
        else:
            run_visualization(args.config_file, args.weights, frames_dir, pred_dir, args.windows_size)

        # 3) reassemble to video
        frames_to_video(pred_dir, out_video, args.fps)

        # 4) also create original video from input frames if no source video exists
        org_video = os.path.join(args.output_root, name, f"{name}_org.mp4")
        if not os.path.isfile(org_video):
            frames_to_video(frames_dir, org_video, args.fps)
            print(f"[DONE] original video → {org_video}")

        print(f"[DONE] prediction video → {out_video}")

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
