#!/usr/bin/env python3
"""
Prepare dataset from NextX/Forking Paths data.
Extracts frames at 2.5fps (GPU via NVDEC), saves per-video manifests.
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

SOURCE_FPS = 30.0
TARGET_FPS = 5
FRAME_SKIP = int(SOURCE_FPS / TARGET_FPS)


def check_gpu_available() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True, timeout=5)
        return 'cuda' in result.stdout
    except Exception:
        return False


def load_tracks(json_path: Path) -> list[dict]:
    """Load annotations into track format: {id, bb: [[x, y, w, h, frame], ...]}."""
    with open(json_path) as f:
        data = json.load(f)

    tracks = defaultdict(list)
    for ann in data:
        if ann['class_name'] != 'Person':
            continue
        if ann['frame_id'] % FRAME_SKIP != 0:
            continue

        frame_id = ann['frame_id'] // FRAME_SKIP
        x, y, w, h = ann['bbox']
        tracks[int(ann['track_id'])].append([x, y, w, h, frame_id])

    return [
        {'id': tid, 'bb': sorted(pts, key=lambda p: p[4])}
        for tid, pts in tracks.items()
    ]


def extract_frames_ffmpeg(video_path: Path, output_dir: Path, frames: set[int], use_gpu: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_frames = sorted([f * FRAME_SKIP for f in frames])
    if not source_frames:
        return

    select_expr = '+'.join(f'eq(n,{f})' for f in source_frames)
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    if use_gpu:
        cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
    cmd.extend([
        '-i', str(video_path),
        '-vf', f"select='{select_expr}'" + (",hwdownload,format=nv12" if use_gpu else ""),
        '-vsync', 'vfr',
        str(output_dir / '%06d.jpg')
    ])
    subprocess.run(cmd, check=True, capture_output=True)

    # FFmpeg outputs 1-indexed sequential files (000001.jpg, 000002.jpg, ...)
    # Rename to match manifest frame indices
    for i, src_frame in enumerate(source_frames):
        old = output_dir / f'{i + 1:06d}.jpg'
        new = output_dir / f'{src_frame // FRAME_SKIP:06d}.jpg'
        if old.exists() and old != new:
            old.rename(new)


def process_video(args: tuple) -> dict:
    video_path, json_path, output_dir, do_extract, use_gpu = args
    name = video_path.stem

    try:
        tracks = load_tracks(json_path)
        if not tracks:
            return {'name': name, 'status': 'skip'}

        # Build per-frame bounding box lookup
        frame_bbs = defaultdict(list)
        max_frame_per_track = {}
        for t in tracks:
            tid = t['id']
            frames_in_track = [pt[4] for pt in t['bb']]
            max_frame_per_track[tid] = max(frames_in_track)
            for x, y, w, h, fid in t['bb']:
                frame_bbs[fid].append({
                    'agent_id': tid, 'x': x, 'y': y, 'width': w, 'height': h
                })

        # Save per-frame manifests
        manifest_dir = output_dir / 'manifests'
        manifest_dir.mkdir(parents=True, exist_ok=True)
        for fid, bbs in frame_bbs.items():
            has_future = any(
                max_frame_per_track[bb['agent_id']] > fid for bb in bbs
            )
            manifest = {
                'video_id': name,
                'frame': fid,
                'has_future': has_future,
                'bbs': bbs,
            }
            with open(manifest_dir / f'{name}_{fid:06d}.json', 'w') as f:
                json.dump(manifest, f)

        # Extract frames
        if do_extract:
            all_frames = {pt[4] for t in tracks for pt in t['bb']}
            extract_frames_ffmpeg(video_path, output_dir / 'frames' / name, all_frames, use_gpu)

        return {'name': name, 'status': 'ok', 'tracks': len(tracks)}
    except Exception as e:
        return {'name': name, 'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, default=Path('./dataset/next_x_v1_dataset'))
    parser.add_argument('--output_dir', type=Path, default=Path('./prepared_dataset'))
    parser.add_argument('--no_frames', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--workers', type=int, default=os.cpu_count() / 2)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    use_gpu = not args.cpu and check_gpu_available()
    print(f"Using {'GPU (NVDEC)' if use_gpu else 'CPU'} for video decoding")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted((args.input_dir / 'bbox').glob('*.json'))[:args.limit]
    tasks = [
        (args.input_dir / 'rgb_videos' / f"{j.stem}.mp4", j, args.output_dir, not args.no_frames, use_gpu)
        for j in json_files if (args.input_dir / 'rgb_videos' / f"{j.stem}.mp4").exists()
    ]

    stats = {'ok': 0, 'skip': 0, 'error': 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_video, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures)):
            r = f.result()
            stats[r['status']] += 1

    print(f"Done: {stats['ok']} videos, {stats['skip']} skipped, {stats['error']} errors")


if __name__ == '__main__':
    main()
