"""
Extract body pose and hand landmarks from video dataset (Video_Dataset)
using shared aspect-ratio and body-scale normalization.

Features per frame (total 225):
- 99 normalized body pose features (centered at mid-shoulder, scaled by shoulder width)
- 63 normalized left hand features (wrist relative to body, fingers relative to wrist)
- 63 normalized right hand features (wrist relative to body, fingers relative to wrist)

Features:
- PARALLEL extraction across CPU cores (each worker builds its own MediaPipe detectors)
- Auto-resume: Skips already extracted classes and videos
- Static image mode for sampled frames to ensure 100% hand detection
- Aspect-ratio corrected: Works identically on 16:9, 9:16, 4:3, etc.
"""

import argparse
import os
import queue
import threading

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from landmark_utils import extract_normalized_features, TOTAL_FEATURES

SEQUENCE_LENGTH = 30


def build_detectors():
    """Create fresh MediaPipe detectors (must be per-thread)."""
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    return pose_detector, hands_detector

def extract_landmarks_from_video(
    video_path,
    pose_detector,
    hands_detector,
    sequence_length=SEQUENCE_LENGTH,
    augment=False
):
    """
    Extracts normalized landmark sequence(s) from a video file.
    Returns a list of np.ndarray, each of shape (sequence_length, TOTAL_FEATURES).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        return []

    index_sets = []

    # 1. Main uniform sample across entire video clip
    indices_full = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    index_sets.append(indices_full)

    # 2. Optional augmentation for longer clips
    if augment and total_frames >= 45:
        start_trim = int(total_frames * 0.1)
        indices_aug1 = np.linspace(start_trim, total_frames - 1, sequence_length, dtype=int)
        index_sets.append(indices_aug1)

        end_trim = int(total_frames * 0.9)
        indices_aug2 = np.linspace(0, end_trim, sequence_length, dtype=int)
        index_sets.append(indices_aug2)

    sequences = []
    frame_cache = {}

    for idx_set in index_sets:
        seq = []
        for idx in idx_set:
            if idx not in frame_cache:
                feats, _, _ = extract_normalized_features(frames[idx], pose_detector, hands_detector)
                frame_cache[idx] = feats
            seq.append(frame_cache[idx])
        sequences.append(np.array(seq, dtype=np.float32))

    return sequences


def _worker(task_queue, result_queue, sequence_length, augment):
    """Worker thread: builds its own detectors and processes video tasks."""
    pose_detector, hands_detector = build_detectors()
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            vid_path, out_dir, out_base = task
            try:
                seqs = extract_landmarks_from_video(
                    vid_path, pose_detector, hands_detector,
                    sequence_length=sequence_length, augment=augment
                )
                saved = 0
                for s_idx, seq in enumerate(seqs):
                    out_filename = f"{out_base}_{s_idx}.npy"
                    np.save(os.path.join(out_dir, out_filename), seq)
                    saved += 1
                result_queue.put((True, vid_path, saved))
            except Exception as e:  # noqa: BLE001
                result_queue.put((False, vid_path, str(e)))
            finally:
                task_queue.task_done()
    finally:
        pose_detector.close()
        hands_detector.close()


def create_dataset_from_video_folder(
    video_folder="Video_Dataset",
    output_folder="data_video",
    sequence_length=SEQUENCE_LENGTH,
    target_classes=None,
    max_videos_per_class=None,
    augment=False,
    overwrite=False,
    num_workers=8
):
    """
    Processes videos from Video_Dataset and saves normalized landmark sequences.
    Uses a thread pool (each thread with its own MediaPipe detectors) for speed.
    """
    if not os.path.exists(video_folder):
        print(f"Error: Video folder '{video_folder}' not found!")
        return 0

    all_subdirs = sorted([
        d for d in os.listdir(video_folder)
        if os.path.isdir(os.path.join(video_folder, d))
    ])

    if target_classes:
        target_lower = [c.lower().strip() for c in target_classes]
        classes = [d for d in all_subdirs if d.lower().strip() in target_lower]
        if not classes:
            print(f"None of specified classes {target_classes} were found in {video_folder}!")
            return 0
    else:
        classes = all_subdirs

    # Build the full list of tasks, honoring resume/overwrite.
    tasks = []
    total_skipped = 0
    for class_name in classes:
        input_class_dir = os.path.join(video_folder, class_name)
        clean_label = class_name.lower().strip()
        output_class_dir = os.path.join(output_folder, clean_label)
        os.makedirs(output_class_dir, exist_ok=True)

        video_files = sorted([
            f for f in os.listdir(input_class_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ])
        if max_videos_per_class:
            video_files = video_files[:max_videos_per_class]

        for vid_file in video_files:
            base_name = os.path.splitext(vid_file)[0]
            first_output = os.path.join(output_class_dir, f"vid_{base_name}_0.npy")
            if os.path.exists(first_output) and not overwrite:
                total_skipped += 1
                continue
            vid_path = os.path.join(input_class_dir, vid_file)
            out_base = f"vid_{base_name}"
            tasks.append((vid_path, output_class_dir, out_base))

    print("=" * 65)
    print("NORMALIZED DATASET EXTRACTION (SCALE & POSITION INVARIANT)")
    print("=" * 65)
    print(f"Target Classes:         {len(classes)}")
    print(f"Already Completed:      {total_skipped} videos (auto-skipping)")
    print(f"Remaining to Process:   {len(tasks)} videos")
    print(f"Parallel Workers:       {num_workers}")
    print("=" * 65)

    if not tasks:
        print("\nAll classes already extracted! Nothing left to process.")
        return 0

    task_queue = queue.Queue()
    result_queue = queue.Queue()
    for t in tasks:
        task_queue.put(t)

    workers = []
    for _ in range(max(1, num_workers)):
        th = threading.Thread(
            target=_worker,
            args=(task_queue, result_queue, sequence_length, augment),
            daemon=True
        )
        th.start()
        workers.append(th)

    total_new_saved = 0
    total_failed = 0
    pbar = tqdm(total=len(tasks), desc="Extracting")
    processed = 0
    while processed < len(tasks):
        ok, vid_path, info = result_queue.get()
        processed += 1
        if ok:
            total_new_saved += int(info)
        else:
            total_failed += 1
            tqdm.write(f"  WARN failed {os.path.basename(vid_path)}: {info}")
        pbar.update(1)
    pbar.close()

    for _ in workers:
        task_queue.put(None)
    for th in workers:
        th.join()

    print("\n" + "=" * 65)
    print(f"EXTRACTION COMPLETE: {total_new_saved} new sequences saved in '{output_folder}'.")
    if total_skipped > 0:
        print(f"Skipped {total_skipped} previously extracted videos.")
    if total_failed > 0:
        print(f"Failed on {total_failed} videos.")
    print("=" * 65)
    return total_new_saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract normalized body pose and hand landmarks from Video_Dataset (parallel)"
    )
    parser.add_argument("--video_folder", default="Video_Dataset",
                        help="Folder containing class video subfolders (default: Video_Dataset)")
    parser.add_argument("--output_folder", default="data_video",
                        help="Folder to save extracted .npy feature files (default: data_video)")
    parser.add_argument("--sequence_length", type=int, default=30,
                        help="Number of frames per sequence (default: 30)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Specific class names to extract (optional)")
    parser.add_argument("--max_videos_per_class", type=int, default=None,
                        help="Max videos to process per class")
    parser.add_argument("--augment", action="store_true",
                        help="Generate augmented sequences from longer video clips")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force re-extraction of all files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker threads (default: 8)")

    args = parser.parse_args()

    create_dataset_from_video_folder(
        video_folder=args.video_folder,
        output_folder=args.output_folder,
        sequence_length=args.sequence_length,
        target_classes=args.classes,
        max_videos_per_class=args.max_videos_per_class,
        augment=args.augment,
        overwrite=args.overwrite,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()

