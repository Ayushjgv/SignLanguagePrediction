"""
Simulate the realtime_detect.py pipeline on a video file (headless).

Runs the EXACT same logic as realtime_detect.py -- rolling 30-frame buffer,
idle detection, prediction smoothing, static_image_mode=False tracking
detectors -- but reads frames from a video instead of a webcam and skips the
cv2.imshow GUI window. This proves the realtime detection logic works even
though this machine has no physical camera/display.

Usage:
    .\\.venv\\Scripts\\python.exe simulate_realtime.py --video Video_Dataset/Crocodile/<file>.mp4 --expect crocodile
    .\\.venv\\Scripts\\python.exe simulate_realtime.py --video test_videos/drink.mp4 --expect drink
"""
import os
import time
import argparse
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_utils import extract_normalized_features
from video_model import load_model

SEQUENCE_LENGTH = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/video_model_mirror.pth")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--smoothing", type=int, default=5)
    ap.add_argument("--buffer", type=int, default=75)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return
    model, labels = load_model(args.model)
    model.eval()
    print(f"Model: {args.model} ({len(labels)} classes)")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

    history_len = max(SEQUENCE_LENGTH, args.buffer)

    sequence_buffer = deque(maxlen=history_len)
    prediction_history = deque(maxlen=args.smoothing)
    current_prediction = "Waiting for gesture..."
    current_confidence = 0.0

    frame_idx = 0
    pred_counts = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # same as realtime (selfie view)

        frame_features, _, _ = extract_normalized_features(frame, pose_detector, hands_detector)
        sequence_buffer.append(frame_features)
        frame_idx += 1

        if len(sequence_buffer) == history_len and frame_idx % args.stride == 0:
            buf_arr = np.array(sequence_buffer, dtype=np.float32)
            sample_idx = np.linspace(0, history_len - 1, SEQUENCE_LENGTH, dtype=int)
            seq_arr = buf_arr[sample_idx]
            hands_activity = np.count_nonzero(seq_arr[:, 99:])
            if hands_activity < 40:
                current_prediction = "Idle"
                current_confidence = 0.0
            else:
                input_tensor = torch.tensor(seq_arr).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
                prediction_history.append(probabilities)
                avg_probs = np.mean(prediction_history, axis=0)
                top_idx = int(np.argmax(avg_probs))
                top_confidence = float(avg_probs[top_idx])
                if top_confidence >= args.threshold:
                    current_prediction = labels[top_idx]
                    current_confidence = top_confidence
                else:
                    current_prediction = f"Uncertain({labels[top_idx]})"
                    current_confidence = top_confidence
                pred_counts[current_prediction] = pred_counts.get(current_prediction, 0) + 1

        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx:4d} -> {current_prediction} ({current_confidence*100:.1f}%)")

    cap.release()
    pose_detector.close()
    hands_detector.close()

    print("\n" + "=" * 55)
    print(f"Video: {args.video}")
    if args.expect:
        hit = pred_counts.get(args.expect, 0)
        total = sum(v for k, v in pred_counts.items() if k != "Idle")
        pct = 100.0 * hit / total if total else 0
        status = "OK" if hit == max(pred_counts, key=pred_counts.get) and hit > 0 else "CHECK"
        print(f"Expected '{args.expect}' predicted on {hit}/{total} active frames ({pct:.0f}%)  [{status}]")
    print("Prediction distribution over active frames:")
    for k, v in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:22s}: {v}")
    print("=" * 55)


if __name__ == "__main__":
    main()
