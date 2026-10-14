"""
Compare realtime buffering strategies on real videos (headless).

For each video we extract ALL frame features ONCE (tracking mode, like
realtime_detect.py), then simulate different buffering strategies on the
feature array:

  A) current  : 30 CONSECUTIVE frames, predict every frame (stride 1)
  B) arc      : long buffer (e.g. 90 frames), linspace-sample 30, stride 3

We report the DOMINANT predicted label over active frames and whether it
matches the expected sign. Strategy B is designed to match how the model was
trained (30 frames spread across the whole gesture, not a 1-second snippet).

Usage:
    .\\.venv\\Scripts\\python.exe compare_strategies.py
"""
import os
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_utils import extract_normalized_features
from video_model import load_model

SEQ = 30
MODEL = "models/video_model_mirror.pth"


def extract_all(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.flip(f, 1))  # selfie view, same as realtime
    cap.release()
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    feats = [extract_normalized_features(f, pose, hands)[0] for f in frames]
    pose.close()
    hands.close()
    return np.array(feats, dtype=np.float32)


def infer(seq, model):
    with torch.no_grad():
        out = model(torch.tensor(seq).unsqueeze(0))
        return torch.softmax(out, dim=1)[0].cpu().numpy()


def strat_consecutive(feats, model, labels, stride=1):
    """30 consecutive frames, slide by stride."""
    buf = deque(maxlen=SEQ)
    counts = Counter()
    for i, f in enumerate(feats):
        buf.append(f)
        if len(buf) == SEQ and i % stride == 0:
            arr = np.array(buf)
            if np.count_nonzero(arr[:, 99:]) < 40:
                continue
            p = infer(arr, model)
            counts[labels[int(np.argmax(p))]] += 1
    return counts


def strat_arc(feats, model, labels, buf_len=90, stride=3):
    """Long buffer, linspace-sample SEQ frames from it, slide by stride."""
    hist = deque(maxlen=buf_len)
    counts = Counter()
    for i, f in enumerate(feats):
        hist.append(f)
        if len(hist) == buf_len and i % stride == 0:
            arr = np.array(hist)
            if np.count_nonzero(arr[:, 99:]) < 40:
                continue
            idx = np.linspace(0, buf_len - 1, SEQ, dtype=int)
            p = infer(arr[idx], model)
            counts[labels[int(np.argmax(p))]] += 1
    return counts


def main():
    model, labels = load_model(MODEL)
    model.eval()

    tests = []
    for cls in ["Crocodile", "Carrot", "Cook", "Hello", "Tiger"]:
        d = os.path.join("Video_Dataset", cls)
        vids = sorted([x for x in os.listdir(d) if x.lower().endswith(".mp4")])
        if vids:
            tests.append((os.path.join(d, vids[0]), cls.lower()))

    print(f"{'expected':12s} | {'A) 30-consecutive':22s} | {'B) arc(90->linspace30)':24s}")
    print("-" * 66)
    a_ok = b_ok = 0
    for path, exp in tests:
        feats = extract_all(path)
        ca = strat_consecutive(feats, model, labels, stride=1)
        cb = strat_arc(feats, model, labels, buf_len=90, stride=3)
        top_a = ca.most_common(1)[0][0] if ca else "-"
        top_b = cb.most_common(1)[0][0] if cb else "-"
        a_ok += int(top_a == exp)
        b_ok += int(top_b == exp)
        print(f"{exp:12s} | {top_a:22s} | {top_b:24s}")
    print("-" * 66)
    print(f"Dominant-prediction accuracy:  A) {a_ok}/{len(tests)}   B) {b_ok}/{len(tests)}")


if __name__ == "__main__":
    main()
