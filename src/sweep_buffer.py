"""
Sweep realtime buffer sizes to find the best configuration (headless).

Extracts each video's features ONCE, then evaluates several buffer lengths for
the "arc" strategy (long buffer -> linspace-sample 30 frames -> model).

Usage:
    .\\.venv\\Scripts\\python.exe sweep_buffer.py
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
BUF_LENS = [30, 45, 60, 90, 120]


def extract_all(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.flip(f, 1))
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


def strat_arc(feats, model, labels, buf_len, stride=3):
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
    return counts.most_common(1)[0][0] if counts else "-"


def main():
    model, labels = load_model(MODEL)
    model.eval()

    tests = []
    for cls in ["Crocodile", "Carrot", "Cook", "Hello", "Tiger", "Bear", "Drink"]:
        d = os.path.join("Video_Dataset", cls)
        vids = sorted([x for x in os.listdir(d) if x.lower().endswith(".mp4")])
        if vids:
            tests.append((os.path.join(d, vids[0]), cls.lower()))

    header = "expected".ljust(12) + " | " + " | ".join(f"buf{b:<4d}" for b in BUF_LENS)
    print(header)
    print("-" * len(header))
    scores = {b: 0 for b in BUF_LENS}
    for path, exp in tests:
        feats = extract_all(path)
        row = exp.ljust(12) + " | "
        cells = []
        for b in BUF_LENS:
            top = strat_arc(feats, model, labels, b)
            ok = (top == exp)
            scores[b] += int(ok)
            cells.append((top + ("*" if ok else "")).ljust(8))
        print(row + " | ".join(cells))
    print("-" * len(header))
    print("accuracy:  " + "  ".join(f"buf{b}={scores[b]}/{len(tests)}" for b in BUF_LENS))
    print("(* = correct)")


if __name__ == "__main__":
    main()
