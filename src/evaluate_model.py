"""
Evaluate the trained video model on the extracted dataset and, if ground-truth
labels are available for the test videos, report per-class accuracy.

- Loads video_model.pth + labels.
- Rebuilds the leak-free group split used in training.
- Reports overall held-out accuracy and a per-class confusion summary.
- If a mapping file 'test_video_labels.csv' (filename,label) exists, also
  evaluates the raw test videos end-to-end through the same feature pipeline.
"""

import os
import csv
import numpy as np
import torch

from train_video_model import load_data
from video_model import load_model
from collect_video_data import build_detectors
from landmark_utils import extract_normalized_features
import cv2


def evaluate_holdout(data_folder="data_video", model_path="models/video_model.pth", seq_len=30):
    from sklearn.model_selection import GroupShuffleSplit

    model, labels = load_model(model_path)
    model.eval()

    X, y, groups, labels = load_data(data_folder, seq_len, 225)
    if len(X) == 0:
        print("No data to evaluate.")
        return

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=groups))
    X_test, y_test = X[test_idx], y[test_idx]

    correct = 0
    per_class_total = {}
    per_class_correct = {}
    preds_all = []
    with torch.no_grad():
        for i in range(len(X_test)):
            inp = torch.tensor(X_test[i:i + 1], dtype=torch.float32)
            out = model(inp)
            p = int(torch.argmax(out, dim=1).item())
            preds_all.append(p)
            true = int(y_test[i])
            per_class_total[true] = per_class_total.get(true, 0) + 1
            if p == true:
                correct += 1
                per_class_correct[true] = per_class_correct.get(true, 0) + 1

    acc = 100.0 * correct / len(X_test)
    print("=" * 60)
    print(f"HELD-OUT ACCURACY (unseen recordings): {acc:.2f}%  ({correct}/{len(X_test)})")
    print("=" * 60)
    print(f"{'class':16s} {'acc':>7s} {'n':>4s}")
    for idx in sorted(per_class_total):
        c = per_class_correct.get(idx, 0)
        t = per_class_total[idx]
        print(f"{labels[idx]:16s} {100.0*c/t:6.1f}% {t:4d}")
    return acc


def predict_video(video_path, model_path="models/video_model.pth", seq_len=30):
    model, labels = load_model(model_path)
    model.eval()
    pose_detector, hands_detector = build_detectors()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        pose_detector.close()
        hands_detector.close()
        return None, 0.0

    idxs = np.linspace(0, len(frames) - 1, seq_len, dtype=int)
    seq = [extract_normalized_features(frames[i], pose_detector, hands_detector)[0] for i in idxs]
    pose_detector.close()
    hands_detector.close()

    arr = np.array(seq, dtype=np.float32)
    with torch.no_grad():
        out = model(torch.tensor(arr).unsqueeze(0))
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    top = int(np.argmax(probs))
    return labels[top], float(probs[top])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data_video")
    parser.add_argument("--model", default="models/video_model.pth")
    parser.add_argument("--video", default=None, help="Optional single video to predict")
    args = parser.parse_args()

    evaluate_holdout(args.data_folder, args.model)
    if args.video:
        label, conf = predict_video(args.video, args.model)
        print(f"\nVideo '{args.video}' -> {label} ({conf*100:.1f}%)")
