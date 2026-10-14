"""
Handedness-invariance test.

Goal: prove that a model trained WITH mirror augmentation recognizes the SAME
sign when the signing hand is swapped (right-hand version vs. its horizontal
mirror = synthetic left-hand version), whereas a model trained WITHOUT mirror
augmentation typically does not.

Method (no new recordings needed):
  For several source videos, extract the real feature sequence, then build the
  horizontally-mirrored sequence with mirror_utils.mirror_sequence (this is
  exactly what a left-hand performance of the same sign produces, up to
  MediaPipe noise). Feed both to the model and check that the predicted label
  is identical.

Usage:
    .\\.venv\\Scripts\\python.exe test_handedness.py --model video_model_mirror.pth
    .\\.venv\\Scripts\\python.exe test_handedness.py --model video_model.pth
"""
import os
import argparse
import numpy as np
import cv2
import torch
import mediapipe as mp

from video_model import load_model
from landmark_utils import extract_normalized_features
from mirror_utils import mirror_sequence

SEQ = 30


def get_feats(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    feats = [extract_normalized_features(f, pose, hands)[0] for f in frames]
    pose.close()
    hands.close()
    return np.array(feats, dtype=np.float32)


def predict(arr, model, labels):
    idxs = np.linspace(0, len(arr) - 1, SEQ, dtype=int)
    with torch.no_grad():
        out = model(torch.tensor(arr[idxs]).unsqueeze(0))
        p = torch.softmax(out, dim=1)[0].cpu().numpy()
    i = int(np.argmax(p))
    return labels[i], float(p[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/video_model_mirror.pth")
    ap.add_argument("--classes", nargs="+",
                    default=["crocodile", "carrot", "cook", "bear", "tiger", "hello"])
    args = ap.parse_args()

    model, labels = load_model(args.model)
    model.eval()
    print(f"Model: {args.model} ({len(labels)} classes)")
    print("=" * 66)
    print(f"{'class':12s} {'orig->pred':>14s} {'mirror->pred':>16s} {'match':>7s}")
    print("-" * 66)

    match = 0
    total = 0
    for cls in args.classes:
        d = None
        for name in os.listdir("Video_Dataset"):
            if name.lower().strip() == cls:
                d = os.path.join("Video_Dataset", name)
                break
        if not d:
            continue
        vids = sorted([x for x in os.listdir(d) if x.lower().endswith(".mp4")])
        if not vids:
            continue
        arr = get_feats(os.path.join(d, vids[0]))
        p_orig, c_orig = predict(arr, model, labels)
        p_mir, c_mir = predict(mirror_sequence(arr), model, labels)
        ok = (p_orig == p_mir)
        total += 1
        match += int(ok)
        print(f"{cls:12s} {p_orig+' '+format(c_orig*100,'.0f')+'%':>14s} "
              f"{p_mir+' '+format(c_mir*100,'.0f')+'%':>16s} {'OK' if ok else 'DIFF':>7s}")

    print("-" * 66)
    print(f"Handedness-invariant matches: {match}/{total}")


if __name__ == "__main__":
    main()
