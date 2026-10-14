"""
Utility to check class counts and labels across:
1. Trained model (video_model.pth)
2. Extracted training data (data_video/ or data/)
3. Original video source (Video_Dataset/)
"""

import os
import argparse
import torch

def inspect_classes(model_path="models/video_model.pth", data_folder="data_video", video_folder="Video_Dataset"):
    print("=" * 65)
    print("SIGN LANGUAGE CLASS INSPECTOR")
    print("=" * 65)

    # 1. Check Trained Model Checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        labels = checkpoint.get("labels", [])
        print(f"\n[1] TRAINED MODEL: '{model_path}'")
        print(f"    Total Classes Trained: {len(labels)}")
        print(f"    Architecture:          {checkpoint.get('model_type', 'lstm').upper()}")
        print("    Class Labels:")
        for i, lbl in enumerate(labels, 1):
            print(f"      {i:2d}. {lbl}")
    else:
        print(f"\n[1] TRAINED MODEL: '{model_path}' NOT FOUND (train first)")

    # 2. Check Extracted Data Folder
    if os.path.exists(data_folder):
        subdirs = sorted([d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))])
        valid_dirs = []
        for d in subdirs:
            count = len([f for f in os.listdir(os.path.join(data_folder, d)) if f.endswith(".npy")])
            if count > 0:
                valid_dirs.append((d, count))

        print(f"\n[2] EXTRACTED DATA: '{data_folder}'")
        print(f"    Total Classes Available: {len(valid_dirs)}")
        total_samples = sum(c for _, c in valid_dirs)
        print(f"    Total Sample Sequences:  {total_samples}")
        print("    Sample Breakdown:")
        for i, (name, count) in enumerate(valid_dirs, 1):
            print(f"      {i:2d}. {name:15s}: {count:3d} samples")
    else:
        print(f"\n[2] EXTRACTED DATA: '{data_folder}' NOT FOUND")

    # 3. Check Source Video_Dataset
    if os.path.exists(video_folder):
        video_classes = sorted([d for d in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, d))])
        print(f"\n[3] RAW SOURCE: '{video_folder}'")
        print(f"    Total Classes in Dataset: {len(video_classes)}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect sign language classes")
    parser.add_argument("--model", default="models/video_model.pth", help="Model file path")
    parser.add_argument("--data", default="data_video", help="Extracted data folder")
    parser.add_argument("--videos", default="Video_Dataset", help="Raw video folder")
    args = parser.parse_args()

    inspect_classes(args.model, args.data, args.videos)
