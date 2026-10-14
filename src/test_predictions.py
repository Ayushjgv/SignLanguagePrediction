"""Quick end-to-end prediction test on real videos from Video_Dataset."""
import os
from evaluate_model import predict_video

# Test the FIRST available video of each class (all are in the trained 20)
test_classes = ["crocodile", "carrot", "bear", "cook", "cauliflower", "cucumber"]

print("=" * 60)
print("END-TO-END VIDEO PREDICTION TEST (first video of each class)")
print("=" * 60)
correct = 0
total = 0
for cls in test_classes:
    src_dir = None
    for d in os.listdir("Video_Dataset"):
        if d.lower().strip() == cls:
            src_dir = os.path.join("Video_Dataset", d)
            break
    if not src_dir:
        print(f"{cls:12s} -> SKIP (no source folder)")
        continue
    vids = sorted([f for f in os.listdir(src_dir)
                   if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])
    if not vids:
        print(f"{cls:12s} -> SKIP (no videos)")
        continue
    path = os.path.join(src_dir, vids[0])
    label, conf = predict_video(path, "models/video_model.pth")
    total += 1
    mark = "OK" if label == cls else "MISMATCH"
    if label == cls:
        correct += 1
    print(f"expected={cls:12s} predicted={label:12s} ({conf*100:5.1f}%)  [{mark}]")
print("=" * 60)
print(f"Accuracy on these samples: {correct}/{total}")
print("=" * 60)


