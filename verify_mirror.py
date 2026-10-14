"""
Empirically determine the correct horizontal-mirror transform for our
225-dim feature layout, by comparing:
    F_orig = features(original frame)
    F_flip = features(cv2.flip(frame, 1))   # ground-truth mirrored scene

We then test candidate transforms of F_orig against F_flip to learn whether
the pose block needs (a) just x-negation, or (b) x-negation + left/right
landmark-index swap, and whether the hand blocks must swap.

Prints the best-matching relationship with errors.
"""
import os
import cv2
import numpy as np
import mediapipe as mp

from landmark_utils import extract_normalized_features

SEQ_UNUSED = None


def build_detectors():
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    return pose, hands


POSE_X = np.arange(0, 99, 3)          # 0,3,...,96
LEFT_X = np.arange(99, 162, 3)        # within [99:162]
RIGHT_X = np.arange(162, 225, 3)      # within [162:225]

# MediaPipe Pose left/right landmark index pairs (anatomical).
POSE_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),      # eyes/ears/mouth
    (11, 12), (13, 14), (15, 16),                  # shoulders/elbows/wrists
    (17, 18), (19, 20), (21, 22),                  # pinky/index/thumb
    (23, 24), (25, 26), (27, 28),                  # hips/knees/ankles
    (29, 30), (31, 32),                            # heels/foot index
]


def mirror_pose_noswap(f):
    g = f.copy()
    g[POSE_X] = -f[POSE_X]
    return g


def mirror_pose_swap(f):
    g = f.copy()
    g[POSE_X] = -f[POSE_X]
    # swap landmark triplets for anatomical L/R pairs
    for a, b in POSE_SWAP_PAIRS:
        ga = g[3 * a:3 * a + 3].copy()
        gb = g[3 * b:3 * b + 3].copy()
        g[3 * a:3 * a + 3] = gb
        g[3 * b:3 * b + 3] = ga
    return g


def negate_x_block(block):
    b = block.copy()
    b[0::3] = -b[0::3]
    return b


def find_frame_with_hand(pose, hands, video_path, want="right"):
    """Return (frame, feats) for a frame where the wanted hand is clearly present."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    for f in frames[::3]:
        feats, _, _ = extract_normalized_features(f, pose, hands)
        left_nz = np.count_nonzero(feats[99:162])
        right_nz = np.count_nonzero(feats[162:225])
        if want == "right" and right_nz > 30 and left_nz < 10:
            return f, feats
        if want == "left" and left_nz > 30 and right_nz < 10:
            return f, feats
    return None, None


def err(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    pose, hands = build_detectors()

    # pick a class video likely to have a clear single hand
    sample = None
    for cls in ["Crocodile", "Carrot", "Cook", "Bear"]:
        d = os.path.join("Video_Dataset", cls)
        if os.path.isdir(d):
            vids = sorted([x for x in os.listdir(d) if x.lower().endswith(".mp4")])
            if vids:
                sample = os.path.join(d, vids[0])
                break
    print("Sample video:", sample)

    frame, F = find_frame_with_hand(pose, hands, sample, want="right")
    if frame is None:
        print("No clear single-right-hand frame found; trying any hand frame.")
        cap = cv2.VideoCapture(sample)
        ret, frame = cap.read()
        cap.release()
        F, _, _ = extract_normalized_features(frame, pose, hands)

    flipped = cv2.flip(frame, 1)
    Fflip, _, _ = extract_normalized_features(flipped, pose, hands)

    print("\n--- Detection presence (nonzero counts) ---")
    print(f"F     : pose={np.count_nonzero(F[:99]):4d} left={np.count_nonzero(F[99:162]):4d} right={np.count_nonzero(F[162:225]):4d}")
    print(f"Fflip : pose={np.count_nonzero(Fflip[:99]):4d} left={np.count_nonzero(Fflip[99:162]):4d} right={np.count_nonzero(Fflip[162:225]):4d}")

    print("\n--- HANDS: does Fflip.left == negx(F.right) and Fflip.right == negx(F.left)? ---")
    e_left_from_right = err(Fflip[99:162], negate_x_block(F[162:225]))
    e_right_from_left = err(Fflip[162:225], negate_x_block(F[99:162]))
    e_left_from_left = err(Fflip[99:162], negate_x_block(F[99:162]))
    print(f"  Fflip.left  vs negx(F.right): {e_left_from_right:.4f}   <- small = HANDS SWAP is correct")
    print(f"  Fflip.right vs negx(F.left) : {e_right_from_left:.4f}   <- small = HANDS SWAP is correct")
    print(f"  Fflip.left  vs negx(F.left) : {e_left_from_left:.4f}   <- small = NO swap (blocks stay)")

    print("\n--- POSE: which transform of F.pose matches Fflip.pose? ---")
    e_noswap = err(Fflip[:99], mirror_pose_noswap(F[:99]))
    e_swap = err(Fflip[:99], mirror_pose_swap(F[:99]))
    print(f"  negate-x only            : {e_noswap:.4f}")
    print(f"  negate-x + L/R index swap: {e_swap:.4f}   <- smaller one is correct")

    print("\n--- Double-mirror identity check on hands (negate twice = original) ---")
    once = negate_x_block(F[162:225])
    twice = negate_x_block(once)
    print(f"  max|negx(negx(block)) - block| = {np.max(np.abs(twice - F[162:225])):.6f}")

    pose.close()
    hands.close()


if __name__ == "__main__":
    main()
