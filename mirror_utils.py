"""
Safe horizontal-mirror (left/right hand) augmentation for the
pose + two-hands 225-dim landmark features.

Feature layout (must match landmark_utils.extract_normalized_features):
    [  0 :  99 ]  body pose   -> 33 landmarks * (x, y, z)
    [ 99 : 162 ]  LEFT hand   -> 21 landmarks * (x, y, z)
    [162 : 225 ]  RIGHT hand  -> 21 landmarks * (x, y, z)

A horizontal mirror of the scene:
  * negates the horizontal (x) coordinate of every landmark,
  * leaves vertical (y) and depth (z) unchanged,
  * swaps anatomical LEFT and RIGHT landmarks (a person's left hand appears
    on the right side of a mirrored image and MediaPipe re-labels it as the
    right hand, and vice-versa),
  * therefore swaps the LEFT-hand and RIGHT-hand feature blocks.

This transform was verified empirically against cv2.flip(frame, 1) followed by
re-extraction (see verify_mirror.py): swapping the hand blocks and the pose
L/R landmark pairs matches the re-extracted mirrored features far better than
negating x alone.

Mirroring is an INVOLUTION: mirror(mirror(x)) == x (both x-negation and the
block/landmark swaps are their own inverse).

IMPORTANT: mirroring is applied to TRAINING samples only, and only for labels
whose meaning is known to be handedness-invariant. Directional or asymmetric
signs must be excluded via the allowlist.
"""

import numpy as np

POSE_DIM = 99          # 33 * 3
HAND_DIM = 63          # 21 * 3
TOTAL_DIM = 225        # 99 + 63 + 63
LEFT_SLICE = slice(POSE_DIM, POSE_DIM + HAND_DIM)     # [99:162]
RIGHT_SLICE = slice(POSE_DIM + HAND_DIM, TOTAL_DIM)   # [162:225]

# x-coordinate indices within the pose block: 0, 3, 6, ..., 96
POSE_X_INDICES = np.arange(0, POSE_DIM, 3)

# MediaPipe BlazePose anatomical left/right landmark index pairs.
# Index 0 (nose) lies on the mid-sagittal line and is not swapped.
POSE_LR_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6),      # eye inner / eye / eye outer
    (7, 8),                      # ears
    (9, 10),                     # mouth left / right
    (11, 12),                    # shoulders
    (13, 14),                    # elbows
    (15, 16),                    # wrists
    (17, 18),                    # pinky
    (19, 20),                    # index
    (21, 22),                    # thumb
    (23, 24),                    # hips
    (25, 26),                    # knees
    (27, 28),                    # ankles
    (29, 30),                    # heels
    (31, 32),                    # foot index
]


def _negate_x_block(block):
    """Negate the x coordinate (every 3rd value along the LAST axis)."""
    out = block.copy()
    out[..., 0::3] = -out[..., 0::3]
    return out


def mirror_sequence(seq):
    """
    Apply the horizontal-mirror transform to a landmark sequence.

    Args:
        seq: np.ndarray of shape (T, 225) or (225,). dtype float.
    Returns:
        Mirrored array of the same shape and dtype.
    """
    single = False
    if seq.ndim == 1:
        seq = seq[None, :]
        single = True

    if seq.shape[-1] != TOTAL_DIM:
        raise ValueError(f"Expected last dim {TOTAL_DIM}, got {seq.shape[-1]}")

    out = seq.copy()

    # --- Hands: swap LEFT and RIGHT blocks, negating x within each ---
    # Copy both blocks first: they are views into `out`, so writing one block
    # before reading the other would corrupt it (aliasing).
    left = out[..., LEFT_SLICE].copy()
    right = out[..., RIGHT_SLICE].copy()
    out[..., LEFT_SLICE] = _negate_x_block(right)
    out[..., RIGHT_SLICE] = _negate_x_block(left)

    # --- Pose: negate x, then swap anatomical L/R landmark triplets ---
    pose = out[..., :POSE_DIM]
    pose[..., POSE_X_INDICES] = -pose[..., POSE_X_INDICES]
    for a, b in POSE_LR_SWAP_PAIRS:
        ta = pose[..., 3 * a:3 * a + 3].copy()
        tb = pose[..., 3 * b:3 * b + 3].copy()
        pose[..., 3 * a:3 * a + 3] = tb
        pose[..., 3 * b:3 * b + 3] = ta

    return out[0] if single else out


# ---------------------------------------------------------------------------
# Per-class mirror safety allowlist.
#
# Mirror augmentation is only valid for signs whose meaning does NOT depend on
# which hand is used or on left/right direction. Mirroring a directional sign
# (e.g. "left" vs "right", or a sign that points a specific way) would create a
# *mislabeled* training example and hurt the model.
#
# Policy:
#   * If a class is explicitly listed in MIRROR_EXCLUDE, it is NEVER mirrored.
#   * If MIRROR_ALLOWLIST is non-empty, ONLY those classes are mirrored
#     (opt-in / conservative mode).
#   * If MIRROR_ALLOWLIST is empty, all classes are mirrored EXCEPT those in
#     MIRROR_EXCLUDE (opt-out mode). This is the default because, for this
#     single-signer dataset, every sign is recorded with the dominant hand and
#     has no directional left/right contrast, so mirroring teaches the model
#     "same sign, other hand" -- exactly the desired handedness invariance.
#
# For a multi-signer dataset that contains genuinely directional signs, switch
# to opt-in mode by populating MIRROR_ALLOWLIST with the safe labels.
# ---------------------------------------------------------------------------

# Signs known/assumed to be handedness-invariant for THIS dataset.
# Empty => allow all (except MIRROR_EXCLUDE) when in opt-out mode.
MIRROR_ALLOWLIST = set()

# Signs that must never be mirrored because their meaning is directional or
# otherwise changes under reflection. Add labels here if you collect data where
# handedness/direction carries meaning.
MIRROR_EXCLUDE = set()


def is_mirrorable(label):
    """Return True if a training sample with this label may be mirrored."""
    label = str(label).lower().strip()
    if label in MIRROR_EXCLUDE:
        return False
    if MIRROR_ALLOWLIST:
        return label in MIRROR_ALLOWLIST
    return True  # opt-out mode: everything not excluded is mirrorable

