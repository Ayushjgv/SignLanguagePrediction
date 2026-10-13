"""
Shared Landmark Extraction & Normalization Utilities.

Solves:
1. Aspect-Ratio Distortion: Corrects for difference between 16:9 landscape and 9:16 portrait videos.
2. Position Invariance: Normalizes all landmarks relative to the body center (mid-shoulder).
3. Scale Invariance: Scales all body and hand landmarks by the true physical shoulder distance in pixels.

Input: Image frame (any resolution, any aspect ratio)
Output: 225 normalized features invariant to camera distance, position, and video aspect ratio.
"""

import cv2
import mediapipe as mp
import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS_PER_HAND = 21

TOTAL_POSE_FEATURES = POSE_LANDMARKS * 3  # 99
TOTAL_HAND_FEATURES = HAND_LANDMARKS_PER_HAND * 3 * 2  # 126
TOTAL_FEATURES = TOTAL_POSE_FEATURES + TOTAL_HAND_FEATURES  # 225


def extract_normalized_features(frame, pose_detector, hands_detector):
    """
    Extracts 225 features that are scale, position, and aspect-ratio invariant.

    Returns:
        features: np.ndarray of shape (225,), dtype=np.float32
        pose_results: raw MediaPipe pose object
        hands_results: raw MediaPipe hands object
    """
    h_img, w_img, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res = pose_detector.process(rgb)
    hands_res = hands_detector.process(rgb)

    # 1. Determine Body Reference Point (Mid-Shoulder) & True Pixel Scale
    # Default center and scale if no pose is found
    mid_x_px = w_img * 0.5
    mid_y_px = h_img * 0.35
    scale_px = float(w_img * 0.25)

    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks.landmark
        l_sh = lms[11]  # Left shoulder
        r_sh = lms[12]  # Right shoulder

        lx_px = l_sh.x * w_img
        ly_px = l_sh.y * h_img
        rx_px = r_sh.x * w_img
        ry_px = r_sh.y * h_img

        shoulder_dist = np.sqrt((lx_px - rx_px) ** 2 + (ly_px - ry_px) ** 2)
        if shoulder_dist > 15.0:  # Valid shoulder detection
            scale_px = shoulder_dist
            mid_x_px = (lx_px + rx_px) / 2.0
            mid_y_px = (ly_px + ry_px) / 2.0
        elif lms[0].visibility > 0.5:  # Fallback to nose
            mid_x_px = lms[0].x * w_img
            mid_y_px = lms[0].y * h_img + (h_img * 0.1)

    # 2. Extract Normalized Pose Landmarks (33 x 3 = 99)
    pose_feat = np.zeros(TOTAL_POSE_FEATURES, dtype=np.float32)
    if pose_res.pose_landmarks:
        for i, lm in enumerate(pose_res.pose_landmarks.landmark):
            # Convert to aspect-corrected offset from mid-shoulder, divided by shoulder distance
            x_norm = ((lm.x * w_img) - mid_x_px) / scale_px
            y_norm = ((lm.y * h_img) - mid_y_px) / scale_px
            z_norm = lm.z  # Relative depth
            pose_feat[i * 3] = x_norm
            pose_feat[i * 3 + 1] = y_norm
            pose_feat[i * 3 + 2] = z_norm

    # 3. Extract Normalized Hand Landmarks (21 x 3 x 2 = 126)
    left_hand = np.zeros(HAND_LANDMARKS_PER_HAND * 3, dtype=np.float32)
    right_hand = np.zeros(HAND_LANDMARKS_PER_HAND * 3, dtype=np.float32)

    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_lms, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            is_left = handedness.classification[0].label == "Left"
            target_arr = left_hand if is_left else right_hand

            # Hand wrist in pixel coordinates
            w_px = hand_lms.landmark[0].x * w_img
            h_px = hand_lms.landmark[0].y * h_img

            # Hand size scale (wrist to middle finger base)
            m_px = hand_lms.landmark[9].x * w_img
            my_px = hand_lms.landmark[9].y * h_img
            hand_scale = np.sqrt((w_px - m_px) ** 2 + (h_px - my_px) ** 2)
            if hand_scale < 5.0:
                hand_scale = scale_px * 0.25

            # Wrist position relative to body
            rel_wrist_x = (w_px - mid_x_px) / scale_px
            rel_wrist_y = (h_px - mid_y_px) / scale_px

            for i, lm in enumerate(hand_lms.landmark):
                if i == 0:
                    # Wrist stores position relative to body
                    target_arr[0] = rel_wrist_x
                    target_arr[1] = rel_wrist_y
                    target_arr[2] = lm.z
                else:
                    # Finger joints store shape relative to wrist, scaled by hand size
                    joint_x = ((lm.x * w_img) - w_px) / hand_scale
                    joint_y = ((lm.y * h_img) - h_px) / hand_scale
                    joint_z = lm.z
                    target_arr[i * 3] = joint_x
                    target_arr[i * 3 + 1] = joint_y
                    target_arr[i * 3 + 2] = joint_z

    combined = np.concatenate([pose_feat, left_hand, right_hand]).astype(np.float32)
    return combined, pose_res, hands_res


def draw_skeleton(frame, pose_res, hands_res):
    """
    Renders aesthetic neon skeleton connections on both body pose and hands.
    """
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Draw body pose
    if pose_res and pose_res.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            pose_res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 128), thickness=3)
        )

    # Draw hands
    if hands_res and hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_lms, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            is_left = handedness.classification[0].label == "Left"
            line_color = (0, 255, 255) if is_left else (255, 140, 0)
            joint_color = (0, 0, 255) if is_left else (255, 0, 140)
            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=joint_color, thickness=2, circle_radius=3),
                connection_drawing_spec=mp_draw.DrawingSpec(color=line_color, thickness=2)
            )
