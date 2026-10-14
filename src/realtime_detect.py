"""
Real-time sign language detection using webcam with normalized pose & hand features.

Features:
- Aspect-ratio, distance, and position invariant feature extraction via landmark_utils
- Rolling 30-frame sequence buffer for continuous real-time classification
- Prediction smoothing across frames to eliminate jitter
- Interactive HUD: live sign name, confidence %, top-3 ranking bars, and FPS
- Hotkeys:
    'q' or ESC : Exit
    'c'        : Clear buffer / reset detection
    'h'        : Toggle landmark skeleton overlay on/off
"""

import argparse
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_utils import extract_normalized_features, draw_skeleton, TOTAL_FEATURES
from video_model import load_model

SEQUENCE_LENGTH = 30


def run_realtime_detection(
    model_path="models/video_model.pth",
    camera_index=0,
    confidence_threshold=0.4,
    smoothing_window=5,
    history_len=75,
    infer_stride=3
):
    """
    history_len  : number of recent frames kept. For each inference we
                   linspace-sample SEQUENCE_LENGTH frames spread across this
                   whole window -- matching how the model was trained (30
                   frames across the full gesture). A short 30-frame buffer
                   only captures ~1s and hurts accuracy; ~60-90 frames (2-3s)
                   works best. Must be >= SEQUENCE_LENGTH.
    infer_stride : run inference every N frames (reduces jitter + CPU load).
    """
    if history_len < SEQUENCE_LENGTH:
        history_len = SEQUENCE_LENGTH

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Train the model first.")
        return

    print(f"Loading model checkpoint from '{model_path}' ...")
    model, labels = load_model(model_path)
    model.eval()
    print(f"Model loaded with {len(labels)} classes: {labels}")

    print(f"Opening webcam (device index {camera_index}) ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Long history buffer: keep HISTORY_LEN recent frames and linspace-sample
    # SEQUENCE_LENGTH of them for each inference (matches training).
    sequence_buffer = deque(maxlen=history_len)
    prediction_history = deque(maxlen=smoothing_window)
    frame_counter = 0

    draw_landmarks = True
    current_prediction = "Waiting for gesture..."
    current_confidence = 0.0
    top3_candidates = []

    fps = 0.0
    frame_times = deque(maxlen=30)
    prev_time = time.time()

    print("\n" + "=" * 60)
    print("REAL-TIME SIGN LANGUAGE DETECTION RUNNING")
    print("=" * 60)
    print("Press 'q' or ESC to Quit | 'c' to Clear buffer | 'h' to Toggle bones")
    print("=" * 60 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        frame_times.append(curr_time - prev_time)
        prev_time = curr_time
        if frame_times:
            fps = len(frame_times) / sum(frame_times)

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Extract normalized features & raw results
        frame_features, pose_res, hands_res = extract_normalized_features(frame, pose_detector, hands_detector)

        if draw_landmarks:
            draw_skeleton(frame, pose_res, hands_res)

        sequence_buffer.append(frame_features)
        frame_counter += 1

        # Inference: only once the history buffer is full, and only every
        # `infer_stride` frames (reduces jitter and CPU load).
        if len(sequence_buffer) == history_len and frame_counter % infer_stride == 0:
            buf_arr = np.array(sequence_buffer, dtype=np.float32)
            # Sample SEQUENCE_LENGTH frames spread across the WHOLE buffer,
            # exactly like training. This is the key realtime accuracy fix.
            sample_idx = np.linspace(0, history_len - 1, SEQUENCE_LENGTH, dtype=int)
            seq_arr = buf_arr[sample_idx]
            hands_activity = np.count_nonzero(seq_arr[:, 99:])

            if hands_activity < 40:
                current_prediction = "Idle (Raise hands to sign)"
                current_confidence = 0.0
                top3_candidates = []
            else:
                input_tensor = torch.tensor(seq_arr).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

                prediction_history.append(probabilities)
                avg_probs = np.mean(prediction_history, axis=0)

                top_idx = int(np.argmax(avg_probs))
                top3_indices = np.argsort(avg_probs)[::-1][:min(3, len(labels))]
                top3_candidates = [(labels[i], float(avg_probs[i])) for i in top3_indices]

                top_confidence = float(avg_probs[top_idx])
                if top_confidence >= confidence_threshold:
                    current_prediction = labels[top_idx]
                    current_confidence = top_confidence
                else:
                    current_prediction = f"Uncertain ({labels[top_idx]})"
                    current_confidence = top_confidence

        # Render Modern HUD
        banner_w = 480
        banner_h = 130 + (len(top3_candidates) * 22)
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (15 + banner_w, 15 + banner_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + banner_w, 15 + banner_h), (0, 220, 110), 2)

        buf_len = len(sequence_buffer)
        buf_pct = buf_len / history_len
        cv2.putText(frame, f"FPS: {fps:4.1f} | Buffer: {buf_len}/{history_len}", (30, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        bar_x, bar_y, bar_w, bar_h = 30, 52, 200, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * buf_pct), bar_y + bar_h), (0, 220, 110), -1)

        pred_color = (0, 255, 128) if current_confidence >= confidence_threshold else (120, 220, 255)
        cv2.putText(frame, f"Sign: {current_prediction.upper()}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, pred_color, 2)
        cv2.putText(frame, f"Confidence: {current_confidence * 100:.1f}%", (30, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        if top3_candidates:
            cv2.line(frame, (30, 134), (15 + banner_w - 20, 134), (80, 80, 80), 1)
            for idx, (cand_label, cand_score) in enumerate(top3_candidates):
                y_offset = 155 + (idx * 22)
                cv2.putText(frame, f"{cand_label:14s}", (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                c_bar_w = int(cand_score * 160)
                cv2.rectangle(frame, (170, y_offset - 10), (170 + 160, y_offset), (50, 50, 50), -1)
                cv2.rectangle(frame, (170, y_offset - 10), (170 + c_bar_w, y_offset), (0, 200, 255), -1)
                cv2.putText(frame, f"{cand_score * 100:4.1f}%", (340, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame, "[Q] Quit | [C] Reset | [H] Landmarks", (w - 380, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Sign Language Real-time Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("c"):
            sequence_buffer.clear()
            prediction_history.clear()
            current_prediction = "Buffer Cleared"
            current_confidence = 0.0
            top3_candidates = []
        elif key == ord("h"):
            draw_landmarks = not draw_landmarks

    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    hands_detector.close()
    print("Real-time detection closed cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time sign language detection with normalized features"
    )
    parser.add_argument("--model", default="models/video_model_mirror.pth", help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--threshold", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--smoothing", type=int, default=5, help="Smoothing window size")
    parser.add_argument("--buffer", type=int, default=75,
                        help="History buffer length in frames (60-90 recommended; ~2-3s). "
                             "30 frames spread are sampled across it for each prediction.")
    parser.add_argument("--stride", type=int, default=3,
                        help="Run inference every N frames (higher = less jitter, less CPU)")

    args = parser.parse_args()

    run_realtime_detection(
        model_path=args.model,
        camera_index=args.camera,
        confidence_threshold=args.threshold,
        smoothing_window=args.smoothing,
        history_len=args.buffer,
        infer_stride=args.stride
    )


if __name__ == "__main__":
    main()
