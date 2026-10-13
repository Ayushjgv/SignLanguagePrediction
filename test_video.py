"""
Test sign language detection on video files with real-time skeleton visualization and normalized predictions.

Features:
- Aspect-ratio & scale invariant feature extraction via landmark_utils
- Live video window with neon skeleton bones (pose in green, hands in yellow/orange)
- Real-time rolling sign prediction updating live as the video plays
- Whole-video gesture classification summary
- Space: Pause/Resume, Q/ESC: Quit, C: Clear buffer
"""

import argparse
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_utils import extract_normalized_features, draw_skeleton, TOTAL_FEATURES
from video_model import load_model

SEQUENCE_LENGTH = 30


def test_video(
    video_path,
    model_path="video_model.pth",
    show=True,
    output_video_path=None,
    confidence_threshold=0.3
):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return None

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Train the model first.")
        return None

    print(f"Loading model: {model_path} ...")
    model, labels = load_model(model_path)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = max(1, int(1000 / fps))

    print(f"\nLoaded Video: {video_path}")
    print(f"Resolution: {width}x{height} | Total Frames: {total_frames} | FPS: {fps:.2f}")

    # Read all frames into memory
    print("Reading video frames...")
    raw_frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        raw_frames.append(f)
    cap.release()

    full_len = len(raw_frames)
    if full_len == 0:
        print("Error: No frames could be read.")
        return None

    # Initialize MediaPipe models
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    # 1. Whole-Video Prediction (Sample 30 frames across full gesture duration)
    print("Extracting normalized sequence for whole-gesture classification...")
    sample_indices = np.linspace(0, full_len - 1, SEQUENCE_LENGTH, dtype=int)
    cached_feats = {}
    sampled_seq = []
    for s_idx in sample_indices:
        if s_idx not in cached_feats:
            feats, _, _ = extract_normalized_features(raw_frames[s_idx], pose_detector, hands_detector)
            cached_feats[s_idx] = feats
        sampled_seq.append(cached_feats[s_idx])

    sampled_arr = np.array(sampled_seq, dtype=np.float32)
    with torch.no_grad():
        out_whole = model(torch.tensor(sampled_arr).unsqueeze(0))
        probs_whole = torch.softmax(out_whole, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs_whole))
    whole_pred = labels[top_idx]
    whole_conf = float(probs_whole[top_idx])

    top3_indices = np.argsort(probs_whole)[::-1][:min(3, len(labels))]
    top3_results = [(labels[idx], float(probs_whole[idx])) for idx in top3_indices]

    print("\n" + "=" * 60)
    print("WHOLE-GESTURE PREDICTION:")
    print(f"  PREDICTED SIGN: {whole_pred.upper()} ({whole_conf * 100:.2f}%)")
    for r, (l, s) in enumerate(top3_results, 1):
        print(f"  {r}. {l:15s}: {s * 100:.2f}%")
    print("=" * 60 + "\n")

    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Will save output video to: {output_video_path}")

    # 2. Live Playback with Skeleton Bones and Rolling Predictions
    rolling_buffer = deque(maxlen=SEQUENCE_LENGTH)
    stride_step = max(1, full_len // SEQUENCE_LENGTH)
    strided_buffer = deque(maxlen=SEQUENCE_LENGTH)

    live_pred = whole_pred
    live_conf = whole_conf
    live_top3 = top3_results

    paused = False
    f_idx = 0

    print("Playing video with live bone tracking and predictions...")
    print("Controls: [Space] Pause/Resume | [Q/ESC] Quit | [C] Clear buffer\n")

    while f_idx < full_len:
        if not paused:
            frame = raw_frames[f_idx]
            display_frame = frame.copy()

            # Process frame with normalized features & get raw landmark objects for drawing
            feats, p_res, h_res = extract_normalized_features(frame, pose_detector, hands_detector)

            # Draw bones
            draw_skeleton(display_frame, p_res, h_res)

            # Rolling prediction
            rolling_buffer.append(feats)
            if f_idx % stride_step == 0:
                strided_buffer.append(feats)

            active_buf = strided_buffer if len(strided_buffer) >= 15 else rolling_buffer
            if len(active_buf) >= 15:
                buf_list = list(active_buf)
                indices_interp = np.linspace(0, len(buf_list) - 1, SEQUENCE_LENGTH, dtype=int)
                eval_seq = np.array([buf_list[i] for i in indices_interp], dtype=np.float32)

                if np.count_nonzero(eval_seq[:, 99:]) > 50:
                    with torch.no_grad():
                        out_live = model(torch.tensor(eval_seq).unsqueeze(0))
                        p_live = torch.softmax(out_live, dim=1)[0].cpu().numpy()

                    top_k = int(np.argmax(p_live))
                    live_pred = labels[top_k]
                    live_conf = float(p_live[top_k])

                    top3_idx = np.argsort(p_live)[::-1][:min(3, len(labels))]
                    live_top3 = [(labels[i], float(p_live[i])) for i in top3_idx]

            # Render HUD
            hud_w = 460
            hud_h = 135 + (len(live_top3) * 22)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.78, display_frame, 0.22, 0, display_frame)
            cv2.rectangle(display_frame, (15, 15), (15 + hud_w, 15 + hud_h), (0, 220, 110), 2)

            progress = (f_idx + 1) / full_len
            cv2.putText(display_frame, f"Frame: {f_idx + 1}/{full_len} ({progress * 100:.0f}%)", (30, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            bar_w = 200
            cv2.rectangle(display_frame, (30, 50), (30 + bar_w, 58), (60, 60, 60), -1)
            cv2.rectangle(display_frame, (30, 50), (30 + int(bar_w * progress), 58), (0, 220, 110), -1)

            p_color = (0, 255, 128) if live_conf >= 0.5 else (100, 220, 255)
            cv2.putText(display_frame, f"Sign: {live_pred.upper()}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, p_color, 2)
            cv2.putText(display_frame, f"Confidence: {live_conf * 100:.1f}%", (30, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

            if live_top3:
                cv2.line(display_frame, (30, 134), (15 + hud_w - 20, 134), (70, 70, 70), 1)
                for idx, (cand_lbl, cand_score) in enumerate(live_top3):
                    y_pos = 155 + (idx * 22)
                    cv2.putText(display_frame, f"{cand_lbl:14s}", (30, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                    c_bar = int(cand_score * 150)
                    cv2.rectangle(display_frame, (170, y_pos - 10), (170 + 150, y_pos), (50, 50, 50), -1)
                    cv2.rectangle(display_frame, (170, y_pos - 10), (170 + c_bar, y_pos), (0, 200, 255), -1)
                    cv2.putText(display_frame, f"{cand_score * 100:4.1f}%", (330, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.putText(display_frame, f"Whole-Clip Final: {whole_pred.upper()} ({whole_conf*100:.1f}%)",
                        (30, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if writer:
                writer.write(display_frame)

            if show:
                cv2.imshow("Sign Language Video Test", display_frame)

            f_idx += 1

        if show:
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("c"):
                rolling_buffer.clear()
                strided_buffer.clear()

    # Final Summary Card
    if show:
        final_frame = raw_frames[-1].copy()
        banner_w = min(600, width - 40)
        overlay = final_frame.copy()
        cv2.rectangle(overlay, (20, height // 2 - 100), (20 + banner_w, height // 2 + 100), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, final_frame, 0.15, 0, final_frame)
        cv2.rectangle(final_frame, (20, height // 2 - 100), (20 + banner_w, height // 2 + 100), (0, 255, 128), 3)

        cv2.putText(final_frame, "VIDEO ANALYSIS COMPLETE", (40, height // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(final_frame, f"DETECTED SIGN: {whole_pred.upper()}", (40, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 128), 3)
        cv2.putText(final_frame, f"CONFIDENCE: {whole_conf * 100:.2f}%", (40, height // 2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_frame, "Press any key or Q to close window", (40, height // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow("Sign Language Video Test", final_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if writer:
        writer.release()
        print(f"Output video saved to: {output_video_path}")

    pose_detector.close()
    hands_detector.close()

    return {
        "prediction": whole_pred,
        "confidence": whole_conf,
        "top3": top3_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test sign language detection on video with normalized features"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="video_model.pth", help="Path to trained model")
    parser.add_argument("--no_show", action="store_true", help="Disable interactive display window")
    parser.add_argument("--output_video", default=None, help="Save annotated output video")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()

    test_video(
        video_path=args.video,
        model_path=args.model,
        show=not args.no_show,
        output_video_path=args.output_video,
        confidence_threshold=args.threshold
    )


if __name__ == "__main__":
    main()
