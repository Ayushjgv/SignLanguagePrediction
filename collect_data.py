import argparse
import os
import cv2
import mediapipe as mp
import numpy as np

SEQUENCE_LENGTH = 30
LANDMARKS_PER_HAND = 63
TOTAL_FEATURES = 126

def get_landmarks(results):
# Always store:
# [Left Hand 63 values] + [Right Hand 63 values]
left_hand = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)
right_hand = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)

```
if results.multi_hand_landmarks and results.multi_handedness:
    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):
        values = []

        for landmark in hand_landmarks.landmark:
            values.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        values = np.array(values, dtype=np.float32)

        hand_label = handedness.classification[0].label

        if hand_label == "Left":
            left_hand = values
        elif hand_label == "Right":
            right_hand = values

return np.concatenate([left_hand, right_hand])
```

def main():
parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="Class name, for example hello")
parser.add_argument(
"--samples",
type=int,
default=50,
help="Number of sequences to collect"
)

```
args = parser.parse_args()

label = args.label.lower().strip()
save_folder = os.path.join("data", label)
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []
saved = 0
recording = False

print("Controls: R = start/stop recording, Q = quit")
print(f"Collecting class: {label}")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if recording:
        sequence.append(get_landmarks(results))

        cv2.putText(
            frame,
            f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if len(sequence) == SEQUENCE_LENGTH:
            filename = os.path.join(
                save_folder,
                f"{saved}.npy"
            )

            np.save(
                filename,
                np.array(sequence, dtype=np.float32)
            )

            saved += 1
            sequence = []

            print(f"Saved {filename}")

            if saved >= args.samples:
                print("Collection completed.")
                break

    cv2.putText(
        frame,
        f"Class: {label} | Saved: {saved}/{args.samples}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if not recording:
        cv2.putText(
            frame,
            "Press R to record",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        recording = not recording
        sequence = []

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
```

if **name** == "**main**":
main()
