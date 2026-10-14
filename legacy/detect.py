from collections import deque
import os
import cv2
import mediapipe as mp
import numpy as np
import torch

from model import load_model

SEQUENCE_LENGTH = 30


def get_landmarks(results):
    if results.multi_hand_landmarks:
        #for one hand only
        hand = results.multi_hand_landmarks[0]
        values = []
        for landmark in hand.landmark:
            values.extend([landmark.x, landmark.y, landmark.z])
        return np.array(values, dtype=np.float32)

    return np.zeros(63, dtype=np.float32)


def main():
    if not os.path.exists("model.pth"):
        print("model.pth not found. Train the model first.")
        return

    model, labels = load_model("model.pth")
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    prediction = "Waiting..."
    confidence = 0.0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        sequence.append(get_landmarks(results))

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.array(sequence, dtype=np.float32)

            if np.count_nonzero(input_data) < 10:
                prediction = "idle"
                confidence = 1.0
            else:
                input_tensor = torch.tensor(input_data).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]

                class_index = torch.argmax(probabilities).item()
                prediction = labels[class_index]
                confidence = probabilities[class_index].item()

        cv2.rectangle(frame, (10, 10), (450, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {prediction}", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
